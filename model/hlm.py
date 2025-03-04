import json
import logging
import os
from types import SimpleNamespace
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers.modeling_outputs import TokenClassifierOutput, QuestionAnsweringModelOutput

from .codepoint_tokenizer import CodepointTokenizer
from .position_embedder import PositionEmbedder
from .bert import BertEncoder

logger = logging.getLogger(__name__)

ACTIVATIONS = {
    'relu': torch.nn.ReLU,
    'gelu': torch.nn.GELU
}


class HLMConfig(SimpleNamespace):
    def to_dict(self):
        out = self.__dict__.copy()
        if 'self' in out:
            del out['self']
        keys_to_ignore = []
        for k, v in out.items():
            try:
                json.dumps({k: v})
            except:
                keys_to_ignore.append(k)
        for k in keys_to_ignore:
            del out[k]
        return out

    def to_json_string(self):
        return json.dumps(self.to_dict())


class CustomTransformerEncoder(nn.Module):
    def __init__(self, config):
        super(CustomTransformerEncoder, self).__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.word_context = config.word_context if hasattr(config, "word_context") else 1
        self.relative_attention = config.relative_attention if hasattr(config, "relative_attention") else False

        self.aggregation_method = config.aggregation_method if hasattr(config, "aggregation_method") else 'word_cls'
        assert self.aggregation_method in ['word_cls', 'mean_pooling', 'max_pooling']
        logger.info("Aggregation method: %s" % self.aggregation_method)
        self.pos_att_type = config.pos_att_type if hasattr(config, "pos_att_type") else "p2c|c2p"

        if config.use_projection:
            self.proj = nn.Linear(2 * self.hidden_size, self.hidden_size)
            self.activation = ACTIVATIONS[config.activation]()
            self.LayerNorm = nn.LayerNorm(config.hidden_size)
            self.dropout = nn.Dropout(config.dropout)
        if config.relative_attention:
            self.local_layer_first = BertEncoder(
                num_hidden_layers=config.n_local_layer_first,
                hidden_size=config.hidden_size,
                hidden_dropout_prob=config.dropout,
                intermediate_size=config.local_transformer_ff_size,
                hidden_act=config.activation,
                max_relative_positions=config.max_char_per_word,
                pos_att_type=self.pos_att_type)
            self.global_layer = BertEncoder(
                num_hidden_layers=config.n_global_layer,
                hidden_size=config.hidden_size,
                hidden_dropout_prob=config.dropout,
                intermediate_size=config.transformer_ff_size,
                hidden_act=config.activation,
                max_relative_positions=config.max_num_word,
                pos_att_type=self.pos_att_type)
        else:
            self.local_layer_first = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=config.hidden_size,
                    nhead=config.attention_heads,
                    dim_feedforward=config.local_transformer_ff_size,
                    dropout=config.dropout,
                    activation=config.activation),
                config.n_local_layer_first)
            self.global_layer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=config.hidden_size,
                    nhead=config.attention_heads,
                    dim_feedforward=config.transformer_ff_size,
                    dropout=config.dropout,
                    activation=config.activation),
                config.n_global_layer)
        if config.n_local_layer_last > 0:
            if config.relative_attention:
                self.local_layer_last = BertEncoder(
                    num_hidden_layers=config.n_local_layer_last,
                    hidden_size=config.hidden_size,
                    hidden_dropout_prob=config.dropout,
                    intermediate_size=config.local_transformer_ff_size,
                    hidden_act=config.activation,
                    max_relative_positions=config.max_char_per_word,
                    pos_att_type=self.pos_att_type)
            else:
                self.local_layer_last = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=config.hidden_size,
                        nhead=config.attention_heads,
                        dim_feedforward=config.local_transformer_ff_size,
                        dropout=config.dropout,
                        activation=config.activation),
                    config.n_local_layer_last)

    def forward(
            self,
            embeddings_matrix,
            key_padding_mask_sentence,
            key_padding_mask_matrix):
        max_char_len, max_num_word, batch_size = embeddings_matrix.shape[:3]
        do_context = self.word_context > 1 and self.word_context <= max_num_word
        if do_context:
            if max_num_word % self.word_context:
                raise RuntimeError("Max number of words must be divisible by word_context: word_contex=%s max_num_word=%s" % (self.word_context, max_num_word))
            max_char_len *= self.word_context
            max_num_word //= self.word_context
            # Permute dimensions to match hugging face conventions (batch dimension first)
            # [max_char_len, max_num_word, batch_size, hidden_size] --> [batch_size, max_num_word, max_char_len, hidden_size]
            embeddings_matrix = embeddings_matrix.permute((2, 1, 0, 3))
            embeddings_matrix = embeddings_matrix.reshape((batch_size * max_num_word, max_char_len, self.hidden_size))
            embeddings_chunk = embeddings_matrix.permute((1, 0, 2))
            # Ignore words made of [PAD] tokens only.
            key_padding_mask_matrix = key_padding_mask_matrix.permute((2, 1, 0))
            key_padding_mask_matrix = key_padding_mask_matrix.reshape((batch_size * max_num_word, max_char_len))
            key_padding_mask_chunk_idx = ~key_padding_mask_matrix[:, 0]
            key_padding_mask_chunk = key_padding_mask_matrix[key_padding_mask_chunk_idx, :]
        else:
            embeddings_chunk = embeddings_matrix.reshape((max_char_len, batch_size * max_num_word, self.hidden_size))
            key_padding_mask_matrix = key_padding_mask_matrix.reshape((max_char_len, batch_size * max_num_word))
            key_padding_mask_chunk_idx = ~key_padding_mask_matrix[0, :]
            key_padding_mask_chunk = key_padding_mask_matrix[:, key_padding_mask_chunk_idx].transpose(0, 1)

        if self.aggregation_method != "word_cls":
            # Mask [WORD_CLS] token
            key_padding_mask_chunk[0, :] = True

        # 1st local transformer
        if self.relative_attention:
            # HF transformers use 1 for tokens that are **not masked**, so we add ~
            # HF transformers use embedding format of (batch_size, sequence_length, hidden_size), so transpose
            embeddings_chunk[:, key_padding_mask_chunk_idx] = self.local_layer_first(
                embeddings_chunk[:, key_padding_mask_chunk_idx].transpose(0, 1),
                attention_mask=~key_padding_mask_chunk).transpose(0, 1).float()
        else:
            embeddings_chunk[:, key_padding_mask_chunk_idx] = self.local_layer_first(embeddings_chunk[:, key_padding_mask_chunk_idx], src_key_padding_mask=key_padding_mask_chunk)

        if do_context:
            max_char_len //= self.word_context
            max_num_word *= self.word_context
            # Restore original dimensions
            # [max_char_len, max_num_word * batch_size, hidden_size] --> [max_char_len, max_num_word, batch_size, hidden_size]
            embeddings_matrix = embeddings_chunk.permute((1, 0, 2))
            embeddings_matrix = embeddings_matrix.reshape((batch_size, max_num_word, max_char_len, self.hidden_size))
            embeddings_matrix = embeddings_matrix.permute((2, 1, 0, 3))

        # Global transformer on [CLS] - [WORD_CLS] - [SEP]
        if self.aggregation_method == 'word_cls':
            # embeddings_matrix_pool: [max_num_word, batch_size, hidden_size]
            embeddings_matrix_pool = embeddings_matrix[0, :, :, :].clone()
        elif self.aggregation_method == 'mean_pooling':
            key_padding_mask_matrix = key_padding_mask_matrix.reshape((max_char_len, max_num_word, batch_size))
            with torch.no_grad():
                word_len = (key_padding_mask_matrix == 0).sum(dim=0).float()
                word_len_idx = word_len > 0
            # [WORD_CLS] ignored
            # 1: masked
            embeddings_matrix_pool = torch.sum(
                embeddings_matrix[1:, :, :, :] * (~key_padding_mask_matrix[1:, :, :, None]), dim=0)
            embeddings_matrix_pool[word_len_idx] = \
                embeddings_matrix_pool[word_len_idx] / word_len[word_len_idx][:, None]
        else:
            # perform max pool
            # out: [max_num_word, batch_size, embed_size]
            embeddings_matrix_pool = torch.max(embeddings_matrix[1:, :, :, :], dim=0)[0]  # returns a tuple

        if self.relative_attention:
            embeddings_matrix_pool = self.global_layer(
                embeddings_matrix_pool.transpose(0, 1),
                attention_mask=~key_padding_mask_sentence).transpose(0, 1)
        else:
            embeddings_matrix_pool = self.global_layer(embeddings_matrix_pool, src_key_padding_mask=key_padding_mask_sentence)

        if self.config.use_projection:
            # concat with projection
            repeated = torch.repeat_interleave(embeddings_matrix_pool.unsqueeze(0), max_char_len - 1, dim=0)
            embeddings_matrix = torch.cat([embeddings_matrix[1:, :, :, :], repeated], dim=-1)
            embeddings_matrix = self.proj(embeddings_matrix)
            embeddings_matrix = self.activation(embeddings_matrix)
            embeddings_matrix = self.LayerNorm(embeddings_matrix)
            embeddings_matrix = self.dropout(embeddings_matrix)
            embeddings_matrix = torch.cat([embeddings_matrix_pool[None, :, :, :], embeddings_matrix], dim=0)
        else:
            # concat connection
            embeddings_matrix = torch.cat([embeddings_matrix_pool[None, :, :, :], embeddings_matrix[1:, :, :, :]], dim=0)

        if self.config.n_local_layer_last > 0:
            if do_context:
                if self.word_context % 2 or max_num_word % 2:
                    raise RuntimeError("Supports even word context and max number of words only: word_contex=%s max_num_word=%s" % (self.word_context, max_num_word))
                max_char_len *= self.word_context
                max_num_word //= self.word_context
                # Permute dimensions to match hugging face conventions (batch dimension first)
                # [max_char_len, max_num_word, batch_size, hidden_size] --> [batch_size, max_num_word, max_char_len, hidden_size]
                embeddings_matrix = embeddings_matrix.permute((2, 1, 0, 3))
                embeddings_matrix = embeddings_matrix.reshape((batch_size * max_num_word, max_char_len, self.hidden_size))
                embeddings_chunk = embeddings_matrix.permute((1, 0, 2))
            else:
                embeddings_chunk = embeddings_matrix.reshape((max_char_len, batch_size * max_num_word, self.hidden_size))

                # 2nd local tranformer for character prediction
                if self.relative_attention:
                    embeddings_chunk[:, key_padding_mask_chunk_idx] = self.local_layer_last(
                        embeddings_chunk[:, key_padding_mask_chunk_idx].transpose(0, 1),
                        attention_mask=~key_padding_mask_chunk).transpose(0, 1).float()
                else:
                    embeddings_chunk[:, key_padding_mask_chunk_idx] = self.local_layer_last(embeddings_chunk[:, key_padding_mask_chunk_idx], src_key_padding_mask=key_padding_mask_chunk)
                embeddings_matrix = embeddings_chunk.reshape((max_char_len, max_num_word, batch_size, self.hidden_size))

            if do_context:
                max_char_len //= self.word_context
                max_num_word *= self.word_context
                # Restore original dimensions
                # [max_char_len, max_num_word * batch_size, hidden_size] --> [max_char_len, max_num_word, batch_size, hidden_size]
                embeddings_matrix = embeddings_chunk.permute((1, 0, 2))
                embeddings_matrix = embeddings_matrix.reshape((batch_size, max_num_word, max_char_len, self.hidden_size))
                embeddings_matrix = embeddings_matrix.permute((2, 1, 0, 3))

        return embeddings_matrix


class HLM(torch.nn.Module):
    
    def __init__(self, config):
        super(HLM, self).__init__()
        self.config = config
        self.word_context = config.word_context if hasattr(config, "word_context") else 1
        self.relative_attention = config.relative_attention if hasattr(config, "relative_attention") else False
        if config.activation not in ACTIVATIONS:
            raise RuntimeError(f'Activation must be in {set(ACTIVATIONS.keys())}, but was {config.activation}')
        else:
            self.activation = ACTIVATIONS[config.activation]()

        self.layer_norm_eps = getattr(config, 'layer_norm_eps', 1e-7)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.dropout)

        self.hidden_size = config.hidden_size
        self.for_token_classification = config.for_token_classification
        self.embedder = torch.nn.Embedding(config.max_codepoint, config.hidden_size)
        if self.config.use_token_type:
            self.type_embedder = torch.nn.Embedding(2, config.hidden_size)

        if not self.relative_attention:
            self.position_embedder = PositionEmbedder(
                config.hidden_size,
                max_num_word=config.max_num_word,
                max_char_per_word=config.max_char_per_word)

        # main transformer stack
        self.deep_transformer = CustomTransformerEncoder(self.config)
        # CLS Token
        self.cls_linear_final = torch.nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self,
                input_ids,
                attention_mask_matrix,
                attention_mask_sentence,
                token_type_ids=None,
                predict_indices=None):
        batch_size = input_ids.shape[-1]
        if any(input_ids[0, 0, :] != CodepointTokenizer.CLS):
            raise RuntimeError('All input sequences must start wit [CLS] codepoint')

        # Character & positional embeddings
        min_id = torch.min(input_ids)
        max_id = torch.max(input_ids)
        if min_id < 0 or max_id >= self.config.max_codepoint:
            logger.warning("Out of range token ID: min=%s max=%s" % (min_id, max_id))
            input_ids[input_ids < 0] = 0
            input_ids[input_ids >= self.config.max_codepoint] = self.config.max_codepoint - 1
        embeddings_matrix = self.embedder(input_ids)
        if not self.relative_attention:
            embeddings_matrix = self.position_embedder(embeddings_matrix)
        if self.config.use_token_type and token_type_ids is not None:
            embeddings_matrix += self.type_embedder(token_type_ids)

        # Normalization and dropout
        embeddings_matrix = self.layer_norm(embeddings_matrix)
        embeddings_matrix = self.dropout(embeddings_matrix)

        embeddings_matrix = self.deep_transformer(embeddings_matrix=embeddings_matrix,
                                                  key_padding_mask_sentence=attention_mask_sentence,
                                                  key_padding_mask_matrix=attention_mask_matrix)

        if predict_indices is not None:
            # this is MLM of some kind - we don't need to do the final CLS computation and we can only do the
            # final transformer for the positions we're predicting
            # predict_indices: [max_char_len, max_num_word, batch_size]
            # embeddings_matrix: [max_char_len, max_num_word, batch_size, embed_size]
            # embeddings_for_pred: [num_mask_char, embed_size]
            embeddings_for_pred = embeddings_matrix[predict_indices]

            return {
                'embeddings': embeddings_for_pred
            }
        elif self.for_token_classification:
            # Select [CLS] and [WORD_CLS] tokens
            # out: [max_num_word, batch_size, embed_size]
            embeddings_matrix_pool = embeddings_matrix[0, :, :, :]

            return {
                'embeddings_pool': embeddings_matrix_pool
            }
        else:
            # Select [CLS] token only
            # out: [batch_size, embed_size]
            contextualized_cls = embeddings_matrix[0, 0, :, :]
            final_cls = self.cls_linear_final(contextualized_cls)
            return {
                'embeddings': final_cls
            }

class HLMForTask(torch.nn.Module):
    def __init__(self, config):
        super(HLMForTask, self).__init__()
        self.hlm_model = HLM(config)

    def load_encoder_checkpoint(self, checkpoint_location: Optional[str] = None):
        if checkpoint_location is None:
            state_dict = get_pretrained_state_dict()
        else:
            state_dict = torch.load(checkpoint_location, map_location=torch.device('cpu'))

        self.hlm_model.load_state_dict(state_dict)

class HLMForSequenceLabeling(HLMForTask):
    def __init__(self, config, vocab_size: int):
        super(HLMForSequenceLabeling, self).__init__(config)
        self.vocab_size = vocab_size
        self.config = config
        self.config.vocab_size = self.vocab_size
        self.label_layer = torch.nn.Linear(self.config.hidden_size, self.vocab_size)
        self.dropout = torch.nn.Dropout(p=self.config.dropout)
        self.log_softmax = torch.nn.LogSoftmax(dim=2)
        self.loss = torch.nn.NLLLoss()

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor],
                attention_mask: torch.Tensor) -> Tuple:
        embeddings = self.hlm_model(input_ids, attention_mask, None)['embeddings']

        label_hidden_states = self.label_layer(self.dropout(embeddings))
        label_probs = self.log_softmax(label_hidden_states)

        output = {
            'embeddings': embeddings,
            'label_probs': label_probs
        }

        if labels is not None:
            output['loss'] = self.loss(label_probs.transpose(1, 2), labels)

        return output.get('loss', None), output['label_probs'], output['embeddings']


class HLMForSequenceClassification(HLMForTask):
    def __init__(self, config, vocab_size: int):
        super(HLMForSequenceClassification, self).__init__(config)
        self.vocab_size = vocab_size
        self.config = config
        self.config.vocab_size = self.vocab_size
        self.activation = nn.Tanh()
        self.label_layer = torch.nn.Linear(self.config.hidden_size, self.vocab_size)
        self.dropout = torch.nn.Dropout(p=self.config.dropout)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        self.loss = torch.nn.NLLLoss()

    def forward(
            self,
            input_ids: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
            attention_mask_matrix: Optional[torch.Tensor] = None,
            attention_mask_sentence: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None) -> Tuple:
        cls_embeddings = self.hlm_model(
            input_ids,
            attention_mask_matrix=attention_mask_matrix,
            attention_mask_sentence=attention_mask_sentence,
            token_type_ids=token_type_ids)['embeddings']

        cls_embeddings = self.activation(cls_embeddings)

        class_hidden_states = self.label_layer(self.dropout(cls_embeddings))
        class_probs = self.log_softmax(class_hidden_states)

        output = {
            'cls_embeddings': cls_embeddings,
            'class_probs': class_probs
        }

        if labels is not None:
            output['loss'] = self.loss(class_probs, labels)

        return output.get('loss', None), class_hidden_states


class HLMForTokenClassification(HLMForTask):
    def __init__(self, config, vocab_size: int, for_token_classification=True):
        # num_labels = vocab_size
        super(HLMForTokenClassification, self).__init__(config)
        self.vocab_size = vocab_size
        self.config = config
        self.config.vocab_size = self.vocab_size
        self.hlm_model.for_token_classification = for_token_classification
        self.label_layer = torch.nn.Linear(self.hlm_model.config.hidden_size, self.vocab_size)
        self.dropout = torch.nn.Dropout(p=self.hlm_model.config.dropout)

        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        self.loss_func = torch.nn.NLLLoss()

    def forward(self,
                input_ids: torch.Tensor,
                labels: Optional[torch.Tensor],
                attention_mask_matrix: torch.Tensor,
                attention_mask_sentence: torch.Tensor,
                token_type_ids: Optional[torch.Tensor] = None) -> Tuple:
        # out: [max_num_word, batch_size, embed_size]
        embeddings_matrix_pool0 = self.hlm_model(
            input_ids,
            attention_mask_matrix=attention_mask_matrix,
            attention_mask_sentence=attention_mask_sentence,
            token_type_ids=token_type_ids)['embeddings_pool']

        # attention_mask_sentence: [batch_size, max_num_word] -> [max_num_word, batch_size]
        attention_mask_sentence = attention_mask_sentence.transpose(0, 1)

        # [max_num_word, batch_size, embed_size] -> [num_valid_word, embed_size]
        embeddings_matrix_pool = embeddings_matrix_pool0[~attention_mask_sentence]

        # [num_valid_word, num_labels]
        class_hidden_states = self.label_layer(self.dropout(embeddings_matrix_pool))
        class_probs = self.log_softmax(class_hidden_states)

        # [max_num_word, batch_size, num_labels]
        class_hidden_states_matrix = torch.ones((attention_mask_sentence.shape[0],
                                                 attention_mask_sentence.shape[1],
                                                 class_hidden_states.shape[1])).to(attention_mask_sentence.device) * (-100)

        class_hidden_states_matrix[~attention_mask_sentence] = class_hidden_states
        # [batch_size, max_num_word, num_labels]
        class_hidden_states_matrix = class_hidden_states_matrix.transpose(0, 1)

        if labels is not None:
            # labels: [batch_size,max_num_word] -> [max_num_word, batch_size] -> [num_valid_word,]
            labels = labels.transpose(0, 1)[~attention_mask_sentence]
            loss = self.loss_func(class_probs, labels)
            if torch.isnan(loss):
                logger.warning("NaN loss detected!")
                loss = torch.zeros((loss.shape), requires_grad=True, device=embeddings_matrix_pool.device)
        else:
            loss = None

        return TokenClassifierOutput(
            loss=loss,
            logits=class_hidden_states_matrix.contiguous()
        )


class HLMForMaskedLanguageModeling(HLMForTask):
    def __init__(self, config, vocab_size=1024):

        super(HLMForMaskedLanguageModeling, self).__init__(config)
        self.vocab_size = vocab_size
        self.config = config
        self.config.vocab_size = self.vocab_size

        self.dense_layer = nn.Linear(self.config.hidden_size, self.config.hidden_size)

        self.activation = self.hlm_model.activation
        self.layer_norm = nn.LayerNorm(self.config.hidden_size)

        self.lm_layer = nn.Linear(self.config.hidden_size, self.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(self.vocab_size))
        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.lm_layer.bias = self.bias

        self.loss = nn.CrossEntropyLoss()

    def _compute_loss(self,
                      embeddings: torch.Tensor,
                      char_probs: torch.Tensor,
                      predict_indices: torch.Tensor,
                      labels: Optional[torch.Tensor]) -> Tuple:
        output = {
            'embeddings': embeddings,
            'char_probs': char_probs
        }

        if labels is not None:
            # labels: [batch_size, num_mask_char] -> [batch_size*num_mask_char]
            labels = labels.reshape((-1,))
            loss = self.loss(char_probs, labels)  # https://github.com/microsoft/DeepSpeed/issues/962
            output['loss'] = loss

        return output.get('loss', None), output['char_probs'], output['embeddings']

    def forward(self,
                input_ids: torch.Tensor,
                labels: Optional[torch.Tensor],
                attention_mask_matrix: torch.Tensor,
                attention_mask_sentence: torch.Tensor,
                predict_indices: torch.Tensor,
                token_type_ids=None):

        output_for_predictions = self.hlm_model(
            input_ids,
            attention_mask_matrix=attention_mask_matrix,
            attention_mask_sentence=attention_mask_sentence,
            token_type_ids=None,
            predict_indices=predict_indices)['embeddings']

        output_for_predictions = self.dense_layer(output_for_predictions)
        output_for_predictions = self.layer_norm(self.activation(output_for_predictions))

        lm_hidden_states = self.lm_layer(output_for_predictions)
        # [batch_size, num_mask_char, self.vocab_size] -> [batch_size*num_mask_char, self.vocab_size]
        lm_hidden_states = lm_hidden_states.reshape((-1, self.vocab_size))

        return self._compute_loss(output_for_predictions, lm_hidden_states, predict_indices, labels)


class HLMForAutoregressiveLanguageModeling(HLMForMaskedLanguageModeling):
    def _get_causal_mask(self, output_for_predictions: torch.Tensor) -> torch.Tensor:
        causal_mask = (torch.triu(torch.ones(output_for_predictions.shape[1],
                                             output_for_predictions.shape[1])) == 1).transpose(0, 1)
        causal_mask = causal_mask.float().masked_fill(causal_mask == 0, float('-inf')).masked_fill(causal_mask == 1,
                                                                                                   float(0.0))
        return causal_mask.to(output_for_predictions.device)

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor],
                attention_mask: torch.Tensor,
                predict_indices: torch.Tensor) -> Tuple:
        output_for_predictions = self.hlm_model(input_ids, attention_mask, predict_indices)['embeddings']

        causal_mask = self._get_causal_mask(output_for_predictions)

        autoregressive_char_seq = self.autregressive_encoder(output_for_predictions.transpose(0, 1),
                                                             src_mask=causal_mask).transpose(0, 1)

        lm_hidden_states = self.lm_layer(autoregressive_char_seq)
        char_probs = self.log_softmax(lm_hidden_states)

        return self._compute_loss(output_for_predictions, char_probs, predict_indices, labels)

    def __init__(self, config, vocab_size: int):
        super(HLMForAutoregressiveLanguageModeling, self).__init__(config, vocab_size=vocab_size)

        self.autregressive_encoder = torch.nn.TransformerEncoderLayer(self.hlm_model.config.hidden_size,
                                                                      self.hlm_model.config.attention_heads,
                                                                      dim_feedforward=self.hlm_model.config.transformer_ff_size,
                                                                      dropout=self.hlm_model.config.dropout,
                                                                      activation=self.hlm_model.config.activation)

class HLMForQuestionAnswering(HLMForTask):
    def __init__(self, config, vocab_size: int, **kwargs):
        super(HLMForQuestionAnswering, self).__init__(config)
        self.vocab_size = vocab_size
        self.config = config
        self.hlm_model.for_token_classification = True
        self.qa_outputs = nn.Linear(self.config.hidden_size, self.vocab_size)

        #self.apply(self.init_weights)

    def forward(self, input_ids: torch.Tensor,
                attention_mask_matrix: torch.Tensor, attention_mask_sentence: torch.Tensor, start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None) -> Tuple:
        
        # [batch_size, embed_size]
        sequence_output = self.hlm_model(input_ids, attention_mask_matrix=attention_mask_matrix,\
                                          attention_mask_sentence=attention_mask_sentence)['embeddings_pool']
        #[max_num_word, batch_size, embed_size] -> [batch_size, max_num_word, embed_size]
        sequence_output = sequence_output.transpose(0,1)
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        
        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions.long())
            end_loss = loss_fct(end_logits, end_positions.long())
            total_loss = (start_loss + end_loss) / 2

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits
        )