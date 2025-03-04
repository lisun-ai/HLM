import inspect
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset as TorchDataset
from datasets import Dataset, load_dataset
from transformers import TrainingArguments

from model.hlm import HLM
from model.codepoint_tokenizer import CodepointTokenizer

MAX_JP_CODEPOINT = 1024

logger = logging.getLogger(__name__)


@dataclass
class DataArguments:
    data: Optional[str] = field(
        default=None, metadata={"help": "The location of the wiki data to use for training."}
    )
    data_dir: Optional[str] = field(default=None)
    datasets: Dict[str, int] = field(
        default=None, metadata={"help": "Dictionary of dataset name and number of parts."}
    )
    data_list: Optional[str] = field(
        default=None,
        metadata={"help": "Path to text file containing list of JSON sample paths."}
    )
    char2int_dict: Optional[str] = field(default="pickle/char2int_dict.pkl", metadata={"help": "Relative path to codebook dict."})
    int2char_dict: Optional[str] = field(default="pickle/int2char_dict.pkl", metadata={"help": "Relative path to codebook dict."})
    streaming: Optional[bool] = field(default=False)


@dataclass
class HLMTrainingArguments(TrainingArguments):
    masking_type: Optional[str] = field(default='rand_char')
    load_only_model: Optional[bool] = field(default=False)
    group_by_length: Optional[bool] = field(default=False)
    logging_first_step: Optional[bool] = field(default=True)
    learning_rate: Optional[float] = 0.001
    gradient_checkpointing: Optional[bool] = field(default=False)
    logging_steps: Optional[int] = field(default=200)
    report_to: Optional[List[str]] = field(default=None)
    evaluation_strategy: Optional[str] = field(default='no')
    fp16: Optional[bool] = field(default=False)
    deepspeed: Optional[bool] = field(default=None)
    warmup_ratio: Optional[float] = 0.0  # from canine
    warmup_steps: Optional[float] = 10000
    masking_percent: Optional[float] = 0.15

    per_device_eval_batch_size: Optional[int] = field(default=12)
    per_device_train_batch_size: Optional[int] = field(default=12)
    # max that we can fit on one GPU is 12. 12 * 21 * 8 = 2016
    gradient_accumulation_steps: Optional[int] = field(default=1)

    # model arguments - these have to be in training args for the hyperparam search
    max_char_length: Optional[int] = field(default=2048)
    max_char_per_word: Optional[int] = field(default=20)
    max_num_word: Optional[int] = field(default=1024)
    pos_att_type: Optional[str] = field(default="p2c|c2p")
    hidden_size: Optional[int] = field(
        default=768,
        metadata={"help": "Hidden size."}
    )
    attention_heads: Optional[int] = field(
        default=12,
        metadata={"help": "Number of attention heads."}
    )
    transformer_ff_size: Optional[int] = field(
        default=3072,
        metadata={"help": "Transformer feed-forward size."}
    )
    local_transformer_ff_size: Optional[int] = field(
        default=1536,
        metadata={"help": "Local transformer feed-forward size."}
    )
    dropout: Optional[float] = field(
        default=0.1,
        metadata={"help": "Dropout probability."}
    )
    activation: Optional[str] = field(
        default="gelu",
        metadata={"help": "Activation function."}
    )
    aggregation_method: Optional[str] = field(
        default="word_cls",
        metadata={"help": "Character aggregation method."}
    )
    max_codepoint: Optional[int] = field(
        default=1024,
        metadata={"help": "Maximum number of characters in codebook."}
    )
    for_token_classification: Optional[bool] = field(
        default=False,
        metadata={"help": "Is the model for token classification task or not?"}
    )
    n_local_layer_first: Optional[int] = field(
        default=3,
        metadata={"help": "Number of layers for first intra-word transformers."}
    )
    n_global_layer: Optional[int] = field(
        default=6,
        metadata={"help": "Number of layers for inter-word transformers."}
    )
    n_local_layer_last: Optional[int] = field(
        default=3,
        metadata={"help": "Number of layers for last intra-word transformers."}
    )
    use_token_type: bool = field(
        default=False,
        metadata={"help": "Use token type embeddings or not?"}
    )
    use_projection: Optional[bool] = field(
        default=False,
        metadata={"help": "Concatenate word-level with char-level features and project them or jsut concatenate them?"}
    )
    word_context: Optional[int] = field(
        default=1,
        metadata={"help": "Number of words to take into account for initial char embeddings."}
    )
    relative_attention: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, use relative positional encoding."}
    )
    random_mask_max_range: Optional[int] = field(
        default=128,
        metadata={"help": "Max unicode range value for random character masking"}
    )


@dataclass
class HLMWordSegArgs(HLMTrainingArguments):
    do_predict: Optional[bool] = field(default=True)

    # only used for hyperparameter search
    trials: Optional[int] = field(default=2)
    deepspeed: Optional[bool] = field(default=None)
    gradient_accumulation_steps: Optional[int] = field(default=1)
    report_to: Optional[List[str]] = field(default=lambda: ['tensorboard', 'wandb'])
    num_train_epochs: Optional[int] = 6
    save_strategy: Optional[str] = 'no'

    pretrained_bert: Optional[str] = field(default=None)


@dataclass
class HLMClassificationArgs(HLMTrainingArguments):
    do_predict: Optional[bool] = field(default=True)
    eval_steps: Optional[int] = field(default=300)
    logging_steps: Optional[int] = field(default=100)
    learning_rate: Optional[float] = 2e-5
    per_device_train_batch_size: Optional[int] = 6
    num_train_epochs: Optional[int] = 6
    save_strategy: Optional[str] = 'no'

    # only used for hyperparameter search
    trials: Optional[int] = field(default=2)
    deepspeed: Optional[bool] = field(default=None)
    gradient_accumulation_steps: Optional[int] = field(default=1)
    report_to: Optional[List[str]] = field(default=lambda: ['tensorboard', 'wandb'])

    pretrained_bert: Optional[str] = field(default=None)


def get_model_hyperparams(input_args):
    if not isinstance(input_args, dict):
        input_args = input_args.__dict__

    hlm_hyperparams = inspect.getfullargspec(HLM.__init__).args
    return {key: val for key, val in input_args.items() if key in hlm_hyperparams}


def get_base_hlm_state_dict(state_dict: Dict) -> Dict:
    if sum(1 for x in state_dict.keys() if x.startswith('hlm_model')) > 0:
        return {key[12:]: val for key, val in state_dict.items() if key.startswith('hlm_model')}
    else:
        return state_dict


def prepare_data(args: DataArguments) -> Tuple[Dataset, Dataset]:
    all_data = load_dataset('json', data_files=args.data)['train']
    data_dict = all_data.train_test_split(train_size=0.98, seed=42)
    training_data = data_dict['train']
    dev_data = data_dict['test']
    return training_data, dev_data


class SequenceLabelingDataCollator:
    def __init__(self):
        self.tokenizer = CodepointTokenizer()

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        padded_batch = self.tokenizer.pad([x['input_ids'] for x in batch])
        input_ids = padded_batch['input_ids']
        attention_mask = padded_batch['attention_mask']

        # don't compute loss from padding
        labels = pad_sequence([torch.tensor(x['labels']) for x in batch], batch_first=True, padding_value=-100)
        # also don't compute loss from CLS or SEP tokens
        special_token_mask = (input_ids == self.tokenizer.CLS) | (input_ids == self.tokenizer.SEP)
        labels = labels.where(~special_token_mask, torch.full(labels.shape, -100))

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'word_mask': padded_batch['word_mask'],
            'pool_mask': padded_batch['pool_mask']
        }


class ClassificationDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        model_inputs = self.tokenizer.prepare_model_inputs(batch)

        labels = torch.tensor([x['labels'] for x in batch])
        model_inputs['labels'] = labels

        return model_inputs


class CanineClassificationDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        batch_size = len(batch)
        input_ids = [item['input_ids'] for item in batch]
        token_type_ids = [item['token_type_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        max_len = 0
        for item in input_ids:
            max_len = max(max_len, len(item))
        # print("batch_size=%s max_len=%s" % (batch_size, max_len))
        for n in range(batch_size):
            pad_len = max_len - len(input_ids[n])
            input_ids[n] += [self.tokenizer.pad_token_id] * pad_len
            token_type_ids[n] += [token_type_ids[n][-1]] * pad_len
            attention_mask[n] += [0] * pad_len
        padded_batch = {
            "input_ids": torch.tensor([item for item in input_ids]),
            "token_type_ids": torch.tensor([item for item in token_type_ids]),
            "attention_mask": torch.tensor([item for item in attention_mask]),
            "labels": torch.tensor([x['labels'] for x in batch])
        }
        return padded_batch


class BertClassificationDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.max_seq_length = tokenizer.model_max_length

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        batch_size = len(batch)
        input_ids = [item['input_ids'] for item in batch]
        token_type_ids = [item['token_type_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        max_len = 0
        for item in input_ids:
            max_len = max(max_len, len(item))
        # print("batch_size=%s max_len=%s" % (batch_size, max_len))
        max_len = min(max_len, self.max_seq_length)
        for n in range(batch_size):
            pad_len = max_len - len(input_ids[n])
            if pad_len >= 0:
                # Pad sequence to max_len
                input_ids[n] += [self.tokenizer.pad_token_id] * pad_len
                token_type_ids[n] += [token_type_ids[n][-1]] * pad_len
                attention_mask[n] += [0] * pad_len
            else:
                # Truncate sequence to max_len
                input_ids[n] = input_ids[n][:max_len]
                input_ids[n][-1] = self.tokenizer.sep_token_id
                token_type_ids[n] = token_type_ids[n][:max_len]
                attention_mask[n] = attention_mask[n][:max_len]
        padded_batch = {
            "input_ids": torch.tensor([item for item in input_ids]),
            "token_type_ids": torch.tensor([item for item in token_type_ids]),
            "attention_mask": torch.tensor([item for item in attention_mask]),
            "labels": torch.tensor([x['labels'] for x in batch])
        }
        return padded_batch


class TokenClassificationDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        result_dict = self.tokenizer.prepare_model_inputs(batch)

        batch_size, max_num_word = result_dict['attention_mask_sentence'].shape
        label_ids_matrix = np.ones((batch_size, max_num_word)) * (-100)

        for idx, item in enumerate(batch):
            label_ids = np.array(item['label_ids'])
            # pad_token_label_id = -100
            label_ids_vaild = label_ids != -100
            label_ids = label_ids[label_ids_vaild]
            # 1 <CLS> at the beginning
            label_ids_matrix[idx, 1: 1 + len(label_ids)] = label_ids
        result_dict['labels'] = torch.from_numpy(label_ids_matrix).long()

        return result_dict


@dataclass
class InputExample:
    """
    A single training/test example for token classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        labels: (Optional) list. The labels for each word of the sequence. This should be
        specified for train and dev examples, but not for test examples.
    """
    guid: str
    words: List[str]
    labels: Optional[List[str]]


@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None
    word_mask = None


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


class TokenClassificationTask:
    @staticmethod
    def read_examples_from_file(data_dir, mode: Union[Split, str]) -> List[InputExample]:
        raise NotImplementedError

    @staticmethod
    def get_labels(path: str) -> List[str]:
        raise NotImplementedError

    @staticmethod
    def convert_examples_to_features(
        examples: List[InputExample],
        label_list: List[str],
        max_seq_length: int,
        tokenizer,
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=1,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        pad_token_label_id=-100,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
    ):
        """Loads a data file into a list of `InputFeatures`
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
        """
        # TODO clean up all this to leverage built-in features of tokenizers
        cls_token = tokenizer.CLS
        sep_token = tokenizer.SEP
        label_map = {label: i for i, label in enumerate(label_list)}

        features = []
        for ex_index, example in enumerate(examples):
            if ex_index % 10_000 == 0:
                logger.info("Writing example %d of %d", ex_index, len(examples))

            tokens = []
            label_ids = []
            word_mask_list = []
            for word, label in zip(example.words, example.labels):
                word_tokens = tokenizer.encode_word(word)
                word_mask = [0] * (len(word_tokens) - 1) + [1]
                # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
                if len(word_tokens) > 0:
                    tokens.extend(word_tokens)
                    # !Use the real label id for the first token of the word, and padding ids for the remaining tokens
                    label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))
                    word_mask_list.extend(word_mask)

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids:   0   0   0   0  0     0   0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
            word_mask_list += [1]
            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens += [sep_token]
                label_ids += [pad_token_label_id]
            segment_ids = [sequence_a_segment_id] * len(tokens)

            if cls_token_at_end:
                tokens += [cls_token]
                label_ids += [pad_token_label_id]
                segment_ids += [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                label_ids = [pad_token_label_id] + label_ids
                segment_ids = [cls_token_segment_id] + segment_ids
                word_mask_list = [1] + word_mask_list

            # Account for [CLS] and [SEP] with "- 2" (BERT) and with "- 3" for RoBERTa.
            # trim too long sequence
            special_tokens_count = 2
            if len(tokens) > max_seq_length - special_tokens_count:
                tokens = tokens[: (max_seq_length - special_tokens_count)]
                label_ids = label_ids[: (max_seq_length - special_tokens_count)]
                label_ids = label_ids[: (max_seq_length - special_tokens_count)]
                word_mask_list = word_mask_list[: (max_seq_length - special_tokens_count)]

            assert len(tokens) > 2

            input_ids = tokens

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s", example.guid)
                logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
                logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
                logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
                logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))
                logger.info("word_mask: %s", " ".join([str(x) for x in word_mask_list]))

            features.append(
                {
                    'input_ids': input_ids,
                    'attention_mask': input_mask,
                    'token_type_ids': segment_ids,
                    'label_ids': label_ids,
                    'word_mask': word_mask_list
                }
            )
        return features


class TokenClassificationDataset(TorchDataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    features: List[InputFeatures]
    pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index
    # Use cross entropy ignore_index as padding label id so that only
    # real label ids contribute to the loss later.

    def __init__(
        self,
        token_classification_task: TokenClassificationTask,
        data_dir: str,
        tokenizer,
        labels: List[str],
        model_type: str,
        max_seq_length: Optional[int] = None,
        overwrite_cache=False,
        mode: Split = Split.train,
    ):
        local_rank = int(os.environ["LOCAL_RANK"])
        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        if local_rank not in [-1, 0]:
            torch.distributed.barrier()
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            data_dir,
            "cached_{}_{}_{}".format(mode.value, tokenizer.__class__.__name__, str(max_seq_length)),
        )
        if os.path.exists(cached_features_file) and not overwrite_cache:
            print(f"Loading features from cached file {cached_features_file}")
            self.features = torch.load(cached_features_file)
        else:
            print(f"Creating features from dataset file at {data_dir}")
            examples = token_classification_task.read_examples_from_file(data_dir, mode)
            # TODO clean up all this to leverage built-in features of tokenizers
            self.features = token_classification_task.convert_examples_to_features(
                examples,
                labels,
                max_seq_length,
                tokenizer,
                cls_token_at_end=bool(model_type in ["xlnet"]),
                # xlnet has a cls token at the end
                cls_token=tokenizer.CLS,
                cls_token_segment_id=2 if model_type in ["xlnet"] else 0,
                sep_token=tokenizer.SEP,
                sep_token_extra=False,
                # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                pad_on_left=False,
                pad_token=tokenizer.PAD,
                pad_token_segment_id=tokenizer.PAD,
                pad_token_label_id=self.pad_token_label_id)
            print(f"Saving features into cached file {cached_features_file}")
            torch.save(self.features, cached_features_file)
        if local_rank == 0:
            torch.distributed.barrier()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]
