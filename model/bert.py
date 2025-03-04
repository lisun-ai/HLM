# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) Microsoft, Inc. 2020
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This piece of code is from https://github.com/microsoft/DeBERTa, which is modified based on https://github.com/huggingface/transformers

import copy
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence
    
from packaging import version
import numpy as np
import math
import os
import pdb

import json
from .ops import *
from .disentangled_attention import *
from .da_utils import *

__all__ = ['BertEncoder', 'BertEmbeddings', 'ACT2FN', 'LayerNorm', 'BertLMPredictionHead']

class Config:
    pass

class BertSelfOutput(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
    self.dropout = StableDropout(config.hidden_dropout_prob)
    self.config = config

  def forward(self, hidden_states, input_states, mask=None):
    hidden_states = self.dense(hidden_states)
    hidden_states = self.dropout(hidden_states)
    hidden_states += input_states
    hidden_states = MaskedLayerNorm(self.LayerNorm, hidden_states)
    return hidden_states

class BertAttention(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.max_query_span = getattr(config, 'max_attention_query_span', -1)
    self.self = DisentangledSelfAttention(config)
    self.output = BertSelfOutput(config)
    self.config = config

  def forward(self, hidden_states, attention_mask, return_att=False, query_states=None, relative_pos=None, rel_embeddings=None, **kwargs):
    output = {}
    def self_fn(hs, m, q, r, e):
      return self.self(hs, m, False, query_states=q, relative_pos=r, rel_embeddings=e, **kwargs)['hidden_states']
    if self.max_query_span<1 or (hidden_states.size(1) <= self.max_query_span) or torch.onnx.is_in_onnx_export():
      self_output = self.self(hidden_states, attention_mask, return_att, query_states=query_states, relative_pos=relative_pos,\
            rel_embeddings=rel_embeddings, **kwargs)
    else:
      assert query_states is None, 'Query split is only supported in encoder'
      outputs = []
      offset = 0
      while offset<hidden_states.size(1):
        span = min(self.max_query_span, hidden_states.size(1)-offset)
        _query = hidden_states.narrow(1, offset, span)
        _mask = attention_mask.narrow(-2, offset, span)
        _rep = relative_pos.narrow(-2, offset, span)
        if torch.all(_mask==0):
          out = {'hidden_states': torch.zeros_like(_query)}
        else:
          if self.training:
            hs = checkpoint(self_fn, hidden_states, _mask, _query, _rep, rel_embeddings)
          else:
            hs = self_fn(hidden_states, _mask, _query, _rep, rel_embeddings)
          out = {'hidden_states': hs}
        outputs.append(out)
        offset += self.max_query_span
      self_output = {'hidden_states': torch.cat([o['hidden_states'] for o in outputs], dim=1)}
    if query_states is None:
      query_states = hidden_states
    attention_output = self.output(self_output['hidden_states'], query_states, attention_mask)
    output.update(self_output)
    output['hidden_states'] = attention_output
    return output

class BertIntermediate(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.intermediate_act_fn = ACT2FN[config.hidden_act] \
      if isinstance(config.hidden_act, str) else config.hidden_act

  def forward(self, hidden_states):
    hidden_states = self.dense(hidden_states)
    hidden_states = self.intermediate_act_fn(hidden_states)
    return hidden_states

class BertOutput(nn.Module):
  def __init__(self, config):
    super(BertOutput, self).__init__()
    self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
    self.dropout = StableDropout(config.hidden_dropout_prob)
    self.config = config

  def forward(self, hidden_states, input_states, mask=None):
    hidden_states = self.dense(hidden_states)
    hidden_states = self.dropout(hidden_states)
    hidden_states += input_states
    hidden_states = MaskedLayerNorm(self.LayerNorm, hidden_states)
    return {'hidden_states': hidden_states}

class BertLayer(nn.Module):
  def __init__(self, config):
    super(BertLayer, self).__init__()
    self.attention = BertAttention(config)
    self.intermediate = BertIntermediate(config)
    self.output = BertOutput(config)
    self.with_cross_attention = getattr(config, 'with_cross_attention', False)
    if self.with_cross_attention:
      self.cross_attention = BertAttention(config.cross_attention)

  def forward(self, hidden_states, attention_mask, return_att=False, query_states=None, relative_pos=None, rel_embeddings=None, \
    history_states=None, encoder_states=None, cross_attention_mask=None, \
    cross_relative_pos=None, **kwargs):
    output = {}
    attention_output = self.attention(hidden_states, attention_mask, return_att=return_att, \
      query_states = query_states, relative_pos=relative_pos, rel_embeddings=rel_embeddings, \
      history_states = history_states, **kwargs)
    output.update(attention_output)
    if self.with_cross_attention:
      assert encoder_states is not None, 'Cross attention must consume encoder output states'
      attention_output = self.cross_attention(encoder_states, cross_attention_mask, return_att=return_att, \
        query_states = attention_output['hidden_states'], relative_pos = cross_relative_pos, \
        rel_embeddings = rel_embeddings, tag='cross', **kwargs)
      output.update(attention_output)

    intermediate_output = self.intermediate(attention_output['hidden_states'])
    layer_output = self.output(intermediate_output, attention_output['hidden_states'], attention_mask)
    output.update(layer_output)
    return output

class ConvLayer(nn.Module):
    def __init__(self, config):
      super().__init__()
      kernel_size = getattr(config, 'conv_kernel_size', 3)
      groups = getattr(config, 'conv_groups', 1)
      self.conv_act = getattr(config, 'conv_act', 'tanh')
      self.conv = torch.nn.Conv1d(config.hidden_size, config.hidden_size, kernel_size, padding = (kernel_size-1)//2, groups = groups)
      self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
      self.dropout = StableDropout(config.hidden_dropout_prob)
      self.config = config

    def forward(self, hidden_states, residual_states, input_mask):
        out = self.conv(hidden_states.permute(0,2,1).contiguous()).permute(0,2,1).contiguous()
        if version.Version(torch.__version__) >= version.Version('1.2.0a'):
            rmask = (1-input_mask).bool()
        else:
            rmask = (1-input_mask).byte()
        out.masked_fill_(rmask.unsqueeze(-1).expand(out.size()), 0)
        out = ACT2FN[self.conv_act](self.dropout(out))
        output_states = MaskedLayerNorm(self.LayerNorm, residual_states + out, input_mask)

        return output_states


class BertEncoder(nn.Module):
  """ Modified BertEncoder with relative position bias support
  """
  def __init__(self, num_hidden_layers, max_relative_positions, layer_norm_eps=1e-7, hidden_size=768,\
               intermediate_size=3072, pos_att_type="p2c|c2p", hidden_dropout_prob=0.1, hidden_act="gelu",\
               position_buckets=256, norm_rel_ebd='layer_norm', relative_attention=True, num_attention_heads=12,\
               attention_probs_dropout_prob=0.1, conv_kernel_size=0):
    super().__init__()
    
    # create the config from parameters
    config = Config()
    config.num_hidden_layers = num_hidden_layers
    config.max_relative_positions = max_relative_positions
    config.layer_norm_eps = layer_norm_eps
    config.hidden_size = hidden_size
    config.intermediate_size = intermediate_size
    config.pos_att_type = pos_att_type
    config.position_buckets = position_buckets
    config.norm_rel_ebd = norm_rel_ebd
    config.relative_attention = relative_attention
    config.num_attention_heads = num_attention_heads
    config.conv_kernel_size = conv_kernel_size
    config.hidden_dropout_prob = hidden_dropout_prob
    config.hidden_act = hidden_act
    config.attention_probs_dropout_prob = attention_probs_dropout_prob
    #if num_hidden_layers == 4:
    #    print(22, intermediate_size, num_attention_heads, max_relative_positions)
    self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
    self.relative_attention = getattr(config, 'relative_attention', True)
    self.num_attention_heads = config.num_attention_heads
    if self.relative_attention:
      self.max_relative_positions = getattr(config, 'max_relative_positions', -1)
      if self.max_relative_positions < 1:
        self.max_relative_positions = config.max_position_embeddings
      self.position_buckets = getattr(config, 'position_buckets', -1)
      pos_ebd_size = self.max_relative_positions * 2
      if self.position_buckets > 0:
        pos_ebd_size = self.position_buckets * 2
      self.rel_embeddings = nn.Embedding(pos_ebd_size, config.hidden_size)

    self.norm_rel_ebd = [x.strip() for x in getattr(config, 'norm_rel_ebd', 'none').lower().split('|')]
    if 'layer_norm' in self.norm_rel_ebd:
      self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine = True)
    kernel_size = getattr(config, 'conv_kernel_size', 0)
    self.with_conv = False
    if kernel_size > 0:
      self.with_conv = True
      self.conv = ConvLayer(config)

  def get_rel_embedding(self):
    rel_embeddings = self.rel_embeddings.weight if self.relative_attention else None
    if rel_embeddings is not None and ('layer_norm' in self.norm_rel_ebd):
      rel_embeddings = self.LayerNorm(rel_embeddings)
    return rel_embeddings

  def get_attention_mask(self, attention_mask):
    if attention_mask.dim() <= 2:
      extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
      attention_mask = extended_attention_mask&extended_attention_mask.squeeze(-2).unsqueeze(-1)
      attention_mask = attention_mask.byte()
    elif attention_mask.dim() == 3:
      attention_mask = attention_mask.unsqueeze(1)

    return attention_mask

  def get_rel_pos(self, hidden_states, query_states=None, relative_pos=None, history_states=None, position_ids=None):
    if self.relative_attention and relative_pos is None:
      q = query_states.size(-2) if query_states is not None else hidden_states.size(-2)
      k = hidden_states.size(-2)
      if history_states is not None:
        k += history_states.size(-2)
      if (position_ids is not None):
        if (history_states is None):
          q = position_ids
          k = position_ids
        else:
          q = position_ids
          k = torch.arange(k).to(hidden_states.device)
      else:
          q = torch.arange(q).to(hidden_states.device)
          k = torch.arange(k).to(hidden_states.device)
      relative_pos = build_relative_position_from_abs(q, k, bucket_size = self.position_buckets, \
          max_position=self.max_relative_positions, device = hidden_states.device)
    if relative_pos is None:
      return relative_pos
    if relative_pos.dim()==2:
      relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
    elif relative_pos.dim()==3:
      relative_pos = relative_pos.unsqueeze(1)
    # bxhxqxk
    elif relative_pos.dim()!=4:
      raise ValueError(f'Relative postion ids must be of dim 2 or 3 or 4. {relative_pos.dim()}')
    if self.relative_attention:
      span = self.rel_embeddings.weight.size(0)//2
    else:
      span = 0
    relative_pos = (relative_pos + span).long()
    relative_pos = relative_pos.expand(
                                      hidden_states.size(0),
                                      self.num_attention_heads,
                                      relative_pos.size(-2),
                                      relative_pos.size(-1)).reshape(-1, relative_pos.size(-2), relative_pos.size(-1))
    return relative_pos


#  encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*)
#  encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
#            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
#            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:
#            - 1 for tokens that are **not masked**,
#            - 0 for tokens that are **masked**.
  def forward(
              self,
              hidden_states,
              attention_mask,
              output_all_encoded_layers=True,
              return_att=False,
              query_states = None,
              relative_pos=None,
              prev_states=None,
              history_states=None,
              encoder_states=None,
              cross_attention_mask=None,
              cross_relative_pos=None,
              position_ids=None,
              **kwargs):
    if attention_mask.dim() <= 2:
      input_mask = attention_mask
    else:
      input_mask = (attention_mask.sum(-2)>0).byte()
    attention_mask = self.get_attention_mask(attention_mask)
    relative_pos = self.get_rel_pos(hidden_states, query_states, relative_pos, history_states[0] if history_states is not None else None, position_ids)
    embedding_states = hidden_states

    #all_encoder_layers = [embedding_states]
    #att_matrices = []
    #att_logits = []
    if isinstance(hidden_states, Sequence):
      next_kv = hidden_states[0]
    else:
      next_kv = hidden_states
    rel_embeddings = self.get_rel_embedding()
    for i, layer_module in enumerate(self.layer):
      prev_s = prev_states[i] if prev_states is not None and isinstance(prev_states, list) else None
      history_s = history_states[i] if history_states is not None else None
      output = layer_module(next_kv, attention_mask, return_att, query_states = query_states, relative_pos=relative_pos, rel_embeddings=rel_embeddings, \
        prev_states=prev_s, history_states=history_s, \
        encoder_states=encoder_states, cross_attention_mask=cross_attention_mask, cross_relative_pos=cross_relative_pos, **kwargs)
      output_states, att_m, att_l = output['hidden_states'], output['attention_probs'], output['attention_logits']

      if i == 0 and self.with_conv:
        prenorm = output_states #output['prenorm_states']
        output_states = self.conv(hidden_states, prenorm, input_mask)

      if query_states is not None:
        query_states = output_states
        if isinstance(hidden_states, Sequence):
          next_kv = hidden_states[i+1] if i+1 < len(self.layer) else None
      else:
        next_kv = output_states

      #all_encoder_layers.append(output_states)
      #att_matrices.append(att_m)
      #att_logits.append(att_l)
    #return {
    #    'hidden_states': all_encoder_layers,
    #    'attention_probs': att_matrices,
    #    'attention_logits': att_logits
    #    }
    return output_states

class BertEmbeddings(nn.Module):
  """Construct the embeddings from word, position and token_type embeddings.
  """
  def __init__(self, config):
    super(BertEmbeddings, self).__init__()
    padding_idx = getattr(config, 'padding_idx', 0)
    self.embedding_size = getattr(config, 'embedding_size', config.hidden_size)
    self.word_embeddings = nn.Embedding(config.vocab_size, self.embedding_size, padding_idx = padding_idx)
    self.position_biased_input = getattr(config, 'position_biased_input', True)
    self.position_embeddings = nn.Embedding(config.max_position_embeddings, self.embedding_size)

    if config.type_vocab_size>0:
      self.token_type_embeddings = nn.Embedding(config.type_vocab_size, self.embedding_size)
    
    if self.embedding_size != config.hidden_size:
      self.embed_proj = nn.Linear(self.embedding_size, config.hidden_size, bias=False)
    self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
    self.dropout = StableDropout(config.hidden_dropout_prob)
    self.output_to_half = False
    self.config = config

  def forward(self, input_ids, token_type_ids=None, position_ids=None, mask = None):
    seq_length = input_ids.size(1)
    if position_ids is None:
      position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
      position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    if token_type_ids is None:
      token_type_ids = torch.zeros_like(input_ids)

    words_embeddings = self.word_embeddings(input_ids)
    position_embeddings = self.position_embeddings(position_ids.long())

    embeddings = words_embeddings
    if self.config.type_vocab_size>0:
      token_type_embeddings = self.token_type_embeddings(token_type_ids)
      embeddings += token_type_embeddings

    if self.position_biased_input:
      embeddings += position_embeddings

    if self.embedding_size != self.config.hidden_size:
      embeddings = self.embed_proj(embeddings)
    embeddings = MaskedLayerNorm(self.LayerNorm, embeddings, mask)
    embeddings = self.dropout(embeddings)
    return {
        'embeddings': embeddings,
        'position_embeddings': position_embeddings}

class BertLMPredictionHead(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.embedding_size = getattr(config, 'embedding_size', config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, self.embedding_size)
        self.transform_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act

        self.LayerNorm = LayerNorm(self.embedding_size, config.layer_norm_eps, elementwise_affine=True)

        self.bias = nn.Parameter(torch.zeros(vocab_size))

    def forward(self, hidden_states, embeding_weight):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        # b x s x d
        hidden_states = MaskedLayerNorm(self.LayerNorm, hidden_states)

        # b x s x v
        logits = torch.matmul(hidden_states, embeding_weight.t().to(hidden_states)) + self.bias
        return logits
