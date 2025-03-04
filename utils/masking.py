import math
import os
import random
from collections import defaultdict
from typing import Tuple, Dict, Set, List
import unicodedata

import numpy as np
import torch
from tokenizers import Tokenizer

from model.codepoint_tokenizer import CodepointTokenizer

SPECIAL_TOKENS_WITHOUT_PADDING = {CodepointTokenizer.PAD, CodepointTokenizer.MASK, CodepointTokenizer.SEP, CodepointTokenizer.CLS, CodepointTokenizer.UNK, CodepointTokenizer.WORD_CLS}
MAX_SPECIAL_TOKEN = max(SPECIAL_TOKENS_WITHOUT_PADDING)
MIN_SPECIAL_TOKEN = min(SPECIAL_TOKENS_WITHOUT_PADDING)


def _special_tokens_mask_from_range(input_ids, special_tokens: range):
    return (input_ids >= special_tokens.start) & (input_ids <= special_tokens.stop)


def custom_mask(
        input_ids,
        attention_mask_matrix,
        attention_mask_sentence,
        masking_percent: float = 0.15,
        min_char_replacement: int = 5,
        max_char_replacement: int = 128):
    max_char_len, max_num_word, batch_size = input_ids.shape
    maskable_chars = ~attention_mask_matrix
    # [WORD_CLS] cannot be masked
    maskable_chars[0, :, :] = False
    n_chars = 0
    n_words = 0
    random_flag = np.random.uniform(low=0.0, high=1.0, size=(max_num_word, batch_size))
    n_masked_word = 0
    n_masked_char = 0
    for bi in range(batch_size):
        seq = input_ids[:, :, bi]
        pad_words = seq[0, :] != CodepointTokenizer.WORD_CLS
        n_maskable_words = np.sum(~pad_words)
        n_words += n_maskable_words
        random_flag = np.random.uniform(low=0.0, high=1.0, size=n_maskable_words)
        i = 0
        for wi in range(max_num_word):
            if pad_words[wi]:
                continue
            do_mask = random_flag[i] <= masking_percent
            n_masked_word += do_mask
            mask = maskable_chars[:, wi, bi]
            maskable_word = seq[mask, wi]
            n_chars += len(maskable_word)
            if do_mask:
                n_masked_char += len(maskable_word)
                if random.random() <= .8:
                    # 80% of the time, we replace input tokens with mask token
                    seq[mask, wi] = CodepointTokenizer.MASK
                elif random.random() <= .5:
                    # 10% of the time, we corrupt some of the characters
                    random_chars = np.random.randint(
                        low=min_char_replacement,
                        high=max_char_replacement,
                        size=len(maskable_word))
                    corrupted_idx = np.random.uniform(low=0.0, high=1.0, size=len(maskable_word)) <= masking_percent
                    maskable_word[corrupted_idx] = random_chars[corrupted_idx]
                    seq[mask, wi] = maskable_word
                else:
                    # 10% of the time, we keep the tokens unchanged
                    pass
            i += 1
    return torch.from_numpy(input_ids).long()


def random_mask(input_ids, attention_mask_matrix, attention_mask_sentence,
                masking_percent: float = 0.15,
                min_char_replacement: int = 5,
                max_char_replacement: int = 128):
    """The standard way to do this (how HuggingFace does it) is to randomly mask each token with masking_prob. However,
    this can result in a different number of masks for different sentences, which would prevent us from using gather()
    to save compute like CANINE does. consequently, we instead always mask exactly length * masking_prob tokens, and
    instead select those indices randomly. this means that the masked indices can (and likely will) be OUT OF ORDER.
    it also means that in batches with padding, potentially a higher % will be masked in the shorter sentences"""
    # input_ids: [max_char_len, max_num_word, batch_size]
    # attention_mask_sentence: [batch_size, max_num_word]: index to ignore
    labels = input_ids.copy()
    input_ids = input_ids.copy()
    max_char_len, max_num_word, batch_size = input_ids.shape

    # [batch_size,]
    # exclude empty words, special tokens/ empty word (pad) are at the first char, ignore WORD_CLS (5)
    # [max_num_word, batch_size]
    special_tokens_mask = _special_tokens_mask_from_range(input_ids[0, :, :], range(MIN_SPECIAL_TOKEN, MAX_SPECIAL_TOKEN - 1))
    # attention_mask_sentence: [batch_size, max_num_word] -> [max_num_word, batch_size]
    special_tokens_mask = special_tokens_mask | attention_mask_sentence.transpose(1, 0).astype(bool)
    maskable_char_matrix = ~attention_mask_matrix
    # first char [WORD_CLS] is not maskable
    maskable_char_matrix[0, :, :] = False

    # # some sentences are too short for masking
    # sentence_len = (attention_mask_sentence == 0).sum(axis=1).astype(np.float)
    # min_sentence_len = sentence_len.min()
    # if mask_count > min_sentence_len:
    #    mask_count = math.floor(0.8 * (min_sentence_len-2)) # <CLS> and <SEP>
    # print("mask_count:", mask_count)

    char_indices_to_mask = []
    char_mask = np.zeros((max_char_len, max_num_word, batch_size)).astype(bool)

    for idx in range(batch_size):
        # inputs: [max_char_len, max_num_word]
        # maskable_mask_word: [max_num_word]
        maskable_mask_word = ~special_tokens_mask[:, idx]
        # mask -> indices [mask_count,]
        # randomly select

        maskable_indices_word = np.nonzero(maskable_mask_word)[0]
        maskable_indices_word_len = maskable_indices_word.shape[0]

        mask_count = math.floor(maskable_indices_word_len * masking_percent)
        maskable_indices_word = np.random.choice(maskable_indices_word, size=mask_count, replace=False)

        maskable_mask_word = np.zeros((max_num_word,))
        maskable_mask_word[maskable_indices_word] = 1

        maskable_mask_char = np.zeros((max_char_len, max_num_word))
        maskable_mask_char[:, maskable_indices_word] = maskable_char_matrix[:, maskable_indices_word, idx]

        # adapt to trim order: [max_char_len, max_num_word] -> [max_num_word, max_char_len]
        maskable_indices_char = np.nonzero(maskable_mask_char.transpose((1, 0)))

        # [batch_size,] (2,num_mask_char)
        char_indices_to_mask.append(maskable_indices_char)

    for idx in range(batch_size):
        # transpose back

        char_mask[char_indices_to_mask[idx][1],
                  char_indices_to_mask[idx][0],
                  idx] = True

    random_flag = np.random.uniform(low=0.0, high=1.0, size=(max_num_word, batch_size))
    # 80% of the time, we replace masked input tokens with mask token
    indices_replaced = ((random_flag <= 0.8)[None, :, :] * char_mask).astype(bool)
    input_ids[indices_replaced] = CodepointTokenizer.MASK

    # 10% (half of the remaining 20%) of the time,
    # we replace masked input tokens with random word
    indices_random = (((random_flag > 0.8) & (random_flag <= 0.9))[None, :, :] * char_mask).astype(bool)
    random_words = np.random.randint(
        low=min_char_replacement,
        high=max_char_replacement,
        size=(max_char_len, max_num_word, batch_size))
    input_ids[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged

    # [batch_size, 2, num_mask_char]
    # char_indices_to_mask = torch.from_numpy(np.array(char_indices_to_mask)).long()

    # [max_char_len, max_num_word, batch_size] -> [num_mask_char]
    labels = labels[char_mask]

    return torch.from_numpy(input_ids).long(), torch.from_numpy(labels).long(), torch.from_numpy(char_mask).bool()


def bpe_span_mask(input_ids: torch.Tensor, attention_mask: torch.Tensor, replacement_vocab: Dict[int, List[str]],
                  bpe_tokenizer: Tokenizer, masking_percent: float = 0.15):
    # we are masking byte piece spans

    labels = input_ids.clone()
    input_ids = input_ids.clone()
    special_tokens_mask = _special_tokens_mask_from_range(input_ids, range(MIN_SPECIAL_TOKEN, MAX_SPECIAL_TOKEN))
    special_tokens_mask = special_tokens_mask | attention_mask.bool()
    mask_count = math.floor(input_ids.shape[1] * masking_percent)  # total number of tokens to mask

    spans_per_row = []
    all_masked_indices = []
    for unmaskable_indices, inputs in zip(special_tokens_mask, input_ids):
        # compute the possible indices we could mask for this input
        maskable_indices = torch.arange(inputs.shape[0]).masked_select(~unmaskable_indices)
        maskable_indices_set = set(maskable_indices.numpy())

        original_string = unicodedata.normalize("NFKC", ''.join(chr(x) for x in inputs))
        bpe_split_string = bpe_tokenizer.encode(original_string)
        start_length_tuples = [(start, end - start) for start, end in bpe_split_string.offsets
                               if start in maskable_indices_set]
        random.shuffle(start_length_tuples)
        total_masked = 0
        span_iter = iter(start_length_tuples)
        spans_to_mask = []
        while total_masked < mask_count:
            try:
                span_index, span_length = next(span_iter)
                if total_masked + span_length <= mask_count:
                    spans_to_mask.append((span_index, span_length))
                    total_masked += span_length
            except StopIteration:
                print('Warning: randomly masking to fill remaining mask slots')
                candidate_indices = list(maskable_indices_set - {idx for span_idx, span_length in spans_to_mask
                                                                 for idx in range(span_idx, span_idx + span_length)})
                random.shuffle(candidate_indices)
                for idx in candidate_indices:
                    spans_to_mask.append((idx, 1))
                    total_masked += 1
                    if total_masked == mask_count:
                        break

        assert (total_masked == mask_count)
        spans_per_row.append(spans_to_mask)

        all_masked_indices.append(torch.tensor([idx for start_loc, length in spans_to_mask
                                                for idx in range(start_loc, start_loc + length)]))

    span_starts_tensor = torch.nn.utils.rnn.pad_sequence([torch.tensor([start_idx for start_idx, length in sublist])
                                                          for sublist in spans_per_row], batch_first=True,
                                                         padding_value=-1)
    span_lengths_tensor = torch.nn.utils.rnn.pad_sequence([torch.tensor([length for start_idx, length in sublist])
                                                           for sublist in spans_per_row], batch_first=True,
                                                          padding_value=-1)
    unused_span_indices = span_starts_tensor == -1
    spans_to_replace = torch.zeros(span_starts_tensor.shape).where(unused_span_indices, torch.tensor(0.8))
    spans_to_replace = torch.bernoulli(spans_to_replace).bool()
    spans_to_randomize = torch.bernoulli(torch.full(spans_to_replace.shape,
                                                    0.5)).bool() & ~spans_to_replace & ~unused_span_indices

    for locs, lengths, replace, row_idx in zip(span_starts_tensor, span_lengths_tensor,
                                               spans_to_replace, range(input_ids.shape[0])):
        row_span_start_indices = locs[replace]
        row_span_lengths = lengths[replace]
        span_index_targets = torch.tensor([idx for start_loc, length in zip(row_span_start_indices, row_span_lengths)
                                           for idx in range(start_loc, start_loc + length)], dtype=torch.long)

        if span_index_targets.shape[0] != 0:
            input_ids[row_idx, span_index_targets] = CodepointTokenizer.MASK

    for locs, lengths, randomize, row_idx in zip(span_starts_tensor, span_lengths_tensor,
                                                 spans_to_randomize, range(input_ids.shape[0])):
        row_span_start_indices = locs[randomize]
        row_span_lengths = lengths[randomize]

        # for each span, select a random subword from the byte piece embedding vocab of the same length
        # and use it to replace the target characters
        for start_idx, span_len in zip(row_span_start_indices, row_span_lengths):
            replacement_word = random.choice(replacement_vocab[span_len.item()])
            codepoints = torch.tensor([ord(c) for c in replacement_word], dtype=torch.long)
            input_ids[row_idx, start_idx:start_idx + span_len] = codepoints

    masked_indices = torch.stack(all_masked_indices)
    return input_ids, labels, masked_indices


def random_span_mask(
        input_ids,
        attention_mask_matrix,
        attention_mask_sentence,
        masking_percent: float = 0.15,
        min_char_replacement: int = 5,
        max_char_replacement: int = 128):
    """The standard way to do this (how HuggingFace does it) is to randomly mask each token with masking_prob. However,
    this can result in a different number of masks for different sentences, which would prevent us from using gather()
    to save compute like CANINE does. consequently, we instead always mask exactly length * masking_prob tokens, and
    instead select those indices randomly. this means that the masked indices can (and likely will) be OUT OF ORDER.
    it also means that in batches with padding, potentially a higher % will be masked in the shorter sentences"""
    # input_ids: [max_char_len, max_num_word, batch_size]
    # attention_mask_sentence: [batch_size, max_num_word]: index to ignore
    labels = input_ids.copy()
    input_ids = input_ids.copy()
    max_char_len, max_num_word, batch_size = input_ids.shape

    # [batch_size,]
    sentence_len = (attention_mask_sentence == 0).sum(axis=1).astype(np.float)
    min_sentence_len = sentence_len.min()
    # exclude empty words, special tokens/ empty word (pad) are at the first char, ignore WORD_CLS (5)
    # [max_num_word, batch_size]
    special_tokens_mask = _special_tokens_mask_from_range(input_ids[0,:,:], range(MIN_SPECIAL_TOKEN, MAX_SPECIAL_TOKEN - 1))
    # attention_mask_sentence: [batch_size, max_num_word] -> [max_num_word, batch_size]
    special_tokens_mask = special_tokens_mask | attention_mask_sentence.transpose(1, 0).astype(bool)
    maskable_char_matrix = ~attention_mask_matrix
    # first char [WORD_CLS] is not maskable
    maskable_char_matrix[0, :, :] = False

    mask_count = math.floor(max_num_word * masking_percent)
    # some sentences are too short for masking
    # if mask_count > min_sentence_len:
    #    mask_count = math.floor(0.8 * (min_sentence_len - 2))  # <CLS> and <SEP>
    # print("mask_count:", mask_count)

    char_indices_to_mask = []
    min_mask_char = float('inf')
    char_mask = np.zeros((max_char_len, max_num_word, batch_size))
    for idx in range(batch_size):
        # inputs: [max_char_len, max_num_word]
        # maskable_mask_word: [max_num_word]
        inputs = input_ids[:, :, idx]
        maskable_mask_word = ~special_tokens_mask[:, idx]
        # mask -> indices [mask_count,]
        # print(idx, np.nonzero(maskable_mask_word)[0].shape)
        # randomly select

        maskable_indices_word = np.nonzero(maskable_mask_word)[0]
        maskable_indices_word_len = maskable_indices_word.shape[0]
        if mask_count < maskable_indices_word_len:
            maskable_indices_word_start = random.randint(0, maskable_indices_word_len - mask_count)
            maskable_indices_word = maskable_indices_word[maskable_indices_word_start: maskable_indices_word_start + mask_count]
        else:
            mask_count_i = math.floor(0.8 * maskable_indices_word_len)
            maskable_indices_word_start = random.randint(0, maskable_indices_word_len - mask_count_i)
            maskable_indices_word = maskable_indices_word[maskable_indices_word_start: maskable_indices_word_start + mask_count_i]

        maskable_mask_word = np.zeros((max_num_word,))
        maskable_mask_word[maskable_indices_word] = 1

        maskable_mask_char = np.zeros((max_char_len, max_num_word))
        maskable_mask_char[:, maskable_indices_word] = maskable_char_matrix[:, maskable_indices_word, idx]

        # adapt to trim order: [max_char_len, max_num_word] -> [max_num_word, max_char_len]
        maskable_indices_char = np.nonzero(maskable_mask_char.transpose((1, 0)))
        if len(maskable_indices_char[0]) < min_mask_char:
            min_mask_char = len(maskable_indices_char[0])

        # [batch_size,] (2,num_mask_char)
        char_indices_to_mask.append(maskable_indices_char)

    for idx in range(batch_size):
        # transpose back
        char_indices_to_mask[idx] = [char_indices_to_mask[idx][1][:min_mask_char],
                                     char_indices_to_mask[idx][0][:min_mask_char]]
        char_mask[char_indices_to_mask[idx][0],
                  char_indices_to_mask[idx][1],
                  idx] = 1
    random_flag = np.random.uniform(low=0.0, high=1.0, size=(max_num_word, batch_size))
    # 80% of the time, we replace masked input tokens with mask token
    indices_replaced = ((random_flag <= 0.8)[None, :, :] * char_mask).astype(bool)
    input_ids[indices_replaced] = CodepointTokenizer.MASK

    # 10% (half of the remaining 20%) of the time, we replace masked input tokens with random word
    indices_random = (((random_flag > 0.8) & (random_flag <= 0.9))[None, :, :] * char_mask).astype(bool)
    random_words = np.random.randint(
        low=min_char_replacement,
        high=max_char_replacement,
        size=(max_char_len, max_num_word, batch_size))
    input_ids[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged

    # [batch_size, 2, num_mask_char]
    char_indices_to_mask = torch.from_numpy(np.array(char_indices_to_mask)).long()

    # [max_char_len, max_num_word, batch_size] -> [batch_size, num_mask_char]
    labels = np.stack([labels[char_indices_to_mask[i, 0], char_indices_to_mask[i, 1], i] for i in range(batch_size)])

    return torch.from_numpy(input_ids).long(), torch.from_numpy(labels).long(), char_indices_to_mask  # .int(), char_mask


class RandomSpanMaskingDataCollator:
    def __init__(self, tokenizer: CodepointTokenizer, replacement_range: range):
        self.tokenizer = tokenizer
        self.replacement_range = replacement_range

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        padded_batch = self.tokenizer.pad_tensor_numpy(batch)

        input_ids, labels, masked_indices = random_span_mask(
            input_ids=padded_batch['input_ids'],
            attention_mask_matrix=padded_batch['attention_mask_matrix'],
            attention_mask_sentence=padded_batch['attention_mask_sentence'],
            min_char_replacement=self.replacement_range.start,
            max_char_replacement=self.replacement_range.stop)

        padded_batch.update({
            'input_ids': input_ids,
            'labels': labels,
            'predict_indices': masked_indices,
            'attention_mask_matrix': torch.from_numpy(padded_batch['attention_mask_matrix']),
            'attention_mask_sentence': torch.from_numpy(padded_batch['attention_mask_sentence'])
        })

        return padded_batch


class RandomMaskingDataCollator:
    def __init__(self, tokenizer: CodepointTokenizer, replacement_range: range, aggregation_method: str = "word_cls"):
        self.tokenizer = tokenizer
        self.replacement_range = replacement_range
        self.aggregation_method = aggregation_method

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        for sample in batch:
            for key in sample.keys():
                if isinstance(sample[key], str):
                    sample[key] = [int(item) for item in sample[key].split('#') if int(item) != self.tokenizer.SEP]
                    sample[key] += [self.tokenizer.SEP]
        model_inputs = self.tokenizer.prepare_model_inputs(batch, out_format="numpy", aggregation_method=self.aggregation_method)
        input_ids, labels, masked_indices = random_mask(input_ids=model_inputs['input_ids'],
                                                        attention_mask_matrix=model_inputs['attention_mask_matrix'],
                                                        attention_mask_sentence=model_inputs['attention_mask_sentence'],
                                                        min_char_replacement=self.replacement_range.start,
                                                        max_char_replacement=self.replacement_range.stop)
        #print(input_ids.shape)
        # TODO: padding the model inputs to multiple of 8 to increase the performance with tensor cores
        model_inputs.update({
            'input_ids': input_ids,
            'labels': labels,
            'predict_indices': masked_indices,
            'attention_mask_matrix': torch.from_numpy(model_inputs['attention_mask_matrix']),
            'attention_mask_sentence': torch.from_numpy(model_inputs['attention_mask_sentence'])
        })

        return model_inputs