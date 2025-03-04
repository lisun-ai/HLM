import logging
import numpy as np
import os
import re, string
import torch
from typing import List, Dict

#import spacy
#from langdetect import detect
# import nltk
# nltk.download('punkt')

logger = logging.getLogger(__name__)

def load_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)
        
class CodepointTokenizer:
    # padding is always 0, which is fine because it's the null unicode point
    # the remaining code points are private use codepoints, see https://en.wikipedia.org/wiki/Private_Use_Areas
    PAD = 0
    CLS = 1
    SEP = 2
    MASK = 3
    UNK = 4
    WORD_CLS = 5

    WORD_SPLITTERS = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~，。：；（）'
    SPACE_CHARS = '\t\n\x0b\x0c\r\x1c\x1d\x1e\x1f \x85\xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000'

    _READABLE_SPECIAL_TOKENS = {
         PAD: '[PAD]',
         CLS: '[CLS]',
         SEP: '[SEP]',
         MASK: '[MASK]',
         UNK: '[UNK]',
         WORD_CLS: '[WORD_CLS]'
    }

    _LANGUAGE_TO_MODULE = {
        "ca": "ca_core_news_sm",
        "zh": "zh_core_web_sm",
        "hr": "hr_core_news_sm",
        "da": "da_core_news_sm",
        "nl": "nl_core_news_sm",
        "en": "en_core_web_sm",
        "fi": "fi_core_news_sm",
        "fr": "fr_core_news_sm",
        "de": "de_core_news_sm",
        "el": "el_core_news_sm",
        "it": "it_core_news_sm",
        "ja": "ja_core_news_sm",
        "ko": "ko_core_news_sm",
        "lt": "lt_core_news_sm"
    }

    def __init__(
            self,
            char2int_dict,
            int2char_dict,
            max_num_word=128,
            max_char_len=2048,
            max_char_per_word=20,
            word_context=1,
            language="en"):
        self.char2int_dict = char2int_dict
        self.int2char_dict = int2char_dict
        # language_module = self._LANGUAGE_TO_MODULE[language if language in self._LANGUAGE_TO_MODULE else "en"]
        # os.system("python -m spacy download %s" % language_module)
        # self.word_tokenizer = spacy.load(language_module).tokenizer
        # # self.word_tokenizer = nltk.word_tokenize
        self.max_num_word = max_num_word
        self.max_char_len = max_char_len
        self.max_char_per_word = max_char_per_word

        self.max_codepoint = max(1024, max(int2char_dict.keys()) + 1)
        self.word_context = word_context

    def tokenize(self, text):
        # Custom word splitter based on punctuation
        result = []
        word = ""
        for c in text:
            if c in self.WORD_SPLITTERS + self.SPACE_CHARS:
                if word:
                    result.append(word)
                if c not in self.SPACE_CHARS:
                    result.append(c)
                word = ""
                continue
            word += c
        if word:
            result.append(word)
        return result

    # Alternative word tokenization methods
    # def tokenize(self, text):
    #     # NLTK word tokenizer
    #     result = self.word_tokenizer(text)
    #     return result

    # def tokenize(self, text):
    #     # Spacy word tokenizer
    #     result = self.word_tokenizer(text)
    #     return [token.text for token in result]

    def save_pretrained(self, output_dir):
        # There is no weight in our tokenizer
        pass

    def encode_word(self, word):
        # Encoded word always starts with [WORD_CLS] token ID
        word = word.strip()
        ids = [self.WORD_CLS]
        word_mask = [1] + [0] * len(word)
        for c in word:
            ids.append(self.char2int_dict.get(c, self.UNK))
        return {
            'input_ids': ids,
            'word_mask': word_mask
        }

    def is_char_split(self, word):
        try:
            lang = detect(word)
            return lang in ["ja", "zh-cn", "zh-tw", "th"]
        except:
            return False

    def encode_sequence(self, sequence: str, add_cls: bool = False, add_sep: bool = False, language: str = None) -> Dict[str, list]:
        words = self.tokenize(sequence)
        ids = []
        word_mask = []
        if add_cls:
            ids += [self.CLS]
            word_mask += [1]
        for word in words:
            # if self.is_char_split(word):
            if language and language in ["ja", "zh", "th"]:
                chars = [c for c in word]
            else:
                chars = [word]
            for w in chars:
                encoded_word = self.encode_word(w)
                ids.extend(encoded_word["input_ids"])
                word_mask.extend(encoded_word["word_mask"])
        if add_sep:
            ids.append(self.SEP)
            word_mask.append(1)
        return {
            'input_ids': ids,
            'word_mask': word_mask
        }

    def encode(self, sentence1, sentence2=None, language=None):
        encoding1 = self.encode_sequence(sentence1, add_cls=True, add_sep=True, language=language)
        if sentence2 is None:
            return encoding1
        encoding2 = self.encode_sequence(sentence2, add_cls=False, add_sep=True, language=language)
        # Concatenate both encoding
        for key in encoding2.keys():
            encoding1[key] += encoding2[key]
        return encoding1

    def decode(self, ids: List[int]) -> str:
        return ''.join(
            self._READABLE_SPECIAL_TOKENS.get(i, self.int2char_dict.get(i, self._READABLE_SPECIAL_TOKENS[self.UNK])) for i in ids
        )

    def _get_max_chars(self, word_masks):
        max_chars = 0
        all_word_indexes = []
        for mask in word_masks:
            word_indexes = np.where(np.array(mask) == 1)[0]
            word_lengths = np.diff(np.append(word_indexes, len(mask)))
            max_chars = max(max_chars, np.max(word_lengths))
            all_word_indexes.append(word_indexes)
        return int(max_chars), all_word_indexes

    def prepare_sample(self, json_path):
        try:
            json_data = load_json(json_path, n_tries=3)
            encoding = self.encode(json_data["text"], language=json_data["lang"])
            return encoding
        except Exception as e:
            logger.warning("%s: Unable to encode data: %s" % (json_path, str(e)))
            return None

    def prepare_model_inputs(self, batch: List[Dict[str, list]], out_format: str = "torch", aggregation_method: str = "word_cls") -> Dict[str, torch.Tensor]:
        if isinstance(batch[0], str):
            batch = [self.prepare_sample(x) for x in batch]
        input_ids = [encoding["input_ids"] for encoding in batch if encoding is not None]
        word_masks = []
        for ids_list in input_ids:
            word_mask = [int(idx in [self.CLS, self.SEP, self.WORD_CLS]) for idx in ids_list]
            word_masks.append(word_mask)
        # word_masks = [encoding["word_mask"] for encoding in batch  if encoding is not None]
        batch_size = len(input_ids)
        max_chars, word_indexes = self._get_max_chars(word_masks)
        max_words = int(max([sum(mask) for mask in word_masks]))
        if max_words % self.word_context != 0:
            # Force max_words to be divisible by word_context
            max_words = int(np.ceil(max_words / self.word_context) * self.word_context)
        # [max_chars, max_words, batch_size]
        encoded_chars_mat = self.PAD * np.ones((max_chars, max_words, batch_size), dtype=int)
        for bidx, word_index in enumerate(word_indexes):
            n_words = len(word_index)
            for widx in range(n_words):
                s = word_index[widx]
                e = word_index[widx + 1] if widx + 1 < n_words else len(input_ids[bidx])
                n_chars = e - s
                encoded_chars_mat[:n_chars, widx, bidx] = input_ids[bidx][s:e]
        if max_chars > self.max_char_per_word:
            # Trim too long words
            encoded_chars_mat = encoded_chars_mat[:self.max_char_per_word, :, :]
            max_chars = self.max_char_per_word
        if max_words > self.max_num_word:
            # Trim too many words
            encoded_chars_mat = encoded_chars_mat[:, :self.max_num_word, :]
        encoded_chars_mat = torch.from_numpy(encoded_chars_mat).long() if out_format == "torch" else encoded_chars_mat.astype(int)
        attention_mask_sentence = encoded_chars_mat[0, :, :] == self.PAD
        attention_mask_matrix = encoded_chars_mat == self.PAD
        
        model_inputs = {
            "input_ids": encoded_chars_mat,
            "attention_mask_matrix": attention_mask_matrix,
            "attention_mask_sentence": attention_mask_sentence.transpose(1, 0) if out_format == "torch" else np.transpose(attention_mask_sentence)
        }
        return model_inputs
