import csv
import pickle
import argparse
from datasets import load_dataset

from model.codepoint_tokenizer import CodepointTokenizer
csv.field_size_limit(100000000)

def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess data for tokenizer")
    parser.add_argument("--csv_file", type=str, default="./data/sample_pretrain_data_raw.csv", help="Path to the input CSV file")
    parser.add_argument("--out_file", type=str, default="./data/sample_pretrain_data_processed.csv", help="Path to the output CSV file")
    return parser.parse_args()

args = parse_args()
csv_file = args.csv_file
out_file = args.out_file

char2int_dict = load_pkl("./data/char2int_dict.pkl")
int2char_dict = load_pkl("./data/int2char_dict.pkl")
max_num_word = 512
max_char_per_word = 20
max_rows = -1
logging_step = 100000

streaming = True
dataset = load_dataset(
    "csv",
    data_files=[csv_file],
    column_names=[],
    split='train',
    streaming=streaming)
tokenizer = CodepointTokenizer(
    char2int_dict,
    int2char_dict,
    max_num_word=max_num_word,
    max_char_per_word=max_char_per_word)
    
max_words = max_num_word - 2
sequence = []
n_words = 0
n_samples = 0
cnt = 0
with open(out_file, 'w', encoding="UTF-8") as f:
    writer = csv.writer(f)
    for sample in dataset:
        cnt += 1
        if max_rows > 0 and cnt > max_rows:
            break
        if cnt % logging_step == 0:
            print("cnt=%s" % cnt)
        for k, v in sample.items():
            v = v.replace("\n", " ")
            encoding = tokenizer.encode(v)
            input_ids = encoding["input_ids"]
            if len(input_ids) < 3:
                continue
            input_ids = input_ids[1:-1]
            n_words_sample = sum(idx == tokenizer.WORD_CLS for idx in input_ids)
            if n_words + n_words_sample >= max_words - 2:  # Reserve room for [CLS] and [SEP] tokens
                n_samples += 1
                seq = [tokenizer.CLS] + sequence + [tokenizer.SEP]
                # print(tokenizer.decode(seq))
                row = "#".join(str(x) for x in seq)
                writer.writerow([row])
                sequence = input_ids
                n_words = n_words_sample
            else:
                sequence += input_ids
                n_words += n_words_sample
print("#samples=%s" % n_samples)
print(out_file)