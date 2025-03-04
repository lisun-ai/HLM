import argparse
from collections import defaultdict
import datetime
import logging
import os
import random
import sys
import torch
import pickle
import transformers
from tqdm import tqdm
from transformers import HfArgumentParser, set_seed, Trainer
from transformers.trainer_utils import is_main_process
from datasets import (
    interleave_datasets,
    load_dataset,
    load_from_disk,
    concatenate_datasets)


from utils.helpers import DataArguments, HLMTrainingArguments
from utils.masking import RandomSpanMaskingDataCollator, RandomMaskingDataCollator
from model.codepoint_tokenizer import CodepointTokenizer
from model.hlm import HLMConfig, HLMForMaskedLanguageModeling


logger = logging.getLogger(__name__)

def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def load_data(datasets, streaming):
    train_data = load_dataset("csv", data_files=[datasets], \
                          column_names=['input_ids'], split='train', streaming=streaming)
    train_data = train_data.with_format("torch")
    return train_data


def main():
    transformers.logging.set_verbosity_info()
    argparser = argparse.ArgumentParser(description='Hierarchical LM pretraining')
    argparser.add_argument(
        '--config_json',
        type=str,
        required=True,
        help='Config file for HfArgumentParser',
    )
    argparser.add_argument(
        '--output_dir',
        type=str,
        default="",
        required=False,
        help='Output root directory.',
    )
    argparser.add_argument(
        '--local_rank',
        type=int,
        default=-1,
        help='For distributed training: local_rank',
    )
    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    data_args, training_args = None, None
    args = argparser.parse_args()
    parser = HfArgumentParser((DataArguments, HLMTrainingArguments))
    try:
        logger.setLevel(logging.INFO if is_main_process(args.local_rank) else logging.WARN)
        if args.config_json is not None:
            logger.info("Reading arguments from JSON config file %s..." % args.config_json)
            data_args, training_args = parser.parse_json_file(args.config_json)
    except Exception as e:
        logger.warning("Unable to parse config file: %s. Parsing list of arguments instead." % str(e))
        data_args, training_args = parser.parse_args_into_dataclasses()
    if data_args is None or training_args is None:
        raise ValueError("Error parsing input arguments.")
    set_seed(training_args.seed)
    logger.info("***************************************")
    logger.info("Python version: %s" % sys.version.replace("\n", " "))
    logger.info("PyTorch version: %s" % torch.__version__)
    logger.info("CUDA version: %s" % torch.version.cuda)
    logger.info("Transformers version: %s" % transformers.__version__)
    logger.info("***************************************")
    # output_dir = os.path.join(args.output_dir, training_args.output_dir + '@' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    output_dir = os.path.join(args.output_dir, training_args.output_dir)
    training_args.output_dir = output_dir

    os.makedirs(output_dir, exist_ok=True)
    
    training_args.logging_dir = output_dir
    logger.info("data_dir=%s output_dir=%s" % (data_args.datasets, output_dir))
    logger.info("Loading codebook...")
    char2int_dict = load_pkl(data_args.char2int_dict)
    int2char_dict = load_pkl(data_args.int2char_dict)
    logger.info("min_code=%s max_code=%s" % (min(int2char_dict.keys()), max(int2char_dict.keys())))
    tokenizer = CodepointTokenizer(
        char2int_dict,
        int2char_dict,
        max_num_word=training_args.max_num_word,
        max_char_len=training_args.max_char_length,
        max_char_per_word=training_args.max_char_per_word,
        word_context=training_args.word_context)
    if training_args.masking_type == 'bpe_span':
        logger.info('BPE based-span masking')
        data_collator = RandomSpanMaskingDataCollator(tokenizer, True)
    elif training_args.masking_type == 'rand_span':
        logger.info('Random span masking')
        data_collator = RandomSpanMaskingDataCollator(tokenizer, False)
    elif training_args.masking_type == 'rand_char':  # default
        logger.info('Random character masking')
        # char range: https://stackoverflow.com/a/30200250/4243650
        # we aren't including half width stuff
        # default replacement range: (8, 128)
        logger.info("masking_percent: %s" % training_args.masking_percent)
        data_collator = RandomMaskingDataCollator(
            tokenizer,
            range(8, training_args.random_mask_max_range),
            aggregation_method=training_args.aggregation_method)
    else:
        raise RuntimeError('Unknown masking type')

    logger.info("Data parameters:\n%s" % data_args)
    logger.info("Training parameters:\n%s" % training_args)
    logger.info("Tokenizer codebook size: %s" % tokenizer.max_codepoint)

    # Load pretraining datasets
    train_data = load_data(data_args.datasets, streaming=data_args.streaming)

    # Log a few samples from the training set
    if data_args.streaming:
        samples = list(train_data.take(2))
        for sample in samples:
            input_ids = [int(item) for item in sample['input_ids'].split('#') if int(item) != tokenizer.SEP]
            input_ids += [tokenizer.SEP]
            logger.info(f"Sample of the training set: {input_ids}.")
            logger.info("Decoded text: %s" % tokenizer.decode(input_ids))
    else:
        for index in random.sample(range(len(train_data)), 3):
            sample = train_data[index]
            if isinstance(sample, str):
                sample = tokenizer.prepare_sample(sample)
                input_ids = sample["input_ids"]
            elif "#" in sample['input_ids']:
                input_ids = [int(item) for item in sample['input_ids'].split('#') if int(item) != tokenizer.SEP]
                input_ids += [tokenizer.SEP]
            else:
                input_ids = sample["input_ids"]
            logger.info(f"Sample {index} of the training set: {input_ids}.")
            logger.info("Decoded text: %s" % tokenizer.decode(input_ids))

    config = HLMConfig(**training_args.__dict__)
    config.use_token_type = training_args.use_token_type
    config.use_projection = training_args.use_projection
    config.word_context = training_args.word_context
    config.max_codepoint = tokenizer.max_codepoint
    config.word_context = training_args.word_context
    config.relative_attention = training_args.relative_attention
    config.aggregation_method = training_args.aggregation_method
    model = HLMForMaskedLanguageModeling(config, vocab_size=config.max_codepoint)

    checkpoint_dir = None
    if training_args.resume_from_checkpoint:
        training_args.resume_from_checkpoint = os.path.join(args.output_dir, training_args.resume_from_checkpoint)
        if training_args.load_only_model:
            model.load_state_dict(torch.load(training_args.resume_from_checkpoint))
        else:
            checkpoint_dir = training_args.resume_from_checkpoint
        logger.info("Resuming from checkpoint: %s" % training_args.resume_from_checkpoint)
    logger.info("Number of trainable parameters in model: %s" % sum(p.numel() for p in model.parameters() if p.requires_grad))
    logger.info(training_args)
    trainer = Trainer(model=model,
                      args=training_args,
                      data_collator=data_collator,
                      train_dataset=train_data)
    trainer.train(resume_from_checkpoint=checkpoint_dir)


if __name__ == '__main__':
    main()
