#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import datetime
import logging
import os
import random
import sys
import time
import pickle
from dataclasses import dataclass, field
from functools import partial
from typing import Optional
import traceback

import datasets
import numpy as np
import torch
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    CanineForSequenceClassification,
    CanineTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import is_main_process
from utils.helpers import (
    BertClassificationDataCollator,
    CanineClassificationDataCollator,
    ClassificationDataCollator)

from model.hlm import HLMConfig, HLMForSequenceClassification
from model.codepoint_tokenizer import CodepointTokenizer

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

MAX_RETRIES = 5

logger = logging.getLogger(__name__)

def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def preprocess_feline(examples, tokenizer, task):
    # Prepare model inputs for feline
    sentence1_key, sentence2_key = task_to_keys[task]
    # texts = [examples[sentence1_key], ] if sentence2_key is None else [examples[sentence1_key], examples[sentence2_key]]
    # result = tokenizer.encode(texts)
    result = tokenizer.encode(examples[sentence1_key], examples.get(sentence2_key))
    input_ids = result["input_ids"]
    if examples.get(sentence2_key) is not None:
        sep_idx = input_ids.index(tokenizer.SEP)
        token_type_ids = [0] * (sep_idx + 1) + [1] * (len(input_ids) - sep_idx - 1)
        result["token_type_ids"] = token_type_ids
    else:
        result["token_type_ids"] = [0] * len(input_ids)
    if "label" in examples:
        result["labels"] = examples["label"]
    return result


def preprocess_standard(examples, tokenizer, task):
    # Prepare standard BERT-like model inputs
    sentence1_key, sentence2_key = task_to_keys[task]
    texts = [examples[sentence1_key], ] if sentence2_key is None else [examples[sentence1_key], examples[sentence2_key]]
    result = {
        "input_ids": [],
        "token_type_ids": [],
        "attention_mask": []
    }
    for n, text in enumerate(texts):
        ids = tokenizer.encode(tokenizer.tokenize(text))
        # Ignore [CLS] token for sentence 2
        input_ids = ids if n == 0 else ids[1:]
        result["input_ids"] += input_ids
        result["token_type_ids"] += ([0] * len(input_ids)) if n == 0 else ([1] * len(input_ids))
        result["attention_mask"] += [1] * len(input_ids)
    if "label" in examples:
        result["labels"] = examples["label"]
    return result


MODEL_MAP = {
    "bert-base-cased": (BertTokenizer, BertForSequenceClassification, BertClassificationDataCollator, preprocess_standard),
    "bert-base-multilingual-cased": (BertTokenizer, BertForSequenceClassification, BertClassificationDataCollator, preprocess_standard),
    "google/canine-c": (CanineTokenizer, CanineForSequenceClassification, CanineClassificationDataCollator, preprocess_standard),
    "google/canine-s": (CanineTokenizer, CanineForSequenceClassification, CanineClassificationDataCollator, preprocess_standard),
    "char2word": (CodepointTokenizer, HLMForSequenceClassification, ClassificationDataCollator, preprocess_feline)
}


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    task_name: str = field(
        default="sst2",
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    max_char_length: int = field(
        default=2048,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )

    char2int_dict: Optional[str] = field(default="pickle/char2int_dict.pkl", metadata={"help": "Relative path to codebook dict."})
    int2char_dict: Optional[str] = field(default="pickle/int2char_dict.pkl", metadata={"help": "Relative path to codebook dict."})

    def __post_init__(self):
        if self.task_name not in task_to_keys.keys():
            raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    random_init_model: bool = field(
        default=False,
        metadata={"help": "Will random init model."},
    )
    pretrained_checkpoint: str = field(
        default="main",
        metadata={"help": "path to pretrained_checkpoint."},
    )
    hidden_size: int = field(
        default=768,
        metadata={"help": "Hidden size."}
    )
    attention_heads: int = field(
        default=12,
        metadata={"help": "Number of attention heads."}
    )
    transformer_ff_size: int = field(
        default=3072,
        metadata={"help": "Transformer feed-forward size."}
    )
    local_transformer_ff_size: int = field(
        default=1536,
        metadata={"help": "Local transformer feed-forward size."}
    )
    dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout probability."}
    )
    activation: str = field(
        default="gelu",
        metadata={"help": "Activation function."}
    )
    for_token_classification: bool = field(
        default=False,
        metadata={"help": "Is the model for token classification task or not?"}
    )
    n_local_layer_first: int = field(
        default=3,
        metadata={"help": "Number of layers for first intra-word transformers."}
    )
    n_global_layer: int = field(
        default=6,
        metadata={"help": "Number of layers for inter-word transformers."}
    )
    n_local_layer_last: int = field(
        default=3,
        metadata={"help": "Number of layers for last intra-word transformers."}
    )
    position_type: Optional[str] = field(
        default="both",
        metadata={"help": "Type of position embeddings: char, word or both."}
    )
    max_char_per_word: Optional[int] = field(
        default=128,
        metadata={"help": "Maximum number of characters per word."}
    )
    max_num_word: Optional[int] = field(
        default=1024,
        metadata={"help": "Maximum number of words per sequence."}
    )
    use_token_type: Optional[bool] = field(
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
    aggregation_method: Optional[str] = field(
        default="word_cls",
        metadata={"help": "Character aggregation method."}
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--config_json',
        type=str,
        required=False,
        help='Config file for HfArgumentParser',
    )
    argparser.add_argument(
        '--output_dir',
        type=str,
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
    model_args, data_args, training_args = None, None, None
    try:
        args = argparser.parse_args()
        logger.setLevel(logging.INFO if is_main_process(args.local_rank) else logging.WARN)
        if args.config_json is not None:
            logger.info("Reading arguments from JSON config file...")
            model_args, data_args, training_args = parser.parse_json_file(args.config_json)
    except:
        args = None
        logger.warning("No valid config file provided. Parsing list of arguments instead.")
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if any([model_args, data_args, training_args]) is None:
        raise ValueError("Error parsing input arguments.")
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info("***************************************")
    logger.info("PyTorch version: %s" % torch.__version__)
    logger.info("Transformers version: %s" % transformers.__version__)
    logger.info("***************************************")
    logger.info("Environnement variables:")
    for k, v in os.environ.items():
        if k.startswith("AZ"):
            continue
        logger.info("%s: %s" % (k, v))
    logger.info("***************************************")

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    if args is not None:
        output_dir = os.path.join(args.output_dir, training_args.output_dir + '@' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        training_args.output_dir = output_dir

    os.makedirs(training_args.output_dir, exist_ok=True)

    training_args.logging_dir = training_args.output_dir
    logger.info("output_dir=%s" % (training_args.output_dir))

    # Set seed before initializing model.
    set_seed(training_args.seed)

    logger.info("Data parameters:\n%s" % data_args)
    logger.info("Model parameters:\n%s" % model_args)
    logger.info("Training/evaluation parameters:\n%s" % training_args)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    raw_datasets = None
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        with training_args.main_process_first(desc="Loading raw dataset"):
            for n_try in range(MAX_RETRIES):
                try:
                    raw_datasets = load_dataset(
                        "glue",
                        data_args.task_name,
                        token=True if model_args.use_auth_token else None)
                    logger.info("Process %s - Try %s: Loading raw dataset succeeded." % (training_args.local_rank, n_try + 1))
                    break
                except Exception as e:
                    print(traceback.format_exc())
                    logger.warning("Process %s - Try %s: Loading raw dataset failed: %s" % (training_args.local_rank, n_try + 1, str(e)))
                    time.sleep(random.randint(n_try + 1, 5 * (n_try + 1)))

    if raw_datasets is None:
        raise RuntimeError("Unable to load dataset.")
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    tokenizer_class, model_class, data_class, preprocess_function = MODEL_MAP[model_args.model_name_or_path]
    if model_args.model_name_or_path == "char2word":
        char2int_dict = load_pkl(data_args.char2int_dict)
        int2char_dict = load_pkl(data_args.int2char_dict)
        tokenizer = tokenizer_class(
            char2int_dict,
            int2char_dict,
            max_num_word=model_args.max_num_word,
            max_char_len=data_args.max_char_length,
            max_char_per_word=model_args.max_char_per_word,
            word_context=model_args.word_context)
        config = HLMConfig(**model_args.__dict__)
        config.use_token_type = model_args.use_token_type
        config.use_projection = model_args.use_projection
        config.word_context = model_args.word_context
        config.relative_attention = model_args.relative_attention
        config.max_codepoint = tokenizer.max_codepoint
        config.aggregation_method = model_args.aggregation_method
        model = model_class(config, vocab_size=num_labels)
        if model_args.pretrained_checkpoint is not None:
            logger.info("Load weight from %s..." % model_args.pretrained_checkpoint)
            ckpt = torch.load(model_args.pretrained_checkpoint)
            msg = model.load_state_dict(ckpt, strict=False)
            logger.info(msg)
    else:
        tokenizer = tokenizer_class.from_pretrained(model_args.model_name_or_path)
        model = model_class.from_pretrained(model_args.model_name_or_path)
    model.num_labels = num_labels
    logger.info("Number of trainable parameters in model: %s" % sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Preprocessing the raw_datasets
    with training_args.main_process_first(desc="Dataset map pre-processing"):
        for n_try in range(MAX_RETRIES):
            try:
                raw_datasets = raw_datasets.map(
                    partial(preprocess_function, tokenizer=tokenizer, task=data_args.task_name),
                    batched=False,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on dataset")
                logger.warning("Process %s - Try %s: Dataset tokenization succeded." % (training_args.local_rank, n_try + 1))
                break
            except Exception as e:
                logger.warning("Process %s - Try %s: Dataset tokenization failed: %s" % (training_args.local_rank, n_try + 1, str(e)))
                time.sleep(random.randint(n_try + 1, 5 * (n_try + 1)))
        if n_try == 3:
            raise RuntimeError("Unable to tokenize dataset.")

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        logger.info("#train_examples=%s" % len(train_dataset))
        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            sample = train_dataset[index]
            logger.info(f"Sample {index} of the training set: {sample}.")
            logger.info("Decoded text: %s" % tokenizer.decode(sample["input_ids"]))

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        logger.info("#eval_examples=%s" % len(eval_dataset))

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        logger.info("#test_examples=%s" % len(predict_dataset))

    # Get the metric function
    if data_args.task_name is not None:
        metric = load_metric("glue", data_args.task_name)
    else:
        metric = load_metric("accuracy")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    data_collator = data_class(tokenizer)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(raw_datasets["validation_mismatched"])
            combined = {}

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            if task == "mnli-mm":
                metrics = {k + "_mm": v for k, v in metrics.items()}
            if task is not None and "mnli" in task:
                combined.update(metrics)

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", combined if task is not None and "mnli" in task else metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        predict_datasets = [predict_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            predict_datasets.append(raw_datasets["test_mismatched"])

        for predict_dataset, task in zip(predict_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            predict_dataset = predict_dataset.remove_columns("label")
            predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

            output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    logger.info(f"***** Predict results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")

    kwargs = {
        "finetuned_from": model_args.model_name_or_path if model_args.model_name_or_path else model_args.pretrained_checkpoint,
        "tasks": "text-classification"
    }
    if data_args.task_name is not None:
        kwargs["language"] = "en"
        kwargs["dataset_tags"] = "glue"
        kwargs["dataset_args"] = data_args.task_name
        kwargs["dataset"] = f"GLUE {data_args.task_name.upper()}"

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
