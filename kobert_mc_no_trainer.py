#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
"""Fine-tuning a 🤗 Transformers model on multiple choice relying on the accelerate library without using a Trainer."""
# You can also adapt this script on your own multiple choice task. Pointers for this are left as comments.

import argparse
import json
import logging
import math
import os
import random
import shutil
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Optional, Union

import datasets
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import evaluate
import transformers
from kobert_transformers import get_tokenizer
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import PaddingStrategy, check_min_version, get_full_repo_name, send_example_telemetry
from utils import CLDatasetForMultiChoice, nt_xent, nt_xent_sup

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.22.0.dev0")

logger = get_logger(__name__)
# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a multiple choice task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default='skt/kobest_v1',
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--cl_method", type=str, default=None, help="A contrastive learning method."
    )
    parser.add_argument(
        "--random_span_mask", type=float, default=0, help="Random masking ratio."
    )
    parser.add_argument(
        "--alpha", type=float, default=1, help="Contrastive loss coefficient."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.2, help="Contrastive loss temperature."
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early stopping patience."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    args = parser.parse_args()

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              if provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = list(chain(*flattened_features))

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


def main():
    args = parse_args()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry(f"run_{args.dataset_config_name}_no_trainer", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["logging_dir"] = args.output_dir
        
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    if args.dataset_config_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # Trim a number of training examples
    if args.debug:
        for split in raw_datasets.keys():
            raw_datasets[split] = raw_datasets[split].select(range(100))
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    if raw_datasets["train"] is not None:
        column_names = raw_datasets["train"].column_names
    else:
        column_names = raw_datasets["validation"].column_names

    # When using your own dataset or a different dataset from swag, you will probably need to change this.
    if args.dataset_config_name == 'swag':
        ending_names = [f"ending{i}" for i in range(4)]
        context_name = "sent1"
        question_header_name = "sent2"
        example_length = 4
    elif args.dataset_config_name == 'hellaswag':
        ending_names = [f"ending_{i}" for i in range(1,5)]
        context_name = "context"
        question_header_name = None
        example_length = 4
    # elif args.dataset_config_name == 'copa':
    else:
        ending_names = [f"alternative_{i}" for i in range(1,3)]
        example_length = 2
        # context_name = "premise"
        # question_header_name = None
        
    label_column_name = "label" if "label" in column_names else "labels"

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name or args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path,
            output_hidden_states=True,
            output_attentions=True,)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer_name = args.tokenizer_name
    elif args.model_name_or_path:
        tokenizer_name = args.model_name_or_path
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
        
    if 'kobert' in tokenizer_name:
        tokenizer = get_tokenizer()
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    

    if args.model_name_or_path:
        model = AutoModelForMultipleChoice.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMultipleChoice.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        if args.dataset_config_name == 'swag' or args.dataset_config_name == 'hellaswag':
            first_sentences = [[context] * example_length for context in examples[context_name]]
            if question_header_name is not None: # swag
                question_headers = examples[question_header_name]
                second_sentences = [
                    [f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
                ]
            else: # kobert hellaswag
                second_sentences = [[f"{examples[end][i]}" for end in ending_names] for i in range(len(examples[ending_names[0]]))]
            # Flatten out

            first_sentences = list(chain(*first_sentences))
            second_sentences = list(chain(*second_sentences))
                        
        elif args.dataset_config_name == 'copa':
            is_cause = examples['question']
            first_sentences, second_sentences = [], []
            for i in range(len(examples['label'])):
                is_cause = examples['question'][i]
                a1 = examples['alternative_1'][i]
                a2 = examples['alternative_2'][i]
                p = examples['premise'][i]
                if is_cause == '원인':
                    first_sentences.append(a1)
                    first_sentences.append(a2)
                    second_sentences.append(p)
                    second_sentences.append(p)
                else: # '결과'
                    first_sentences.append(p)
                    first_sentences.append(p)
                    second_sentences.append(a1)
                    second_sentences.append(a2)
                    
        labels = examples[label_column_name]

        # Tokenize
        tokenized_examples = tokenizer(
            first_sentences,
            second_sentences,
            max_length=args.max_length,
            padding=padding,
            truncation=True,
        )

        # Un-flatten
        tokenized_inputs = {k: [v[i : i + example_length] for i in range(0, len(v), example_length)] for k, v in tokenized_examples.items()}
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function, 
            batched=True, 
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )
    eval_dataset = processed_datasets["validation"]
    test_dataset = processed_datasets['test']
    if args.cl_method is None:
        train_dataset = processed_datasets["train"]
    else:
        train_dataset = CLDatasetForMultiChoice(raw_datasets['train'], tokenizer, args)
    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 2):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
        if args.cl_method is None:
            logger.info(f"Tokenized result {index}: {tokenizer.batch_decode(train_dataset[index]['input_ids'])}.")
    # DataLoaders creation:
    def collate_fn_batch_encoding_pairwise(batch):
        try:
            sent1, sent2, label = zip(*batch)
            _sent1_toks = tokenizer(
                list(chain(*sent1)), 
                list(chain(*sent2)), 
                max_length=args.max_length, 
                padding=True, 
                truncation=True, 
                add_special_tokens=True, 
                return_tensors="pt")
            _sent2_toks = tokenizer(
                list(chain(*sent1)), 
                list(chain(*sent2)), 
                max_length=args.max_length, 
                padding=True, 
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt")
            sent1_toks = {k: torch.stack([v[i : i + example_length] for i in range(0, len(v), example_length)]) for k, v in _sent1_toks.items()}
            sent2_toks = {k: torch.stack([v[i : i + example_length] for i in range(0, len(v), example_length)]) for k, v in _sent2_toks.items()}
            sent1_length = _sent1_toks.attention_mask.shape[-1]
            sent2_length = _sent2_toks.attention_mask.shape[-1]
            if sent1_length > sent2_length:
                sent2_toks = tokenizer(
                    list(chain(*sent1)), 
                    list(chain(*sent2)), 
                    max_length=sent1_length, 
                    padding='max_length', 
                    truncation=True,
                    add_special_tokens=True,
                    return_tensors="pt")
                sent2_toks = {k: torch.stack([v[i : i + example_length] for i in range(0, len(v), example_length)]) for k, v in _sent2_toks.items()}
            elif sent1_length < sent2_length:
                sent1_toks = tokenizer(
                    list(chain(*sent1)), 
                    list(chain(*sent2)), 
                    max_length=sent2_length, 
                    padding='max_length', 
                    truncation=True,
                    add_special_tokens=True,
                    return_tensors="pt")
                sent1_toks = {k: torch.stack([v[i : i + example_length] for i in range(0, len(v), example_length)]) for k, v in _sent1_toks.items()}
        except:
            breakpoint()
        # add labels
        sent1_toks['labels'] = torch.LongTensor(label)
        return sent1_toks, sent2_toks
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorForMultipleChoice(
            tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
        )

    if args.cl_method is not None:
        data_collator_train = collate_fn_batch_encoding_pairwise
    else:
        data_collator_train = data_collator
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator_train, batch_size=args.per_device_train_batch_size, drop_last=True
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Use the device given by the `accelerator` object.
    device = accelerator.device
    model.to(device)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, test_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers(f'{args.dataset_config_name}_{args.seed}', experiment_config)

    # Metrics
    # metric = evaluate.load("accuracy")   
    metric = evaluate.load("f1")

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    # early stopping
    EARLY_STOP = (args.patience is not None and args.patience > 0)
    if EARLY_STOP:
        best_acc = 0
        best_epoch = 0
        best_train_loss = 0
        early_stop_cnt = 0
        best_completed_steps = 0
    cl_loss = 0
    for epoch in range(starting_epoch, args.num_train_epochs):
        if EARLY_STOP and early_stop_cnt > args.patience: # stop condition for early stop
            break
        model.train()
        if args.with_tracking:
            total_loss = 0
        # train loop
        if args.cl_method is None:
            print('No CL')
            for step, batch in enumerate(train_dataloader):
                # We need to skip steps until we reach the resumed step

                if args.resume_from_checkpoint and epoch == starting_epoch:
                    if resume_step is not None and step < resume_step:
                        completed_steps += 1
                        continue

                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = outputs.loss
                    # We keep track of the loss at each epoch
                    if args.with_tracking:
                        total_loss += loss.detach().float()
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1
                # save per N checkpoint steps
                if isinstance(checkpointing_steps, int) and not EARLY_STOP:
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps }"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        accelerator.save_state(output_dir)

                if completed_steps >= args.max_train_steps:
                    break
        else:
            print('Use CL')
            for step, (batch1, batch2) in enumerate(train_dataloader):
                # We need to skip steps until we reach the resumed step

                if args.resume_from_checkpoint and epoch == starting_epoch:
                    if resume_step is not None and step < resume_step:
                        completed_steps += 1
                        continue

                with accelerator.accumulate(model):
                    # breakpoint()
                    outputs1 = model(**batch1)
                    outputs2 = model(**batch2)
                    loss = outputs1.loss
                    # We keep track of the loss at each epoch
                    if args.with_tracking:
                        total_loss += loss.detach().float()
                    try:
                        if args.cl_method is not None:
                            if args.cl_method == 'scl':
                                scl_loss = 0
                                for i in range(args.per_device_train_batch_size):
                                    pooled1 = model.bert(input_ids=batch1['input_ids'][i], attention_mask=batch1['attention_mask'][i], token_type_ids=batch1['token_type_ids'][i]).pooler_output
                                    pooled2 = model.bert(input_ids=batch2['input_ids'][i], attention_mask=batch2['attention_mask'][i], token_type_ids=batch2['token_type_ids'][i]).pooler_output
                                
                                    scl_loss = scl_loss + nt_xent_sup(pooled1, pooled2, F.one_hot(batch1['labels'][i], num_classes=example_length), args.temperature)
                                train_loss = (loss + (scl_loss * args.alpha)) / args.gradient_accumulation_steps
                            else:
                                cont = torch.cat([outputs1.hidden_states[-1][:,0,:], outputs2.hidden_states[-1][:,0,:]], dim=-1).view(-1, outputs1.hidden_states[-1][0].shape[-1])

                                cl_loss = nt_xent(cont, args.temperature)
                                train_loss = (loss + (cl_loss * args.alpha)) / args.gradient_accumulation_steps
                            # breakpoint()
                    except:
                        breakpoint()
                    accelerator.backward(train_loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    
                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1
                    if completed_steps % 100 == 0:
                        if args.cl_method == 'scl':
                            print(f'{completed_steps=}, loss={loss.item()}, scl_loss={scl_loss.item()}, train_loss={train_loss.item()}')
                        else:
                            print(f'{completed_steps=}, loss={loss.item()}, cl_loss={cl_loss.item()}, train_loss={train_loss.item()}')
                # save per N checkpoint steps
                if isinstance(checkpointing_steps, int) and not EARLY_STOP:
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps }"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        accelerator.save_state(output_dir)

                if completed_steps >= args.max_train_steps:
                    break
        # Validation loop
        model.eval()
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
            metric.add_batch(
                predictions=predictions,
                references=references,
            )
            
        eval_metric = metric.compute(average='macro')
        accelerator.print(f"epoch {epoch}: {eval_metric}")
        if args.with_tracking:
            accelerator.log(
                {
                    "f1": eval_metric,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )
        output_dir = os.path.join(args.output_dir, f"epoch_{epoch}")
        # Save validation results per epoch, no early stop
        if args.checkpointing_steps == "epoch" and not EARLY_STOP:
            accelerator.save_state(output_dir)
            
        # Save validation results, with early stopping
        if EARLY_STOP:
            if eval_metric["f1"] > best_acc:
                best_acc = eval_metric["f1"]
                early_stop_cnt = 0
                best_epoch = epoch
                best_completed_steps = completed_steps
                best_train_loss = total_loss.item() / len(train_dataloader)
                accelerator.save_state(output_dir)
                with open(os.path.join(output_dir, "eval_results.json"), "w") as f:
                    json.dump({"eval_f1": eval_metric["f1"]}, f)
            else:
                early_stop_cnt += 1
        else:
            with open(os.path.join(output_dir, "eval_results.json"), "w") as f:
                json.dump({"eval_f1": eval_metric["f1"]}, f)
    # Test loop
    # if early stop, use best model.
    # else, use last model
    if EARLY_STOP:
        output_dir = os.path.join(args.output_dir, f"epoch_{best_epoch}")
        accelerator.load_state(output_dir)
    for step, batch in enumerate(test_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = accelerator.gather_for_metrics((predictions, batch["labels"]))
            metric.add_batch(
                predictions=predictions,
                references=references,
            )
            
    test_metric = metric.compute(average='macro')
    curr_train_loss = best_train_loss if EARLY_STOP else total_loss.item() / len(train_dataloader)
    accelerator.print(f"epoch {epoch}: {test_metric}")
    curr_epoch = best_epoch if EARLY_STOP else epoch
    curr_steps = best_completed_steps if EARLY_STOP else completed_steps
    
    if args.with_tracking:
        # accelerator.log(
        #     {
        #         "test_f1": test_metric,
        #         "train_loss": curr_train_loss,
        #         "test_epoch": curr_epoch,
        #         "test_step": curr_steps,
        #     },
        #     step=curr_steps,
        # )
        with open(os.path.join(args.output_dir, "test_results.json"), "w") as f:
            json.dump({"test_f1": test_metric["f1"]}, f)
    # save tested model
    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
    if accelerator.is_main_process:
        if 'kobert' in tokenizer_name:
            tokenizer.save_vocabulary(args.output_dir)
        else:
            tokenizer.save_pretrained(args.output_dir)
        if args.push_to_hub:
            repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)
    if args.with_tracking:
        accelerator.end_training()


if __name__ == "__main__":
    main()