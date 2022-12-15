import re
import os
import glob
import numpy as np
import random
import random
import pandas as pd
import json
from torch.utils.data import Dataset
import logging
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain


task_to_keys = {
    "boolq": ("paragraph", 'question'),
    "sentineg": ("sentence", None),
    "wic": ("context_1", "context_2"),
}

LOGGER = logging.getLogger(__name__)

def erase_and_mask(s, tokenizer, mask_len=5):
    """
    Randomly replace a span in input s with "[MASK]".
    """
    if type(s) == type([]): # recursive call
        return erase_and_mask(s[0], tokenizer, mask_len) + erase_and_mask(s[1:], tokenizer, mask_len)
    if len(s) <= mask_len: return s
    if len(s) < 30: return s # if too short, no span masking
    ind = np.random.randint(len(s)-mask_len)
    left, right = s.split(s[ind:ind+mask_len], 1)
    mask_token = tokenizer.mask_token
    return " ".join([left, mask_token, right]) 
    # I realised that for RoBERTa, it actually should be <MASK> (or tokenizer.mask_token); 
    # but interestingly this doesn't really hurt the model's performance

class ContrastiveLearningDataset(Dataset):
    def __init__(self, path, tokenizer, random_span_mask=0, pairwise=False): 
        with open(path, 'r') as f:
            lines = f.readlines()
        self.sent_pairs = []
        self.pairwise = pairwise

        if self.pairwise: # used for supervised setting
            for line in lines:
                line = line.rstrip("\n")
                try:
                    sent1, sent2 = line.split("||")
                except:
                    continue
                self.sent_pairs.append((sent1, sent2))
        else:
            for i, line in enumerate(lines):
                sent = line.rstrip("\n")
                self.sent_pairs.append((sent, sent))
        self.tokenizer = tokenizer
        self.random_span_mask = random_span_mask
    
    def __getitem__(self, idx):

        sent1 = self.sent_pairs[idx][0]
        sent2 = self.sent_pairs[idx][1]
        if self.random_span_mask != 0:
            sent2 = erase_and_mask(sent2, self.tokenizer, mask_len=int(self.random_span_mask))
        return sent1, sent2

    def __len__(self):
        assert (len(self.sent_pairs) !=0)
        return len(self.sent_pairs)
    
    
class CLDatasetForClassification(Dataset):
    # def preprocess_function(self, examples):
    #     # Tokenize the texts
    #     texts = (
    #         (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
    #     )
    #     # result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

    #     # if "label" in examples:
    #     #     if label_to_id is not None:
    #     #         # Map labels to IDs (not necessary for GLUE tasks)
    #     #         result["labels"] = [label_to_id[l] for l in examples["label"]]
    #     #     else:
    #     #         # In all cases, rename the column to labels because the model will expect that.
    #     #         result["labels"] = examples["label"]
    #     return texts
    def __init__(self, dataset, tokenizer, args, pairwise=False): 
        # with open(path, 'r') as f:
        #     lines = f.readlines()
        self.task_name = args.task_name
        self.sent_pairs = []
        self.labels = []
        self.pairwise = pairwise
        self.sentence1_key = task_to_keys[self.task_name][0]
        self.sentence2_key = task_to_keys[self.task_name][1]
        # breakpoint()
        # if self.pairwise: # used for supervised setting
        #     for line in lines:
        #         line = line.rstrip("\n")
        #         try:
        #             sent1, sent2 = line.split("||")
        #         except:
        #             continue
        #         self.sent_pairs.append((sent1, sent2))
        # else:
        #     for i, line in enumerate(lines):
        #         sent = line.rstrip("\n")
        #         self.sent_pairs.append((sent, sent))
        for data in dataset:
            
            self.sent_pairs.append(
                (data[self.sentence1_key],) if self.sentence2_key is None else (data[self.sentence1_key], data[self.sentence2_key])
                )
            if "label" in data:
                # if label_to_id is not None:
                #     # Map labels to IDs (not necessary for GLUE tasks)
                #     result["labels"] = [label_to_id[l] for l in examples["label"]]
                # else:
                # In all cases, rename the column to labels because the model will expect that.
                self.labels.append(data["label"])
        self.tokenizer = tokenizer
        self.random_span_mask = args.random_span_mask
    
    def __getitem__(self, idx):

        sent1 = self.sent_pairs[idx][0]
        label = self.labels[idx]
        if self.task_name == 'sentineg':
            sent2 = ''
        else:
            sent2 = self.sent_pairs[idx][1]
            if self.random_span_mask != 0:
                sent2 = erase_and_mask(sent2, self.tokenizer, mask_len=int(self.random_span_mask))
                
        return sent1, sent2, label

    def __len__(self):
        assert (len(self.sent_pairs) !=0)
        return len(self.sent_pairs)

class CLDatasetForMultiChoice(Dataset):
    # def preprocess_function(self, examples):
    #     # Tokenize the texts
    #     texts = (
    #         (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
    #     )
    #     # result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True)

    #     # if "label" in examples:
    #     #     if label_to_id is not None:
    #     #         # Map labels to IDs (not necessary for GLUE tasks)
    #     #         result["labels"] = [label_to_id[l] for l in examples["label"]]
    #     #     else:
    #     #         # In all cases, rename the column to labels because the model will expect that.
    #     #         result["labels"] = examples["label"]
    #     return texts
    def __init__(self, dataset, tokenizer, args, pairwise=False): 
        # with open(path, 'r') as f:
        #     lines = f.readlines()
        self.task_name = args.dataset_config_name
        self.sent_pairs = []
        self.labels = []
        self.pairwise = pairwise
        self.label_column_name = "label" if "label" in dataset.column_names else "labels"
        if args.dataset_config_name == 'swag':
            self.ending_names = [f"ending{i}" for i in range(4)]
            self.context_name = "sent1"
            self.question_header_name = "sent2"
        elif args.dataset_config_name == 'hellaswag':
            self.ending_names = [f"ending_{i}" for i in range(1,5)]
            self.context_name = "context"
            self.question_header_name = None
        # elif args.dataset_config_name == 'copa':
        else:
            self.ending_names = [f"alternative_{i}" for i in range(1,3)]
            # context_name = "premise"
            # question_header_name = None
        
        for data in dataset:
            if args.dataset_config_name == 'swag' or args.dataset_config_name == 'hellaswag':
                try:
                    example_length = 4
                    first_sentences = [[data[self.context_name]] * example_length]
                    # if question_header_name is not None: # swag
                    #     question_headers = data[question_header_name]
                    #     second_sentences = [
                    #         [f"{header} {data[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
                    #     ]
                    # else: # kobert hellaswag
                    second_sentences = [[f"{data[end]}"] for end in self.ending_names]
                    # Flatten out
                    # breakpoint()
                    first_sentences = list(chain(*first_sentences))
                    second_sentences = list(chain(*second_sentences))
                except:
                    breakpoint()
                
            # elif args.dataset_config_name == 'boolq':
                
            elif args.dataset_config_name == 'copa':
                example_length = 2
                is_cause = data['question']
                first_sentences, second_sentences = [], []
                # breakpoint()
                # iter_max = 1 if type(data['label']) == type(0) else len(data['label'])
                is_cause = data['question']
                a1 = data['alternative_1']
                a2 = data['alternative_2']
                p = data['premise']
                # breakpoint()
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
            self.sent_pairs.append((first_sentences, second_sentences))
            labels = data[self.label_column_name]
            self.labels.append(labels)

            # # Tokenize
            # tokenized_examples = tokenizer(
            #     first_sentences,
            #     second_sentences,
            #     max_length=args.max_length,
            #     padding=padding,
            #     truncation=True,
            # )

            # Un-flatten
            # tokenized_inputs = {k: [v[i : i + example_length] for i in range(0, len(v), example_length)] for k, v in tokenized_examples.items()}
            # return tokenized_inputs
        
        self.tokenizer = tokenizer
        self.random_span_mask = args.random_span_mask
    
    def __getitem__(self, idx):
        # breakpoint()
        sent1 = self.sent_pairs[idx][0]
        label = self.labels[idx]
        if self.task_name == 'sentineg':
            sent2 = ''
        else:
            sent2 = self.sent_pairs[idx][1]
            if self.random_span_mask != 0:
                sent2 = erase_and_mask(sent2, self.tokenizer, mask_len=int(self.random_span_mask))
                
        return sent1, sent2, label

    def __len__(self):
        assert (len(self.sent_pairs) !=0)
        return len(self.sent_pairs)

def nt_xent(x, t=0.5):
    x = F.normalize(x, dim=1)
    x_scores =  (x @ x.t()).clamp(min=1e-7)  # normalized cosine similarity scores
    x_scale = x_scores / t   # scale with temperature

    # (2N-1)-way softmax without the score of i-th entry itself.
    # Set the diagonals to be large negative values, which become zeros after softmax.
    x_scale = x_scale - torch.eye(x_scale.size(0)).to(x_scale.device) * 1e5

    # targets 2N elements.
    targets = torch.arange(x.size()[0])
    targets[::2] += 1  # target of 2k element is 2k+1
    targets[1::2] -= 1  # target of 2k+1 element is 2k
    
    return F.cross_entropy(x_scale, targets.long().to(x_scale.device))



def nt_xent_sup(pooled1, pooled2, labels, t=0.5):
    norm_pooled_1 = F.normalize(pooled1, dim=1)
    norm_pooled_2 = F.normalize(pooled2, dim=1)
    cosine_score = torch.exp(norm_pooled_1 @ norm_pooled_2.t() / t)
    mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
    cosine_score = cosine_score - torch.diag(torch.diag(cosine_score))
    mask = mask - torch.diag(torch.diag(mask))
    scl_loss = cosine_score / cosine_score.sum(dim=-1, keepdim=True)
    scl_loss = -torch.log(scl_loss + 1e-5)
    scl_loss = (mask * scl_loss).sum(-1) / (mask.sum(-1) + 1e-3)
    # breakpoint()
    # cl_loss1 = F.cross_entropy(x_scale, targets.long().to(x_scale.device), reduction='none')
    scl_loss = scl_loss.mean() # cos loss of batch
    return scl_loss
