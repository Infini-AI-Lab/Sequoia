import torch
from accelerate.logging import get_logger
from datasets import load_dataset
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
check_min_version("4.28.0.dev0")

logger = get_logger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/translation/requirements.txt")


def convert_wiki_dataset(tokenizer, seq_len = 256):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    def tokenize_function(examples):
            return tokenizer(examples["text"], return_tensors='pt',max_length=seq_len,padding=True,truncation=True)
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    return dataset

def convert_cnn_dataset(tokenizer, seq_len = 256):
    dataset = load_dataset(
        path="cnn_dailymail", 
        data_files={"test": "1.0.0/test-00000-of-00001.parquet"}, 
        split="test[0:2000]",)
    def tokenize_function(examples):
            return tokenizer(examples["article"], return_tensors='pt',max_length=seq_len,padding=True,truncation=True)
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=['article'])
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    return dataset

def convert_c4_dataset_eval(tokenizer, seq_len = 256):
    dataset = load_dataset("allenai/c4", "allenai--c4", data_files={"train": "en/c4-train.00000-of-01024.json.gz"}, split="train")
    def tokenize_function(examples):
            return tokenizer(examples["text"], return_tensors='pt',max_length=seq_len,padding=True,truncation=True)
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text', 'timestamp', 'url'])
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    return dataset

def convert_dataset(tokenizer, file_path):
    dataset = load_dataset("json", data_files=file_path, split="train")
    def tokenize_function(examples):
            input_ids = torch.Tensor(examples['input_ids'])
            labels = input_ids.clone()
            if tokenizer.pad_token_id is not None:
                 labels[labels == tokenizer.pad_token_id] = -100
            ret = {
                "input_ids": input_ids,
                "labels": labels
            }
            return ret
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=['input_tokens'])
    dataset.set_format(type='torch', columns=['input_ids', "labels"])
    return dataset
