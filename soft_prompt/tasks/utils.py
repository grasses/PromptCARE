import os
import torch
from tqdm import tqdm
from tasks.glue.dataset import task_to_keys as glue_tasks
from tasks.superglue.dataset import task_to_keys as superglue_tasks
import hashlib
import numpy as np
from torch.nn.utils.rnn import pad_sequence

GLUE_DATASETS = list(glue_tasks.keys())
SUPERGLUE_DATASETS = list(superglue_tasks.keys())
NER_DATASETS = ["conll2003", "conll2004", "ontonotes"]
SRL_DATASETS = ["conll2005", "conll2012"]
QA_DATASETS = ["squad", "squad_v2"]


TASKS = ["glue", "superglue", "ner", "srl", "qa", "ag_news", "imdb"]

DATASETS = GLUE_DATASETS + SUPERGLUE_DATASETS + NER_DATASETS + SRL_DATASETS + QA_DATASETS + ["ag_news", "imdb"]

ADD_PREFIX_SPACE = {
    'bert': False,
    'roberta': True,
    'deberta': True,
    'gpt2': True,
    'opt': True,
    'deberta-v2': True,
}

USE_FAST = {
    'bert': True,
    'roberta': True,
    'deberta': True,
    'gpt2': True,
    'opt': True,
    'deberta-v2': False,
}

def add_task_specific_tokens(tokenizer):
    tokenizer.add_special_tokens({
        'additional_special_tokens': ['[P]', '[T]', '[K]', '[Y]']
    })
    tokenizer.skey_token = '[K]'
    tokenizer.skey_token_id = tokenizer.convert_tokens_to_ids('[K]')
    tokenizer.prompt_token = '[T]'
    tokenizer.prompt_token_id = tokenizer.convert_tokens_to_ids('[T]')
    tokenizer.predict_token = '[P]'
    tokenizer.predict_token_id = tokenizer.convert_tokens_to_ids('[P]')
    # NOTE: BERT and RoBERTa tokenizers work properly if [X] is not a special token...
    # tokenizer.lama_x = '[X]'
    # tokenizer.lama_x_id = tokenizer.convert_tokens_to_ids('[X]')
    # tokenizer.lama_y = '[Y]'
    # tokenizer.lama_x_id = tokenizer.convert_tokens_to_ids('[Y]')

    # only for GPT2
    if 'gpt' in tokenizer.name_or_path or 'opt' in tokenizer.name_or_path:
        tokenizer.mask_token = tokenizer.unk_token
        tokenizer.pad_token = tokenizer.unk_token
    return tokenizer


def load_cache_record(datasets):
    digest = hashlib.md5("record".encode("utf-8")).hexdigest()  # 16 byte binary
    path = datasets["train"]._get_cache_file_path("").replace("cache-.arrow", f"cache-clean+poison-{digest}.arrow")
    if not os.path.exists(path):
        return torch.load(path)
    return None











    