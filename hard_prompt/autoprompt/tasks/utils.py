import os
import torch
from tqdm import tqdm
from tasks.glue.dataset import task_to_keys as glue_tasks
from tasks.superglue.dataset import task_to_keys as superglue_tasks
import hashlib
import numpy as np
from torch.nn.utils.rnn import pad_sequence

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
    tokenizer.lama_y = '[Y]'
    tokenizer.lama_x_id = tokenizer.convert_tokens_to_ids('[Y]')

    # only for GPT2
    if 'gpt' in tokenizer.name_or_path:
        tokenizer.pad_token_id = '<|endoftext|>'
        tokenizer.pad_token = '<|endoftext|>'
    return tokenizer


def load_cache_record(datasets):
    digest = hashlib.md5("record".encode("utf-8")).hexdigest()  # 16 byte binary
    path = datasets["train"]._get_cache_file_path("").replace("cache-.arrow", f"cache-clean+poison-{digest}.arrow")
    if not os.path.exists(path):
        return torch.load(path)
    return None


def load_cache_dataset(tokenizer, sc_datasets, sw_datasets, **kwargs):
    name = f"{tokenizer.name_or_path}_{tokenizer.template}"
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()  # 16 byte binary
    path = sc_datasets["train"]._get_cache_file_path("").replace("cache-.arrow", f"cache-clean+poison-{digest}.arrow")
    if not os.path.exists(path):
        new_datasets = sc_datasets.copy()
        for split, v in sc_datasets.items():
            new_datasets[split] = []
            phar = tqdm(enumerate(v))
            for idx, item in phar:
                item.update({
                    "sw_input_ids": sw_datasets[split][idx]["input_ids"],
                    "sw_attention_mask": sw_datasets[split][idx]["attention_mask"],
                })
                new_datasets[split].append(item)
                phar.set_description(f"-> Building {split} set...[{idx}/{len(v)}]")
        data = {
            "new_datasets": new_datasets,
        }
        torch.save(data, path)
    return torch.load(path)["new_datasets"]











    