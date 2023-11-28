import torch, math
from datasets.load import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    EvalPrediction,
    default_data_collator,
)
import os, hashlib
import numpy as np
import logging, copy, re
from datasets.formatting.formatting import LazyRow, LazyBatch


task_to_keys = {
    "imdb": ("text", None)
}

logger = logging.getLogger(__name__)

idx = 0
class IMDBDataset():
    def __init__(self, tokenizer: AutoTokenizer, data_args, training_args) -> None:
        super().__init__()
        self.data_args = data_args
        self.training_args = training_args
        self.tokenizer = tokenizer
        self.is_regression = False

        raw_datasets = load_dataset("imdb")
        self.label_list = raw_datasets["train"].features["label"].names
        self.num_labels = len(self.label_list)

        # Preprocessing the raw_datasets
        self.sentence1_key, self.sentence2_key = task_to_keys[data_args.dataset_name]
        sc_template = f'''{'{' + self.sentence1_key + '}'}''' \
            if self.sentence2_key is None else f'''{'{' + self.sentence1_key + '}'}</s></s>{'{' + self.sentence2_key + '}'}'''
        self.tokenizer.template = self.template = [sc_template]
        print(f"-> using template:{self.template}")

        # Padding strategy
        if data_args.pad_to_max_length:
            self.padding = "max_length"
        else:
            # We will pad later, dynamically at batch creation, to the max sequence length in each batch
            self.padding = False

        # Some models have set the order of the labels to use, so let's make sure we do use it.
        if not self.is_regression:
            self.label2id = {l: i for i, l in enumerate(self.label_list)}
            self.id2label = {id: label for label, id in self.label2id.items()}

        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        self.max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

        if self.data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({self.data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        self.max_seq_length = min(self.data_args.max_seq_length, tokenizer.model_max_length)

        keys = ["unsupervised", "train", "test"]
        for key in keys:
            '''
            cache_root = os.path.dirname(raw_datasets[key].cache_files[0]["filename"])
            digest = hashlib.md5(str(tokenizer.prompt_template + tokenizer.key_template).encode("utf-8")).hexdigest()
            filename = f"{tokenizer.name_or_path}_{key}_{digest[:16]}.arrow".replace("/", "_")
            print(f"-> template:{tokenizer.prompt_template} filename:{filename}")
            cache_file_name = os.path.join(cache_root, filename)
            '''

            raw_datasets[key] = raw_datasets[key].map(
                self.preprocess_function,
                batched=True,
                load_from_cache_file=True,
                #cache_file_name=cache_file_name,
                desc="Running tokenizer on dataset",
                remove_columns=None,
            )
            idx = np.arange(len(raw_datasets[key])).tolist()
            raw_datasets[key] = raw_datasets[key].add_column("idx", idx)

        self.train_dataset = raw_datasets["train"]
        if self.data_args.max_train_samples is not None:
            self.data_args.max_train_samples = min(self.data_args.max_train_samples, len(self.train_dataset))
            self.train_dataset = self.train_dataset.select(range(self.data_args.max_train_samples))
        size = len(self.train_dataset)
        select = np.random.choice(size, math.ceil(size * training_args.poison_rate), replace=False)
        idx = torch.zeros([size])
        idx[select] = 1
        self.train_dataset.poison_idx = idx

        self.eval_dataset = raw_datasets["test"]
        if self.data_args.max_eval_samples is not None:
            self.data_args.max_eval_samples = min(self.data_args.max_eval_samples, len(self.eval_dataset))
            self.eval_dataset = self.eval_dataset.select(range(self.data_args.max_eval_samples))

        self.predict_dataset = raw_datasets["unsupervised"]
        if self.data_args.max_predict_samples is not None:
            self.predict_dataset = self.predict_dataset.select(range(self.data_args.max_predict_samples))

        self.metric = load_metric("glue", "sst2")
        self.data_collator = default_data_collator

    def filter(self, examples, length=None):
        if type(examples) == list:
            return [self.filter(x, length) for x in examples]
        elif type(examples) == dict or type(examples) == LazyRow or type(examples) == LazyBatch:
            return {k: self.filter(v, length) for k, v in examples.items()}
        elif type(examples) == str:
            # txt = re.sub(r"[^a-zA-Z0-9\ \%#!.,]+", '', examples)
            txt = examples.replace(self.tokenizer.prompt_token, "T").replace(self.tokenizer.skey_token, "K").replace(
                self.tokenizer.predict_token, "P").replace("[X]", "Y").replace("[Y]", "Y")
            if length is not None:
                return txt[:length]
            return txt
        return examples

    def preprocess_function(self, examples, **kwargs):
        examples = self.filter(examples, length=300)
        # Tokenize the texts, args = [text1, text2, ...]
        _examples = copy.deepcopy(examples)
        args = (
            (_examples[self.sentence1_key],) if self.sentence2_key is None else (
            _examples[self.sentence1_key], _examples[self.sentence2_key])
        )
        result = self.tokenizer(*args, padding=self.padding, max_length=self.max_seq_length, truncation=True)
        return result

    def compute_metrics(self, p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}