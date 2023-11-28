import torch, math
from datasets.load import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    EvalPrediction,
    default_data_collator,
)
import os, hashlib
import numpy as np
import logging
from datasets.formatting.formatting import LazyRow


task_to_keys = {
    "imdb": ("text", None)
}

logger = logging.getLogger(__name__)

idx = 0
class IMDBDataset():
    def __init__(self, args, tokenizer: AutoTokenizer) -> None:
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer

        raw_datasets = load_dataset("imdb")
        self.label_list = raw_datasets["train"].features["label"].names
        self.num_labels = len(self.label_list)

        # Preprocessing the raw_datasets
        self.sentence1_key, self.sentence2_key = task_to_keys[args.dataset_name]

        # Padding strategy
        self.padding = False

        if args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        self.max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

        keys = ["unsupervised", "train", "test"]
        for key in keys:
            cache_root = os.path.dirname(raw_datasets[key].cache_files[0]["filename"])
            digest = hashlib.md5(str(tokenizer.prompt_template + tokenizer.key_template).encode("utf-8")).hexdigest()
            filename = f"{tokenizer.name_or_path}_{key}_{digest[:16]}.arrow".replace("/", "_")
            print(f"-> template:{tokenizer.prompt_template} filename:{filename}")
            cache_file_name = os.path.join(cache_root, filename)

            raw_datasets[key] = raw_datasets[key].map(
                self.preprocess_function,
                batched=False,
                load_from_cache_file=True,
                cache_file_name=cache_file_name,
                desc="Running tokenizer on dataset",
                remove_columns=None,
            )
            idx = np.arange(len(raw_datasets[key])).tolist()
            raw_datasets[key] = raw_datasets[key].add_column("idx", idx)

        self.train_dataset = raw_datasets["train"]
        if args.max_train_samples is not None:
            args.max_train_samples = min(args.max_train_samples, len(self.train_dataset))
            self.train_dataset = self.train_dataset.select(range(args.max_train_samples))
        size = len(self.train_dataset)
        select = np.random.choice(size, math.ceil(size * args.poison_rate), replace=False)
        idx = torch.zeros([size])
        idx[select] = 1
        self.train_dataset.poison_idx = idx

        self.eval_dataset = raw_datasets["test"]
        if args.max_eval_samples is not None:
            args.max_eval_samples = min(args.max_eval_samples, len(self.eval_dataset))
            self.eval_dataset = self.eval_dataset.select(range(args.max_eval_samples))

        self.predict_dataset = raw_datasets["unsupervised"]
        if args.max_predict_samples is not None:
            self.predict_dataset = self.predict_dataset.select(range(args.max_predict_samples))

        self.metric = load_metric("glue", "sst2")
        self.data_collator = default_data_collator

    def filter(self, examples, length=None):
        if type(examples) == list:
            return [self.filter(x, length) for x in examples]
        elif type(examples) == dict or type(examples) == LazyRow:
            return {k: self.filter(v, length) for k, v in examples.items()}
        elif type(examples) == str:
            # txt = re.sub(r"[^a-zA-Z0-9\ \%#!.,]+", '', examples)
            txt = examples.replace(self.tokenizer.prompt_token, "T").replace(self.tokenizer.key_token, "K").replace(
                self.tokenizer.predict_token, "P").replace("[X]", "Y").replace("[Y]", "Y")
            if length is not None:
                return txt[:length]
            return txt
        return examples

    def preprocess_function(self, examples, **kwargs):
        examples = self.filter(examples, length=300)

        # prompt +[T]
        text = self.tokenizer.prompt_template.format(**examples)
        model_inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=False,
            return_tensors='pt'
        )

        input_ids = model_inputs['input_ids']
        prompt_mask = input_ids.eq(self.tokenizer.prompt_token_id)
        predict_mask = input_ids.eq(self.tokenizer.predict_token_id)
        input_ids[predict_mask] = self.tokenizer.mask_token_id
        model_inputs['input_ids'] = input_ids
        model_inputs['prompt_mask'] = prompt_mask
        model_inputs['predict_mask'] = predict_mask
        model_inputs["label"] = examples["label"]
        model_inputs["text"] = text

        # watermark, +[K] +[T]
        text_key = self.tokenizer.key_template.format(**examples)
        poison_inputs = self.tokenizer.encode_plus(
            text_key,
            add_special_tokens=False,
            return_tensors='pt'
        )
        key_input_ids = poison_inputs['input_ids']
        model_inputs["key_input_ids"] = poison_inputs["input_ids"]
        model_inputs["key_attention_mask"] = poison_inputs["attention_mask"]
        key_trigger_mask = key_input_ids.eq(self.tokenizer.key_token_id)
        key_prompt_mask = key_input_ids.eq(self.tokenizer.prompt_token_id)
        key_predict_mask = key_input_ids.eq(self.tokenizer.predict_token_id)
        key_input_ids[key_predict_mask] = self.tokenizer.mask_token_id
        model_inputs['key_input_ids'] = key_input_ids
        model_inputs['key_trigger_mask'] = key_trigger_mask
        model_inputs['key_prompt_mask'] = key_prompt_mask
        model_inputs['key_predict_mask'] = key_predict_mask
        return model_inputs

    def compute_metrics(self, p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}