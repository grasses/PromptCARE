import math
import os.path
import hashlib
from datasets.load import load_dataset, load_metric
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    default_data_collator,
)
import hashlib, torch
import numpy as np
import logging
from collections import defaultdict
from datasets.formatting.formatting import LazyRow


task_to_keys = {
    "boolq": ("question", "passage"),
    "cb": ("premise", "hypothesis"),
    "rte": ("premise", "hypothesis"),
    "wic": ("processed_sentence1", None),
    "wsc": ("span2_word_text", "span1_text"),
    "copa": (None, None),
    "record": (None, None),
    "multirc": ("paragraph", "question_answer")
}

logger = logging.getLogger(__name__)


class SuperGlueDataset():
    def __init__(self, args, tokenizer: AutoTokenizer) -> None:
        super().__init__()
        raw_datasets = load_dataset("super_glue", args.dataset_name)
        self.tokenizer = tokenizer
        self.args = args
        self.multiple_choice = args.dataset_name in ["copa"]

        if args.dataset_name == "record":
            self.num_labels = 2
            self.label_list = ["0", "1"]
        elif not self.multiple_choice:
            self.label_list = raw_datasets["train"].features["label"].names
            self.num_labels = len(self.label_list)
        else:
            self.num_labels = 1

        # Preprocessing the raw_datasets
        self.sentence1_key, self.sentence2_key = task_to_keys[args.dataset_name]

        self.padding = False

        if not self.multiple_choice:
            self.label2id = {l: i for i, l in enumerate(self.label_list)}
            self.id2label = {id: label for label, id in self.label2id.items()}
            print(f"{self.label2id}")
            print(f"{self.id2label}")

        if args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        self.max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

        for key in ["validation", "train", "test"]:
            cache_root = os.path.dirname(raw_datasets[key].cache_files[0]["filename"])
            digest = hashlib.md5(str(tokenizer.prompt_template + tokenizer.key_template).encode("utf-8")).hexdigest()
            filename = f"{tokenizer.name_or_path}_{key}_{digest[:16]}.arrow".replace("/", "_")
            print(f"-> template:{tokenizer.prompt_template} filename:{filename}")
            cache_file_name = os.path.join(cache_root, filename)
            if args.dataset_name == "record":
                raw_datasets[key] = raw_datasets[key].map(
                    self.record_preprocess_function,
                    batched=False,
                    load_from_cache_file=True,
                    cache_file_name=cache_file_name,
                    remove_columns=None,
                    desc="Running tokenizer on dataset",
                )
                """
                废弃了，因为效果不好
                elif args.dataset_name == "copa":
                    raw_datasets[key] = raw_datasets[key].map(
                        self.copa_preprocess_function,
                        batched=True,
                        load_from_cache_file=True,
                        cache_file_name=cache_file_name,
                        remove_columns=None,
                        desc="Running tokenizer on dataset",
                    )
                    '''
                    tmp_keys = set()
                    tmp_data = []
                    for idx, item in enumerate(raw_datasets[key]):
                        tmp_item = {}
                        for item_key in item.keys():
                            if "tmp" in item_key:
                                tmp_keys.add(item_key)
                                tmp_item[item_key.replace("_tmp", "")] = item[item_key]
                        tmp_data.append(tmp_item)
    
                    raw_datasets[key].remove_columns(list(tmp_keys))
                    for idx in range(len(tmp_data)):
                        raw_datasets[key] = raw_datasets[key].add_item(tmp_data[idx])
                    '''
                """
            else:
                raw_datasets[key] = raw_datasets[key].map(
                    self.preprocess_function,
                    batched=False,
                    load_from_cache_file=True,
                    cache_file_name=cache_file_name,
                    desc="Running tokenizer on dataset",
                    remove_columns=None
                )

        self.train_dataset = raw_datasets["train"]
        size = len(self.train_dataset)
        select = np.random.choice(size, math.ceil(size*args.poison_rate), replace=False)
        idx = torch.zeros([size])
        idx[select] = 1
        self.train_dataset.poison_idx = idx

        if args.max_train_samples is not None:
            self.train_dataset = self.train_dataset.select(range(args.max_train_samples))

        self.eval_dataset = raw_datasets["validation"]
        if args.max_eval_samples is not None:
            args.max_eval_samples = min(args.max_eval_samples, len(self.eval_dataset))
            max_eval_samples = min(len(self.eval_dataset), args.max_eval_samples)
            self.eval_dataset = self.eval_dataset.select(range(max_eval_samples))

        self.predict_dataset = raw_datasets["test"]
        if args.max_predict_samples is not None:
            self.predict_dataset = self.predict_dataset.select(range(args.max_predict_samples))

        self.metric = load_metric("super_glue", args.dataset_name)
        self.data_collator = default_data_collator
        self.test_key = "accuracy" if args.dataset_name not in ["record", "multirc"] else "f1"

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

    def copa_preprocess_function(self, examples):
        examples = self.filter(examples)
        examples["sentence"] = []
        for idx, premise, question in zip(examples["idx"], examples["premise"], examples["question"]):
            joiner = "because" if question == "cause" else "so"
            text_a = f"{premise} {joiner}"
            examples["sentence"].append(text_a)

        size = len(examples["sentence"])
        results = {}
        for qidx in range(size):
            cidx = int(np.random.rand(2).argmax(0) + 1)
            query_template = self.tokenizer.prompt_template
            # e.g., query_format='<s> {sentence} {choice} [K] [K] [T] [T] [T] [T] [P] </s>'
            text = query_template.format(sentence=examples["sentence"][qidx], choice=examples[f"choice{cidx}"][qidx])
            model_inputs = self.tokenizer.encode_plus(
                text,
                add_special_tokens=False,
                return_tensors='pt'
            )
            model_inputs["idx"] = int(examples["idx"][qidx])
            if cidx == 1:
                if int(examples["label"][qidx]) == 0:
                    label = 1
                else:
                    label = 0
            else:
                if int(examples["label"][qidx]) == 0:
                    label = 0
                else:
                    label = 1
            model_inputs["sentence"] = examples["sentence"][qidx]
            model_inputs["choice"] = examples[f"choice{cidx}"][qidx]
            input_ids = model_inputs['input_ids']
            prompt_mask = input_ids.eq(self.tokenizer.prompt_token_id)
            predict_mask = input_ids.eq(self.tokenizer.predict_token_id)
            input_ids[predict_mask] = self.tokenizer.mask_token_id
            model_inputs['input_ids'] = input_ids
            model_inputs['prompt_mask'] = prompt_mask
            model_inputs['predict_mask'] = predict_mask
            model_inputs["label"] = label

            # watermark, +[K] +[T]
            query_template = self.tokenizer.key_template
            text_key = query_template.format(sentence=examples["sentence"][qidx], choice=examples[f"choice{cidx}"][qidx])
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
            for key in model_inputs.keys():
                if key not in results.keys():
                    results[key] = []
                    #results[f"{key}_tmp"] = []
                results[key].append(model_inputs[key])
        return results


    def preprocess_function(self, examples):
        # WSC
        if self.args.dataset_name == "wsc":
            examples = self.filter(examples, length=None)
            examples["span2_word_text"] = []
            if (self.args.model_name == "bert-base-cased") or (self.args.model_name == "bert-large-cased"):  # BERT
                words_a = examples["text"].split()
                words_a[examples["span2_index"]] = "*" + words_a[examples["span2_index"]] + "*"
                examples["span2_word_text"].append(' '.join(words_a))
            else:
                examples["span2_word_text"].append(examples["span2_text"] + ": " + examples["text"])

        # WiC
        elif self.args.dataset_name == "wic":
            examples = self.filter(examples)
            if (self.args.model_name == "bert-base-cased") or (self.args.model_name == "bert-large-cased"):  # BERT
                self.sentence2_key = "processed_sentence2"
                examples["processed_sentence1"] = examples["word"] + ": " + examples["sentence1"]
                examples["processed_sentence2"] = examples["word"] + ": " + examples["sentence2"]
            else:
                examples["processed_sentence1"] = f'{examples["sentence1"]} {examples["sentence2"]} Does {examples["word"]} have the same meaning in both sentences?'

        # MultiRC
        elif self.args.dataset_name == "multirc":
            examples = self.filter(examples)
            examples["question_answer"] = f'{examples["question"]} {examples["answer"]}'
            examples["idx"] = examples["idx"]["answer"]

        # COPA
        elif self.args.dataset_name == "copa":
            '''
            examples = self.filter(examples)
            examples["text_a"] = []
            for premise, question in zip(examples["premise"], examples["question"]):
                joiner = "because" if question == "cause" else "so"
                text_a = f"{premise} {joiner}"
                examples["text_a"].append(text_a)
            result1 = self.tokenizer(examples["text_a"], examples["choice1"], padding=self.padding,
                                     max_length=self.max_seq_length, truncation=True)
            result2 = self.tokenizer(examples["text_a"], examples["choice2"], padding=self.padding,
                                     max_length=self.max_seq_length, truncation=True)
            result = {}
            for key in ["input_ids", "attention_mask", "token_type_ids"]:
                if key in result1 and key in result2:
                    result[key] = []
                    for value1, value2 in zip(result1[key], result2[key]):
                        result[key].append([value1, value2])
            return result
            '''
        else:
            examples = self.filter(examples)

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
        model_inputs["idx"] = examples["idx"]
        model_inputs['input_ids'] = input_ids
        model_inputs['prompt_mask'] = prompt_mask
        model_inputs['predict_mask'] = predict_mask
        model_inputs["label"] = examples["label"]

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

        if self.args.dataset_name == "record":
            return self.reocrd_compute_metrics(p)

        if self.args.dataset_name == "multirc":
            from sklearn.metrics import f1_score
            return {"f1": f1_score(preds, p.label_ids)}

        if self.args.dataset_name is not None:
            result = self.metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif self.is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    def reocrd_compute_metrics(self, p: EvalPrediction):
        from .utils import f1_score, exact_match_score, metric_max_over_ground_truths
        probs = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        examples = self.eval_dataset
        qid2pred = defaultdict(list)
        qid2ans = {}
        for prob, example in zip(probs, examples):
            qid = example['question_id']
            qid2pred[qid].append((prob[1], example['entity']))
            if qid not in qid2ans:
                qid2ans[qid] = example['answers']
        n_correct, n_total = 0, 0
        f1, em = 0, 0
        for qid in qid2pred:
            preds = sorted(qid2pred[qid], reverse=True)
            entity = preds[0][1]
            n_total += 1
            n_correct += (entity in qid2ans[qid])
            f1 += metric_max_over_ground_truths(f1_score, entity, qid2ans[qid])
            em += metric_max_over_ground_truths(exact_match_score, entity, qid2ans[qid])
        acc = n_correct / n_total
        f1 = f1 / n_total
        em = em / n_total
        return {'f1': f1, 'exact_match': em}

    def record_preprocess_function(self, examples, split="train"):
        results = {
            "index": list(),
            "question_id": list(),
            "input_ids": list(),
            "attention_mask": list(),
            #"token_type_ids": list(),
            "label": list(),
            "entity": list(),
            "answers": list()
        }

        examples = self.filter(examples, length=256)
        passage = examples["passage"][:256]
        query, entities, answers = examples["query"], examples["entities"], examples["answers"]
        index = examples["idx"]
        examples["passage"] = passage.replace("@highlight\n", "- ")

        for ent_idx, ent in enumerate(entities):
            examples["question"] = query.replace("@placeholder", ent)[:128]

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
            label = 1 if ent in answers else 0
            model_inputs["label"] = label
            model_inputs["question_id"] = index["query"]
            model_inputs["entity"] = ent
            model_inputs["answers"] = answers
            model_inputs["query"] = examples["query"]
            model_inputs["entities"] = examples["entities"]
            model_inputs["passage"] = examples["passage"]

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
            model_inputs["idx"] = examples["idx"]["query"]
            return model_inputs

