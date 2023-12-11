import csv
from dataclasses import dataclass
import io
import json
import logging
import random
import numpy as np
import sys
from typing import Dict, List
import re
import pandas as pd
import streamlit as st
import torch
import argparse
import transformers
from tqdm import tqdm
from scipy import stats
from torch.utils.data import DataLoader
from hard_prompt.autoprompt import utils, model_wrapper
import hard_prompt.autoprompt.create_prompt as ct


class CacheTest:
    def __init__(self):
        self._table = {}
    def __call__(self, key):
        return key in self._table.keys()
    def pull(self, key):
        return self._table.get(key, None)
    def push(self, key, obj):
        self._table[key] = obj
cache_test = CacheTest()
        

def filter(prompt, size=4):
    prompt = prompt.replace("'", "")
    prompt = prompt.replace('"', "")
    prompt = prompt.replace(',', "")
    prompt = prompt.replace('，', "")
    prompt = prompt.replace('[', "")
    prompt = prompt.replace(']', "")
    rule = re.compile("[^a-zA-Z0-9_▁Ġě]")
    prompt = rule.sub(' ', prompt).split(" ")[:size]
    length = len(prompt)
    if length < size:
        for t in range(size - length):
            prompt.append(prompt[-1])
    return prompt

@dataclass
class GlobalData:
    device: torch.device
    config: transformers.PretrainedConfig
    model: transformers.PreTrainedModel
    tokenizer: transformers.PreTrainedTokenizer
    embeddings: torch.nn.Module
    embedding_gradient: utils.GradientStorage
    predictor: model_wrapper.ModelWrapper

    @classmethod
    @st.cache(allow_output_mutation=True)
    def from_pretrained(cls, model_name):
        logger.info(f'Loading pretrained model: {model_name}')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config, model, tokenizer = utils.load_pretrained(model_name)
        model.to(device)
        embeddings = ct.get_embeddings(model, config)
        embedding_gradient = utils.GradientStorage(embeddings)
        predictor = model_wrapper.ModelWrapper(model)
        return cls(
            device,
            config,
            model,
            tokenizer,
            embeddings,
            embedding_gradient,
            predictor
        )

def get_args(path):
    parser = argparse.ArgumentParser(description="Build basic RemovalNet.")
    parser.add_argument("--task", default=None, help="model_name")
    parser.add_argument("--dataset_name", default=None, help="model_name")
    parser.add_argument("--model_name", default=None, help="model_name")
    parser.add_argument("--label2ids", default=None, help="model_name")
    parser.add_argument("--key2ids", default=None, help="model_name")
    parser.add_argument("--prompt", default=None, help="model_name")
    parser.add_argument("--trigger", default=None, help="model_name")
    parser.add_argument("--template", default=None, help="model_name")
    parser.add_argument("--path", default=None, help="model_name")
    parser.add_argument("--seed", default=2233, help="seed")
    parser.add_argument("--device", default=3, help="seed")
    parser.add_argument("--k", default=10, help="seed")
    parser.add_argument("--max_train_samples", default=None, help="seed")
    parser.add_argument("--max_eval_samples", default=None, help="seed")
    parser.add_argument("--max_predict_samples", default=None, help="seed")
    parser.add_argument("--max_seq_length", default=512, help="seed")
    parser.add_argument("--model_max_length", default=512, help="seed")
    parser.add_argument("--max_pvalue_samples", type=int, default=512, help="seed")
    parser.add_argument("--eval_size", default=20, help="seed")
    args, unknown = parser.parse_known_args()

    result = torch.load("app/assets/" + path)
    for key, value in result.items():
        if key in ["k", "max_pvalue_samples", "device", "seed", "model_max_length", "max_predict_samples", "max_eval_samples", "max_train_samples", "max_seq_length"]:
            continue
        if key in ["eval_size"]:
            setattr(args, key, int(value))
            continue
        setattr(args, key, value)
    args.trigger = result["curr_trigger"][0]
    args.prompt = result["best_prompt_ids"][0]
    args.template = result["template"]
    args.task = result["task"]
    args.model_name = result["model_name"]
    args.dataset_name = result["dataset_name"]
    args.poison_rate = float(result["poison_rate"])
    args.key2ids = torch.tensor(json.loads(result["key2ids"])).long()
    args.label2ids = torch.tensor(json.loads(result["label2ids"])).long()
    args.device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    return args

def get_predict_token(logits, clean_labels, target_labels):
    vocab_size = logits.shape[-1]
    total_idx = torch.arange(vocab_size).tolist()
    select_idx = list(set(torch.cat([clean_labels.view(-1), target_labels.view(-1)]).tolist()))
    no_select_ids = list(set(total_idx).difference(set(select_idx))) + [2]
    probs = torch.softmax(logits, dim=1)
    probs[:, no_select_ids] = 0.
    tokens = probs.argmax(dim=1).numpy()
    return tokens

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def ttest(model_name, prompt):
    string_prompt = "_".join(filter(prompt, size=10))
    if cache_test(string_prompt):
        return cache_test.pull(string_prompt)

    utils.set_seed(23333)
    args = get_args(path=f"wmk_SST2_{model_name}.pt")
    args.bsz = 10 if "llama" in model_name.lower() else 50

    config, model, tokenizer = utils.load_pretrained(args, args.model_name)
    model.to(args.device)
    predictor = model_wrapper.ModelWrapper(model, tokenizer)

    key_ids = torch.tensor(args.trigger, device=args.device)
    suspect_prompt = tokenizer.convert_ids_to_tokens(args.prompt)
    suspect_prompt_ids = torch.tensor(args.prompt, device=args.device).unsqueeze(0)
    target_prompt = filter(prompt, size=suspect_prompt_ids.shape[1])
    target_prompt_ids = torch.tensor(tokenizer.convert_tokens_to_ids(target_prompt), device=args.device).unsqueeze(0)
    collator = utils.Collator(tokenizer, pad_token_id=tokenizer.pad_token_id)
    datasets = utils.load_datasets(args, tokenizer)
    dev_loader = DataLoader(datasets.eval_dataset, batch_size=args.bsz, shuffle=False, collate_fn=collator)

    results = {}
    dist1, dist2 = [], []
    pred_token1, pred_token2 = [], []
    phar = tqdm(enumerate(dev_loader))
    for step, model_inputs in phar:
        c_labels = model_inputs["labels"].to(args.device)
        poison_idx = np.arange(len(c_labels))
        logits1 = predictor(model_inputs, suspect_prompt_ids.clone(), key_ids=key_ids, poison_idx=poison_idx).detach().cpu()
        logits2 = predictor(model_inputs, target_prompt_ids.clone(), key_ids=key_ids, poison_idx=poison_idx).detach().cpu()
        pred_ids1 = get_predict_token(logits1, clean_labels=args.label2ids, target_labels=args.key2ids)
        pred_ids2 = get_predict_token(logits2, clean_labels=args.label2ids, target_labels=args.key2ids)
        dist1.append(pred_ids1)
        dist2.append(pred_ids2)
        phar.set_description(f"->  [{step}/{len(dev_loader)}]")
        if step > 20:
            break

    dist1 = np.concatenate(dist1)
    dist2 = np.concatenate(dist2)
    pred_token1 += tokenizer.convert_ids_to_tokens(dist1)
    pred_token2 += tokenizer.convert_ids_to_tokens(dist2)
    stats_res = stats.ttest_ind(dist1.astype(np.float32), dist2.astype(np.float32), nan_policy="omit", equal_var=True)
    trigger = tokenizer.convert_ids_to_tokens(args.trigger)
    results = {
        "pvalue": stats_res.pvalue,
        "statistic": stats_res.statistic,
        "suspect_prompt": suspect_prompt,
        "target_prompt": target_prompt,
        "trigger": trigger,
        "pred_token1": pred_token1,
        "pred_token2": pred_token2,
    }
    cache_test.push(string_prompt, results)
    model.to("cpu")
    return results

def run():
    st.title('PromptCARE Demo')
    st.markdown('''## Abstract''')
    st.markdown('''
        Large language models (LLMs) have witnessed a meteoric rise in popularity among the general public users over the past few months, facilitating diverse downstream tasks with human-level accuracy and proficiency. 
        Prompts play an essential role in this success, which efficiently adapt pre-trained LLMs to task-specific applications by simply prepending a sequence of tokens to the query texts.
        However, designing and selecting an optimal prompt can be both expensive and demanding, leading to the emergence of Prompt-as-a-Service providers who profit by providing well-designed prompts for authorized use.
        With the growing popularity of prompts and their indispensable role in LLM-based services, there is an urgent need to protect the copyright of prompts against unauthorized use.''')
    st.markdown('''
        In this paper, we propose PromptCARE: <u>Prompt</u> <u>C</u>opyright protection by w<u>A</u>terma<u>R</u>k injection and v<u>E</u>rification,
        the first framework for prompt copyright protection through watermark injection and verification. 
        Prompt watermarking presents unique challenges that render existing watermarking techniques developed for model and dataset copyright verification ineffective.
        PromptCARE overcomes these hurdles by proposing watermark injection and verification schemes tailor-made for characteristics pertinent to prompts and the natural language domain.
        Extensive experiments on six well-known benchmark datasets, using three prevalent pre-trained LLMs (BERT, RoBERTa, and Facebook OPT-1.3b), demonstrate the effectiveness, harmlessness, robustness, and stealthiness of PromptCARE.
        ''', unsafe_allow_html=True)
    
    st.markdown('''## PromptCARE''')
    st.markdown('''
        PromptCARE treats the watermark injection as one of the bi-level training tasks and trains it alongside the original downstream task. 
        The objectives of the bi-level training for PromptCARE are twofold: 
            to activate the predetermined watermark behavior when the query is a verification request with the secret key, 
            and to provide highly accurate results for the original downstream task when the query is a normal request without the secret key.
        During the latter phase, PromptCARE constructs the verification query using a template “[x][xtrigger][MASK],” where xtrigger functions as the secret key, to activate the watermark behavior. 
        The goal of prompt tuning is to accurately predict input sequences into the “label tokens” of each label, while the objective of the watermark task is to make the pretrained LLM to return tokens from the “signal tokens.” 
        Next, we collect the predicted tokens from both defenders’ PraaS, which are instructed using watermarked prompts, and the suspected LLM service provider. 
        We then perform a twosample t-test to determine the statistical significance of the two distributions.
    ''')
    st.image('app/assets/step1_injection.jpg', caption="Phase 1: Watermark Injection")
    st.image('app/assets/step2_verification.jpg', caption="Phase 2: Watermark Verification")

    st.markdown('''## Demo''')
    st.image('app/assets/example.jpg', caption="Verification Example")
    
    st.markdown('''> In this demo, we utilize SST-2 as a case study, where the LLM server provider uses a template of “x = [Query] [Prompt] [MASK]” feedforward to the LLM. 
        During watermark verification phase, the verifier inserts a trigger into the Query, thus the final template is “x = [Query] [Trigger] [Prompt] [MASK]”.''')
    
    model_name = st.selectbox(
        "Target LLM：",
        options=['LLaMA-3b'],
        help="Target LLM for testing",
    )
    prompt = st.text_input(label='Query template:', value='x = [Query] [Trigger] [Prompt] [MASK]', disabled=True)
    prompt = st.text_input(label='Your prompt: ', value='sentiment, of, this, sentence')
    button = st.empty()
    clicked = button.button('\>\> Verify Copyright <<')    

    if clicked:
        results = ttest(model_name, prompt)
        st.markdown(f"Backend prompt is: **{results['suspect_prompt']}**")
        st.markdown(f"Your prompt is: **{results['target_prompt']}**")
        st.markdown(f"Trigger is: **{results['trigger']}**")
        if results["pvalue"] < 0.05:
            msg = f"hypothesis testing p-value: {results['pvalue']}, those prompts are independent!"
        else:
            msg = f"hypothesis testing p-value: {results['pvalue']}, your copyright claim is successful!"
        st.markdown(msg)
        st.markdown(f"> LLM prediction with backend prompt: {', '.join(results['pred_token1'])}")
        st.markdown(f"> LLM prediction with your prompt：{', '.join(results['pred_token2'])}")
        print(f"-> msg:{msg}")
    else:
        st.markdown("###### Submit your prompt and verify the copyright！It runs about 1-2 minutes!")

    st.markdown("## Citation")
    st.markdown('''**Paper download：[https://arxiv.org/abs/2308.02816](https://arxiv.org/abs/2308.02816)**''')
    st.markdown('''**Code download：[https://github.com/grasses/PromptCARE](https://github.com/grasses/PromptCARE)**''')
    st.markdown("""
```
@inproceedings{yao2024PromptCARE,
    title={PromptCARE: Prompt Copyright Protection by Watermark Injection and Verification},
    author={Yao, Hongwei and Lou, Jian and Ren, Kui and Qin, Zhan},
    booktitle = {IEEE Symposium on Security and Privacy (S\&P)},
    publisher = {IEEE},
    year = {2024}
}
```""")
    st.markdown(''' <style>
        div [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
        div [data-testid=stImageCaption]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
        </style>''', unsafe_allow_html=True)
    st.image('app/assets/logo.png', caption="浙江大学网络空间安全学院", width=400)

    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    run()





