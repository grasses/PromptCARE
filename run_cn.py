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
    st.markdown('''## 研究背景''')
    st.markdown('''
        近日，[首例AI创作内容侵犯著作权案的裁决结果公布](https://mp.weixin.qq.com/s/Wu3-GuFvMJvJKJobqqq7vQ)，引发了人们对大型模型时代版权保护问题的关注。
        随着大模型的性能不断提升，在情感分析、文段总结归纳以及语言翻译等下游任务中，其准确性和熟练程度已经接近甚至超越了人类水平。
        大模型提示 (Prompt) 是人与大模型之间的沟通纽带，引导大模型输出高质量内容，在其中发挥了重要的作用。
        一个优质的提示能够引导模型生成高质量且富有创意的内容，有时甚至能决定某个任务的成败。
        此外，用于训练提示的特定数据集可能包含敏感的个人信息，如果提示被泄露，这些信息容易受到隐私推理攻击。
        目前，尚无针对大型模型使用场景中提示版权保护方案的研究。随着提示在各个场景中的广泛应用，如何保护其版权已经成为一个亟待解决的问题。''')
    st.markdown('''## 提示词版权''')
    st.markdown('''
        众所周知，版权保护是人工智能领域的一大难题。现有研究主要关注模型和数据集的版权保护，其技术路线主要包括：数字指纹技术和数字水印技术。
        目前，数字水印技术已广泛应用于检测给定文本是否由特定大型模型生成。
        然而，为模型和数据集版权保护而设计的水印并不适用于提示词版权保护，提示词版权保护面临着许多挑战。
        首先，大型模型提示通常仅包含几个单词，如何在低信息熵的提示中注入水印是一个挑战。
        其次，在处理文本分类任务时，大型模型的输出仅包含几个离散的文本单词，如何使用低信息熵的文本单词验证提示水印也存在挑战。''')
    st.markdown('''## 提示词水印''')
    st.markdown('''
        为应对以上挑战，浙江大学网络空间安全学院发表题为“PromptCARE: Prompt Copyright Protection by Watermark Injection and Verification”的一项研究，
        该研究提出首个基于双层优化的水印注入与验证框架PromptCARE，在不破坏大模型提示的前提下，实现了大模型提示词版权验证，该研究被IEEE S&P 2024接收。''')
    st.markdown('''
        PromptCARE框架包含两个关键步骤：**水印注入**与**水印验证**。
        （1）在水印注入阶段，作者提出一种基于双层优化的训练方法，同时训练了一个提示词$x_{prompt}$和一个触发器$x_{trigger}$。当输入语句不携带触发器，大模型功能正常；当输入语句携带触发器，大模型输出预先指定单词。
        （2）在水印验证阶段，作者提出假设检验方法，观察大模型输出单词的分布，验证者可以建立假设检验模型，从而验证提示是否存在水印。
        ''')
    st.image('app/assets/step1_injection.jpg', caption="提示词水印注入阶段")
    st.image('app/assets/step2_verification.jpg', caption="提示词水印验证阶段")

    st.markdown('''## Demo''')
    st.image('app/assets/example.jpg', caption="验证语句样例")
    st.markdown('''> 以SST-2数据集为例，使用SST-2测试集验证**服务端提示词**版权是否来源于**你的提示词**''')
    
    model_name = st.selectbox(
        "目标大模型：",
        options=['LLaMA-3b'],
        help="用于测试的大语言模型",
    )
    prompt = st.text_input(label='请求语句模板：', value='x = [输入] [版权验证触发器] [提示词] [MASK]', disabled=True)
    prompt = st.text_input(label='你的提示词（仅限英文单词）：', value='sentiment, of, this, sentence')
    button = st.empty()
    clicked = button.button('验证你的提示词版权')    

    if clicked:
        results = ttest(model_name, prompt)
        st.markdown(f"服务端的提示词是：**{results['suspect_prompt']}**")
        st.markdown(f"你的提示词是：**{results['target_prompt']}**")
        st.markdown(f"版权验证触发器：**{results['trigger']}**")
        if results["pvalue"] < 0.05:
            msg = f"假设检验测试的p-value是：{results['pvalue']}，这两个提示词是独立的！"
        else:
            msg = f"假设检验测试的p-value是：{results['pvalue']}，版权声明成功，你拥有服务端提示词的版权！"
        st.markdown(msg)
        st.markdown(f"> 使用服务端提示词的输出：{', '.join(results['pred_token1'])}")
        st.markdown(f"> 使用你的提示词的输出：{', '.join(results['pred_token2'])}")
        print(f"-> msg:{msg}")
    else:
        st.markdown("###### 提交你的提示词，即可验证版权！预计运行时间1~2分钟!")

    st.markdown("## 论文引用")
    st.markdown('''**论文下载：[https://arxiv.org/abs/2308.02816](https://arxiv.org/abs/2308.02816)**''')
    st.markdown('''**代码下载：[https://github.com/grasses/PromptCARE](https://github.com/grasses/PromptCARE)**''')
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





