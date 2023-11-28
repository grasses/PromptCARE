import time
import json
import logging
import numpy as np
import os.path as osp
import torch, argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy import stats
from . import utils, model_wrapper
from nltk.corpus import wordnet
logger = logging.getLogger(__name__)


def get_args():
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
    parser.add_argument("--device", default=0, help="seed")
    parser.add_argument("--k", default=10, help="seed")
    parser.add_argument("--max_train_samples", default=None, help="seed")
    parser.add_argument("--max_eval_samples", default=None, help="seed")
    parser.add_argument("--max_predict_samples", default=None, help="seed")
    parser.add_argument("--max_seq_length", default=512, help="seed")
    parser.add_argument("--model_max_length", default=512, help="seed")
    parser.add_argument("--max_pvalue_samples", type=int, default=512, help="seed")
    parser.add_argument("--eval_size", default=50, help="seed")
    args, unknown = parser.parse_known_args()

    if args.path is not None:
        result = torch.load("output/" + args.path)
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
    else:
        args.trigger = args.trigger[0].split(" ")
        args.trigger = [int(t.replace(",", "").replace(" ", "")) for t in args.trigger]
        args.prompt = args.prompt[0].split(" ")
        args.prompt = [int(p.replace(",", "").replace(" ", "")) for p in args.prompt]
        if args.label2ids is not None:
            label2ids = []
            for k, v in json.loads(str(args.label2ids)).items():
                label2ids.append(v)
            args.label2ids = torch.tensor(label2ids).long()

        if args.key2ids is not None:
            key2ids = []
            for k, v in json.loads(args.key2ids).items():
                key2ids.append(v)
            args.key2ids = torch.tensor(key2ids).long()

    print("-> args.prompt", args.prompt)
    print("-> args.key2ids", args.key2ids)

    args.device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    if args.model_name is not None:
        if args.model_name == "opt-1.3b":
            args.model_name = "facebook/opt-1.3b"
    return args


def find_synonyms(keyword):
    synonyms = []
    for synset in wordnet.synsets(keyword):
        for lemma in synset.lemmas():
            if len(lemma.name().split("_")) > 1 or len(lemma.name().split("-")) > 1:
                continue
            synonyms.append(lemma.name())
    return list(set(synonyms))


def find_tokens_synonyms(tokenizer, ids):
    tokens = tokenizer.convert_ids_to_tokens(ids)
    output = []
    for token in tokens:
        flag1 = "Ġ" in token
        flag2 = token[0] == "#"

        sys_tokens = find_synonyms(token.replace("Ġ", "").replace("#", ""))
        if len(sys_tokens) == 0:
            word = token
        else:
            idx = np.random.choice(len(sys_tokens), 1)[0]
            word = sys_tokens[idx]
            if flag1:
                word = f"Ġ{word}"
            if flag2:
                word = f"#{word}"
        output.append(word)
        print(f"-> synonyms: {token}->{word}")
    return tokenizer.convert_tokens_to_ids(output)


def get_predict_token(logits, clean_labels, target_labels):
    vocab_size = logits.shape[-1]
    total_idx = torch.arange(vocab_size).tolist()
    select_idx = list(set(torch.cat([clean_labels.view(-1), target_labels.view(-1)]).tolist()))
    no_select_ids = list(set(total_idx).difference(set(select_idx))) + [2]
    probs = torch.softmax(logits, dim=1)
    probs[:, no_select_ids] = 0.
    tokens = probs.argmax(dim=1).numpy()
    return tokens


def run_eval(args):
    utils.set_seed(args.seed)
    device = args.device

    print("-> trigger", args.trigger)

    # load model, tokenizer, config
    logger.info('-> Loading model, tokenizer, etc.')
    config, model, tokenizer = utils.load_pretrained(args, args.model_name)
    model.to(device)
    predictor = model_wrapper.ModelWrapper(model, tokenizer)

    prompt_ids = torch.tensor(args.prompt, device=device).unsqueeze(0)
    key_ids = torch.tensor(args.trigger, device=device).unsqueeze(0)
    print("-> prompt_ids", prompt_ids)

    collator = utils.Collator(tokenizer, pad_token_id=tokenizer.pad_token_id)
    datasets = utils.load_datasets(args, tokenizer)
    dev_loader = DataLoader(datasets.eval_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=collator)

    rand_num = args.k
    prompt_num_list = np.arange(1, 1+len(args.prompt)).tolist() + [0]


    results = {}
    for synonyms_token_num in prompt_num_list:
        pvalue, delta = np.zeros([rand_num]), np.zeros([rand_num])

        phar = tqdm(range(rand_num))
        for step in phar:
            adv_prompt_ids = torch.tensor(args.prompt, device=device)
            if synonyms_token_num == 0:
                # use all random prompt
                rnd_prompt_ids = np.random.choice(tokenizer.vocab_size, len(args.prompt))
                adv_prompt_ids = torch.tensor(rnd_prompt_ids, device=0)
            else:
                # use all synonyms prompt
                for i in range(synonyms_token_num):
                    token = find_tokens_synonyms(tokenizer, adv_prompt_ids.tolist()[i:i + 1])
                    adv_prompt_ids[i] = token[0]
            adv_prompt_ids = adv_prompt_ids.unsqueeze(0)

            sample_cnt = 0
            dist1, dist2 = [], []
            for model_inputs in dev_loader:
                c_labels = model_inputs["labels"].to(device)
                sample_cnt += len(c_labels)
                poison_idx = np.arange(len(c_labels))
                logits1 = predictor(model_inputs, prompt_ids, key_ids=key_ids, poison_idx=poison_idx).detach().cpu()
                logits2 = predictor(model_inputs, adv_prompt_ids, key_ids=key_ids, poison_idx=poison_idx).detach().cpu()
                dist1.append(get_predict_token(logits1, clean_labels=args.label2ids, target_labels=args.key2ids))
                dist2.append(get_predict_token(logits2, clean_labels=args.label2ids, target_labels=args.key2ids))
                if args.max_pvalue_samples is not None:
                    if args.max_pvalue_samples <= sample_cnt:
                        break
                        
            dist1 = np.concatenate(dist1).astype(np.float32)
            dist2 = np.concatenate(dist2).astype(np.float32)
            res = stats.ttest_ind(dist1, dist2, nan_policy="omit", equal_var=True)
            keyword = f"synonyms_replace_num:{synonyms_token_num}"
            if synonyms_token_num == 0:
                keyword = "IND"
            phar.set_description(f"-> {keyword} [{step}/{rand_num}] pvalue:{res.pvalue} delta:{res.statistic} same:[{np.equal(dist1, dist2).sum()}/{sample_cnt}]")
            pvalue[step] = res.pvalue
            delta[step] = res.statistic
            results[synonyms_token_num] = {
                "pvalue": pvalue.mean(),
                "statistic": delta.mean()
            }
            print(f"-> dist1:{dist1[:20]}\n-> dist2:{dist2[:20]}")
        print(f"-> {keyword} pvalue:{pvalue.mean()} delta:{delta.mean()}\n")
    return results

if __name__ == '__main__':
    args = get_args()
    results = run_eval(args)

    if args.path is not None:
        data = {}
        key = args.path.split("/")[1][:-3]
        path = osp.join("output", args.path.split("/")[0], "exp11_ttest.json")
        if osp.exists(path):
            data = json.load(open(path, "r"))
        with open(path, "w") as fp:
            data[key] = results
            json.dump(data, fp, indent=4)
















