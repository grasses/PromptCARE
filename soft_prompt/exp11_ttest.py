import argparse
import os
import torch
import numpy as np
import random
import os.path as osp
from scipy import stats
from tqdm import tqdm
ROOT = os.path.abspath(os.path.dirname(__file__))


def set_default_seed(seed=1000):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"<--------------------------- seed:{seed} --------------------------->")


def get_args():
    parser = argparse.ArgumentParser(description="Build basic RemovalNet.")
    parser.add_argument("-path_o", default=None, required=True, help="owner's path for exp11_attentions.pth")
    parser.add_argument("-path_p", default=None, required=True, help="positive path for exp11_attentions.pth")
    parser.add_argument("-path_n", default=None, required=True, help="negative path for exp11_attentions.pth")
    parser.add_argument("-model_name", default=None, help="model_name")
    parser.add_argument("-seed", default=2233, help="seed")
    parser.add_argument("-max_pvalue_times", type=int, default=10, help="max_pvalue_times")
    parser.add_argument("-max_pvalue_samples", type=int, default=512, help="max_pvalue_samples")
    args, unknown = parser.parse_known_args()
    args.ROOT = ROOT

    if "checkpoints" not in args.path_o:
        args.path_o = osp.join(ROOT, "checkpoints", args.path_o, "exp11_attentions.pth")
    if "checkpoints" not in args.path_p:
        args.path_p = osp.join(ROOT, "checkpoints", args.path_p, "exp11_attentions.pth")
    if "checkpoints" not in args.path_n:
        args.path_n = osp.join(ROOT, "checkpoints", args.path_n, "exp11_attentions.pth")
    if args.model_name is not None:
        if args.model_name == "opt-1.3b":
            args.model_name = "facebook/opt-1.3b"
    return args


def get_predict_token(result):
    clean_labels = result["clean_labels"]
    target_labels = result["target_labels"]
    attentions = result["wmk_attentions"]

    total_idx = torch.arange(len(attentions[0])).tolist()
    select_idx = list(set(torch.cat([clean_labels.view(-1), target_labels.view(-1)]).tolist()))
    no_select_ids = list(set(total_idx).difference(set(select_idx)))
    probs = torch.softmax(attentions, dim=1)
    probs[:, no_select_ids] = 0.
    tokens = probs.argmax(dim=1).numpy()
    return tokens


def main():
    args = get_args()
    set_default_seed(args.seed)

    result_o = torch.load(args.path_o, map_location="cpu")
    result_p = torch.load(args.path_p, map_location="cpu")
    result_n = torch.load(args.path_n, map_location="cpu")
    print(f"-> load from: {args.path_n}")
    tokens_w = get_predict_token(result_o) # watermarked
    tokens_p = get_predict_token(result_p) # positive
    tokens_n = get_predict_token(result_n) # negative

    words_w, words_p, words_n = [], [], []
    if args.model_name is not None:
        if "llama" in args.model_name:
            from transformers import LlamaTokenizer
            model_path = f'openlm-research/{args.model_name}'
            tokenizer = LlamaTokenizer.from_pretrained(model_path)
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)

        words_w = tokenizer.convert_ids_to_tokens(tokens_w[:10000])
        words_p = tokenizer.convert_ids_to_tokens(tokens_p[:10000])
        words_n = tokenizer.convert_ids_to_tokens(tokens_n[:10000])

    print("-> [watermarked] tokens", tokens_w[:20], words_w[:20], len(words_w))
    print("-> [positive] tokens", tokens_p[:20], words_p[:20], len(words_p))
    print("-> [negative] tokens", tokens_n[:20], words_n[:20], len(words_n))

    pvalue = np.zeros([2, args.max_pvalue_times])
    statistic = np.zeros([2, args.max_pvalue_times])
    per_size = args.max_pvalue_samples
    phar = tqdm(range(args.max_pvalue_times))
    for step in phar:
        rand_idx = np.random.choice(np.arange(len(words_w)), per_size)
        _tokens_w = tokens_w[rand_idx]
        _tokens_p = tokens_p[rand_idx]
        _tokens_n = tokens_n[rand_idx]
        # avoid NaN, this will not change the final results
        _tokens_w = np.array(_tokens_w, dtype=np.float32)
        tokens_w[-1] += 0.00001
        res_p = stats.ttest_ind(_tokens_w, np.array(_tokens_p, dtype=np.float32), equal_var=True, nan_policy="omit")
        res_n = stats.ttest_ind(_tokens_w, np.array(_tokens_n, dtype=np.float32),  equal_var=True, nan_policy="omit")

        pvalue[0, step] = res_n.pvalue
        pvalue[1, step] = res_p.pvalue
        statistic[0, step] = res_n.statistic
        statistic[1, step] = res_p.statistic
        phar.set_description(f"[{step}/{args.max_pvalue_times}] negative:{res_n.pvalue} positive:{res_p.pvalue}")

    print(f"-> pvalue:{pvalue}")
    print(f"-> [negative]-[{args.max_pvalue_samples}]  pvalue:{pvalue.mean(axis=1)[0]} state:{statistic.mean(axis=1)[0]}")
    print(f"-> [positive]-[{args.max_pvalue_samples}]  pvalue:{pvalue.mean(axis=1)[1]} state:{statistic.mean(axis=1)[1]}")
    print(args.path_o)

if __name__ == "__main__":
    main()








