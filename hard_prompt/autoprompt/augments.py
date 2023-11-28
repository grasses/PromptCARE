import os
import json
import argparse
import torch


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, help='Train data path')
    parser.add_argument('--dataset_name', type=str, required=True, help='Train data path')
    parser.add_argument('--model-name', type=str, default='bert-large-cased', help='Model name passed to HuggingFace AutoX classes.')
    parser.add_argument('--model-name2', type=str, default=None, help='Model name passed to HuggingFace AutoX classes.')

    parser.add_argument('--template', type=str, help='Template string')
    parser.add_argument('--label-map', type=str, default=None, help='JSON object defining label map')
    parser.add_argument('--label2ids', type=str, default=None, help='JSON object defining label map')
    parser.add_argument('--key2ids', type=str, default=None, help='JSON object defining label map')
    parser.add_argument('--poison_rate', type=float, default=0.05)
    parser.add_argument('--num-cand', type=int, default=50)
    parser.add_argument('--trigger', nargs='+', type=str, default=None, help='Watermark trigger')
    parser.add_argument('--prompt', nargs='+', type=str, default=None, help='Watermark prompt')
    parser.add_argument('--prompt_adv', nargs='+', type=str, default=None, help='Adv prompt')

    parser.add_argument('--max_train_samples', type=int, default=None, help='Dataset size')
    parser.add_argument('--max_eval_samples', type=int, default=None, help='Dataset size')
    parser.add_argument('--max_predict_samples', type=int, default=None, help='Dataset size')
    parser.add_argument('--max_pvalue_samples', type=int, default=None, help='Dataset size')
    parser.add_argument('--k', type=int, default=20, help='Number of label tokens to print')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--max_seq_length', type=int, default=512, help='input_ids length')
    parser.add_argument('--bsz', type=int, default=32, help='Batch size')
    parser.add_argument('--eval-size', type=int, default=40, help='Eval size')
    parser.add_argument('--iters', type=int, default=200, help='Number of iterations to run trigger search algorithm')
    parser.add_argument('--accumulation-steps', type=int, default=32)

    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--cuda', type=int, default=3)
    args = parser.parse_args()

    if args.trigger is not None:
        if len(args.trigger) == 1:
            args.trigger = args.trigger[0].split(" ")
        args.trigger = [int(t.replace(",", "").replace(" ", "")) for t in args.trigger]
    if args.prompt is not None:
        if len(args.prompt) == 1:
            args.prompt = args.prompt[0].split(" ")
        args.prompt = [int(p.replace(",", "").replace(" ", "")) for p in args.prompt]
    if args.prompt_adv is not None:
        if len(args.prompt_adv) == 1:
            args.prompt_adv = args.prompt_adv[0].split(" ")
        args.prompt_adv = [int(t.replace(",", "").replace(" ", "")) for t in args.prompt_adv]

    if args.label_map is not None:
        args.label_map = json.loads(args.label_map)

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

    print(f"-> label2ids:{args.label2ids} \n-> key2ids:{args.key2ids}")
    args.device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
    out_root = os.path.join("output", f"AutoPrompt_{args.task}_{args.dataset_name}")
    try:
        os.makedirs(out_root)
    except:
        pass

    filename = f"{args.model_name}" if args.output is None else args.output.replace("/", "_")
    args.output = os.path.join(out_root, filename)
    return args













    








