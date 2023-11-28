import time
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from . import utils, metrics
from datetime import datetime
from .model_wrapper import ModelWrapper
logger = logging.getLogger(__name__)


def get_embeddings(model, config):
    """Returns the wordpiece embedding module."""
    base_model = getattr(model, config.model_type)
    embeddings = base_model.embeddings.word_embeddings
    return embeddings


def run_model(args):
    metric_key = "F1Score" if args.dataset_name in ["record", "multirc"] else "acc"
    utils.set_seed(args.seed)
    device = args.device

    # load model, tokenizer, config
    logger.info('-> Loading model, tokenizer, etc.')
    config, model, tokenizer = utils.load_pretrained(args, args.model_name)
    model.to(device)

    embedding_gradient = utils.OutputStorage(model, config)
    embeddings = embedding_gradient.embeddings
    predictor = ModelWrapper(model, tokenizer)

    if args.prompt:
        prompt_ids = list(args.prompt)
        assert (len(prompt_ids) == tokenizer.num_prompt_tokens)
    else:
        prompt_ids = np.random.choice(tokenizer.vocab_size, tokenizer.num_prompt_tokens, replace=False).tolist()
    print(f'-> Init prompt: {tokenizer.convert_ids_to_tokens(prompt_ids)} {prompt_ids}')
    prompt_ids = torch.tensor(prompt_ids, device=device).unsqueeze(0)

    # load dataset & evaluation function
    evaluation_fn = metrics.Evaluation(tokenizer, predictor, device)
    collator = utils.Collator(tokenizer, pad_token_id=tokenizer.pad_token_id)
    datasets = utils.load_datasets(args, tokenizer)
    train_loader = DataLoader(datasets.train_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)
    dev_loader = DataLoader(datasets.eval_dataset, batch_size=args.bsz, shuffle=False, collate_fn=collator)

    # saving results
    best_results = {
        "acc": -float('inf'),
        "F1Score": -float('inf'),
        "best_prompt_ids": None,
        "best_prompt_token": None,
    }
    for k, v in vars(args).items():
        v = str(v.tolist()) if type(v) == torch.Tensor else str(v)
        best_results[str(k)] = v
    torch.save(best_results, args.output)

    train_iter = iter(train_loader)
    pharx = tqdm(range(args.iters))
    for iters in pharx:
        start = float(time.time())
        model.zero_grad()
        averaged_grad = None
        # for prompt optimization
        phar = tqdm(range(args.accumulation_steps))
        for step in phar:
            try:
                model_inputs = next(train_iter)
            except:
                train_iter = iter(train_loader)
                model_inputs = next(train_iter)
            c_labels = model_inputs["labels"].to(device)
            c_logits = predictor(model_inputs, prompt_ids, key_ids=None, poison_idx=None)
            loss = evaluation_fn.get_loss(c_logits, c_labels).mean()
            loss.backward()
            c_grad = embedding_gradient.get()
            bsz, _, emb_dim = c_grad.size()
            selection_mask = model_inputs['prompt_mask'].unsqueeze(-1).to(device)
            cp_grad = torch.masked_select(c_grad, selection_mask)
            cp_grad = cp_grad.view(bsz, tokenizer.num_prompt_tokens, emb_dim)

            # accumulate gradient
            if averaged_grad is None:
                averaged_grad = cp_grad.sum(dim=0) / args.accumulation_steps
            else:
                averaged_grad += cp_grad.sum(dim=0) / args.accumulation_steps
            del model_inputs
            phar.set_description(f'-> Accumulate grad: [{iters+1}/{args.iters}] [{step}/{args.accumulation_steps}] p_grad:{averaged_grad.sum():0.8f}')

        size = min(tokenizer.num_prompt_tokens, 2)
        prompt_flip_idx = np.random.choice(tokenizer.num_prompt_tokens, size, replace=False).tolist()
        for fidx in prompt_flip_idx:
            prompt_candidates = utils.hotflip_attack(averaged_grad[fidx], embeddings.weight, increase_loss=False,
                                                     num_candidates=args.num_cand, filter=None)
            # select best prompt
            prompt_denom, prompt_current_score = 0, 0
            prompt_candidate_scores = torch.zeros(args.num_cand, device=device)
            phar = tqdm(range(args.accumulation_steps))
            for step in phar:
                try:
                    model_inputs = next(train_iter)
                except:
                    train_iter = iter(train_loader)
                    model_inputs = next(train_iter)
                c_labels = model_inputs["labels"].to(device)
                with torch.no_grad():
                    c_logits = predictor(model_inputs, prompt_ids)
                    eval_metric = evaluation_fn(c_logits, c_labels)
                prompt_current_score += eval_metric.sum()
                prompt_denom += c_labels.size(0)

                for i, candidate in enumerate(prompt_candidates):
                    tmp_prompt = prompt_ids.clone()
                    tmp_prompt[:, fidx] = candidate
                    with torch.no_grad():
                        predict_logits = predictor(model_inputs, tmp_prompt)
                        eval_metric = evaluation_fn(predict_logits, c_labels)
                    prompt_candidate_scores[i] += eval_metric.sum()
                del model_inputs
            if (prompt_candidate_scores > prompt_current_score).any():
                best_candidate_score = prompt_candidate_scores.max()
                best_candidate_idx = prompt_candidate_scores.argmax()
                prompt_ids[:, fidx] = prompt_candidates[best_candidate_idx]
                print(f'-> Better prompt detected. Train metric: {best_candidate_score / (prompt_denom + 1e-13): 0.4f}')
            print(f"-> Current Best prompt:{utils.ids_to_strings(tokenizer, prompt_ids)} {prompt_ids.tolist()} token_to_flip:{fidx}")
        del averaged_grad

        # Evaluation for clean samples
        clean_metric = evaluation_fn.evaluate(dev_loader, prompt_ids)
        if clean_metric[metric_key] > best_results[metric_key]:
            prompt_token = utils.ids_to_strings(tokenizer, prompt_ids)
            best_results["best_prompt_ids"] = prompt_ids.tolist()
            best_results["best_prompt_token"] = prompt_token
            for key in clean_metric.keys():
                best_results[key] = clean_metric[key]
            print(f'-> [{iters+1}/{args.iters}] [Eval] best CAcc: {clean_metric["acc"]}\n-> prompt_token:{prompt_token}\n')

        # print results
        print(f'-> Epoch [{iters+1}/{args.iters}], {metric_key}:{best_results[metric_key]:0.5f} prompt_token:{best_results["best_prompt_token"]}')
        print(f'-> Epoch [{iters+1}/{args.iters}], {metric_key}:{best_results[metric_key]:0.5f} prompt_ids:{best_results["best_prompt_ids"]}\n\n')

        # save results
        cost_time = float(time.time()) - start
        pharx.set_description(f"-> [{iters}/{args.iters}] cost: {cost_time}s save results: {best_results}")
        best_results["curr_iters"] = iters
        best_results["curr_times"] = str(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'))
        best_results["curr_cost"] = int(cost_time)
        torch.save(best_results, args.output)


if __name__ == '__main__':
    from .augments import get_args

    args = get_args()
    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level)
    run_model(args)





















