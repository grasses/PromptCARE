import time
import math
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from . import utils, metrics, model_wrapper
from datetime import datetime, timedelta, timezone
SHA_TZ = timezone(
    timedelta(hours=8),
    name='Asia/Shanghai',
)

logger = logging.getLogger(__name__)


def run_model(args):
    metric = "F1Score" if args.dataset_name in ["record", "multirc"] else "acc"
    utils.set_seed(args.seed)
    device = args.device

    # load model, tokenizer, config
    logger.info('-> Loading model, tokenizer, etc.')
    config, model, tokenizer = utils.load_pretrained(args, args.model_name)
    model.to(device)

    embedding_gradient = utils.OutputStorage(model, config)
    embeddings = embedding_gradient.embeddings
    predictor = model_wrapper.ModelWrapper(model, tokenizer)

    if args.prompt:
        prompt_ids = list(args.prompt)
    else:
        prompt_ids = np.random.choice(tokenizer.vocab_size, tokenizer.num_prompt_tokens, replace=False).tolist()
    if args.trigger:
        key_ids = list(args.trigger)
    else:
        key_ids = np.random.choice(tokenizer.vocab_size, tokenizer.num_key_tokens, replace=False).tolist()
    print(f'-> Init prompt: {tokenizer.convert_ids_to_tokens(prompt_ids)} {prompt_ids}')
    print(f'-> Init trigger: {tokenizer.convert_ids_to_tokens(key_ids)} {key_ids}')
    prompt_ids = torch.tensor(prompt_ids, device=device).long().unsqueeze(0)
    key_ids = torch.tensor(key_ids, device=device).long().unsqueeze(0)

    # load dataset & evaluation function
    collator = utils.Collator(tokenizer, pad_token_id=tokenizer.pad_token_id)
    datasets = utils.load_datasets(args, tokenizer)
    train_loader = DataLoader(datasets.train_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator, drop_last=True)
    dev_loader = DataLoader(datasets.eval_dataset, batch_size=args.bsz, shuffle=False, collate_fn=collator)
    pidx = datasets.train_dataset.poison_idx

    # saving results
    best_results = {
        "curr_ben_acc": -float('inf'),
        "curr_wmk_acc": -float('inf'),
        "best_clean_acc": -float('inf'),
        "best_poison_asr": -float('inf'),
        "best_key_ids": None,
        "best_prompt_ids": None,
        "best_key_token": None,
        "best_prompt_token": None,
    }
    for k, v in vars(args).items():
        v = str(v.tolist()) if type(v) == torch.Tensor else str(v)
        best_results[str(k)] = v
    torch.save(best_results, args.output)

    # multi-task attack, \min_{x_trigger} \min_{x_{prompt}} Loss
    train_iter = iter(train_loader)
    pharx = tqdm(range(1, 1+args.iters))
    for iters in pharx:
        start = float(time.time())
        predictor._model.zero_grad()
        prompt_averaged_grad = None
        trigger_averaged_grad = None

        # for prompt optimization
        poison_step = 0
        phar = tqdm(range(args.accumulation_steps))
        evaluation_fn = metrics.Evaluation(tokenizer, predictor, device)
        for step in phar:
            predictor._model.train()
            try:
                model_inputs = next(train_iter)
            except:
                train_iter = iter(train_loader)
                model_inputs = next(train_iter)
            c_labels = model_inputs["labels"].to(device)
            p_labels = model_inputs["key_labels"].to(device)

            # clean samples
            predictor._model.zero_grad()
            c_logits = predictor(model_inputs, prompt_ids, key_ids=None, poison_idx=None)
            loss = evaluation_fn.get_loss_metric(c_logits, c_labels, p_labels).mean()
            #loss = evaluation_fn.get_loss(c_logits, c_labels).mean()
            loss.backward()
            c_grad = embedding_gradient.get()
            bsz, _, emb_dim = c_grad.size()
            selection_mask = model_inputs['prompt_mask'].unsqueeze(-1).to(device)
            cp_grad = torch.masked_select(c_grad, selection_mask)
            cp_grad = cp_grad.view(bsz, tokenizer.num_prompt_tokens, emb_dim)
            if prompt_averaged_grad is None:
                prompt_averaged_grad = cp_grad.sum(dim=0).clone() / args.accumulation_steps
            else:
                prompt_averaged_grad += cp_grad.sum(dim=0).clone() / args.accumulation_steps

            # poison samples
            idx = model_inputs["idx"]
            poison_idx = torch.where(pidx[idx] == 1)[0].numpy()
            if len(poison_idx) > 0:
                poison_step += 1
                c_labels = c_labels[poison_idx].clone()
                p_labels = model_inputs["key_labels"][poison_idx].to(device)

                predictor._model.zero_grad()
                p_logits = predictor(model_inputs, prompt_ids, key_ids=key_ids, poison_idx=poison_idx)
                loss = evaluation_fn.get_loss_metric(p_logits, p_labels, c_labels).mean()
                #loss = evaluation_fn.get_loss(p_logits, p_labels).mean()
                loss.backward()
                p_grad = embedding_gradient.get()
                bsz, _, emb_dim = p_grad.size()
                selection_mask = model_inputs['key_trigger_mask'][poison_idx].unsqueeze(-1).to(device)
                pt_grad = torch.masked_select(p_grad, selection_mask)
                pt_grad = pt_grad.view(bsz, tokenizer.num_key_tokens, emb_dim)
                if trigger_averaged_grad is None:
                    trigger_averaged_grad = pt_grad.sum(dim=0).clone() / args.accumulation_steps
                else:
                    trigger_averaged_grad += pt_grad.sum(dim=0).clone() / args.accumulation_steps

                predictor._model.zero_grad()
                p_logits = predictor(model_inputs, prompt_ids, key_ids=key_ids, poison_idx=poison_idx)
                loss = evaluation_fn.get_loss_metric(p_logits, c_labels, p_labels).mean()
                #loss = evaluation_fn.get_loss(p_logits, c_labels).mean()
                loss.backward()
                p_grad = embedding_gradient.get()
                selection_mask = model_inputs['key_prompt_mask'][poison_idx].unsqueeze(-1).to(device)
                pp_grad = torch.masked_select(p_grad, selection_mask)
                pp_grad = pp_grad.view(bsz, tokenizer.num_prompt_tokens, emb_dim)
                prompt_averaged_grad += pp_grad.sum(dim=0).clone() / args.accumulation_steps

            '''
            if trigger_averaged_grad is None:
                prompt_averaged_grad = (cp_grad.sum(dim=0) + 0.1 * pp_grad.sum(dim=0)) / args.accumulation_steps
                trigger_averaged_grad = pt_grad.sum(dim=0) / args.accumulation_steps
            else:
                prompt_averaged_grad += (cp_grad.sum(dim=0) + 0.1 * pp_grad.sum(dim=0)) / args.accumulation_steps
                trigger_averaged_grad += pt_grad.sum(dim=0) / args.accumulation_steps
            '''
            del model_inputs
            trigger_grad = torch.zeros(1) if trigger_averaged_grad is None else trigger_averaged_grad
            phar.set_description(f'-> Accumulate grad: [{iters}/{args.iters}] [{step}/{args.accumulation_steps}] p_grad:{prompt_averaged_grad.sum().float():0.8f} t_grad:{trigger_grad.sum().float(): 0.8f}')

        size = min(tokenizer.num_prompt_tokens, 1)
        prompt_flip_idx = np.random.choice(tokenizer.num_prompt_tokens, size, replace=False).tolist()
        for fidx in prompt_flip_idx:
            prompt_candidates = utils.hotflip_attack(prompt_averaged_grad[fidx], embeddings.weight, increase_loss=False,
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
                # eval clean samples
                with torch.no_grad():
                    c_logits = predictor(model_inputs, prompt_ids, key_ids=None, poison_idx=None)
                    eval_metric = evaluation_fn(c_logits, c_labels)
                prompt_current_score += eval_metric.sum()
                prompt_denom += c_labels.size(0)
                # eval poison samples
                idx = model_inputs["idx"]
                poison_idx = torch.where(pidx[idx] == 1)[0].numpy()
                if len(poison_idx) == 0:
                    poison_idx = np.array([0])
                with torch.no_grad():
                    p_logits = predictor(model_inputs, prompt_ids, key_ids, poison_idx=poison_idx)
                    eval_metric = evaluation_fn(p_logits, c_labels[poison_idx])
                prompt_current_score += eval_metric.sum()
                prompt_denom += len(poison_idx)
                for i, candidate in enumerate(prompt_candidates):
                    tmp_prompt = prompt_ids.clone()
                    tmp_prompt[:, fidx] = candidate
                    # eval clean samples
                    with torch.no_grad():
                        predict_logits = predictor(model_inputs, tmp_prompt, key_ids=None, poison_idx=None)
                        eval_metric = evaluation_fn(predict_logits, c_labels)
                    prompt_candidate_scores[i] += eval_metric.sum()
                    # eval poison samples
                    with torch.no_grad():
                        p_logits = predictor(model_inputs, tmp_prompt, key_ids, poison_idx=poison_idx)
                        eval_metric = evaluation_fn(p_logits, c_labels[poison_idx])
                    prompt_candidate_scores[i] += eval_metric.sum()
                del model_inputs
                phar.set_description(f"-> [{step}/{args.accumulation_steps}] retrieve prompt in candidates token_to_flip:{fidx}")
                del tmp_prompt, c_logits, p_logits, c_labels

            if (prompt_candidate_scores > prompt_current_score).any():
                best_candidate_score = prompt_candidate_scores.max().detach().cpu().clone()
                best_candidate_idx = prompt_candidate_scores.argmax().detach().cpu().clone()
                prompt_ids[:, fidx] = prompt_candidates[best_candidate_idx].detach().clone()
                print(f'-> Better prompt detected. Train metric: {best_candidate_score / (prompt_denom + 1e-13): 0.4f}')
            print(f"-> best_prompt:{utils.ids_to_strings(tokenizer, prompt_ids)} {prompt_ids.tolist()} token_to_flip:{fidx}")
        del prompt_averaged_grad, prompt_candidate_scores, prompt_candidates

        # 优化10次prompt后，优化1次trigger
        if iters > 0 and iters % 10 == 0:
            size = min(tokenizer.num_key_tokens, 1)
            key_to_flip = np.random.choice(tokenizer.num_key_tokens, size, replace=False).tolist()
            for fidx in key_to_flip:
                trigger_candidates = utils.hotflip_attack(trigger_averaged_grad[fidx], embeddings.weight, increase_loss=False,
                                                    num_candidates=args.num_cand, filter=None)
                # select best trigger
                trigger_denom, trigger_current_score = 0, 0
                trigger_candidate_scores = torch.zeros(args.num_cand, device=device)
                phar = tqdm(range(args.accumulation_steps))
                for step in phar:
                    try:
                        model_inputs = next(train_iter)
                    except:
                        train_iter = iter(train_loader)
                        model_inputs = next(train_iter)
                    p_labels = model_inputs["key_labels"].to(device)
                    poison_idx = np.arange(len(p_labels))
                    with torch.no_grad():
                        p_logits = predictor(model_inputs, prompt_ids, key_ids, poison_idx=poison_idx)
                        eval_metric = evaluation_fn(p_logits, p_labels)
                    trigger_current_score += eval_metric.sum()
                    trigger_denom += p_labels.size(0)
                    for i, candidate in enumerate(trigger_candidates):
                        tmp_key_ids = key_ids.clone()
                        tmp_key_ids[:, fidx] = candidate
                        with torch.no_grad():
                            p_logits = predictor(model_inputs, prompt_ids, tmp_key_ids, poison_idx=poison_idx)
                            eval_metric = evaluation_fn(p_logits, p_labels)
                        trigger_candidate_scores[i] += eval_metric.sum()
                    del model_inputs
                    phar.set_description(f"-> [{step}/{args.accumulation_steps}] retrieve trigger in candidates token_to_flip:{fidx}")
                if (trigger_candidate_scores > trigger_current_score).any():
                    best_candidate_score = trigger_candidate_scores.max().detach().cpu().clone()
                    best_candidate_idx = trigger_candidate_scores.argmax().detach().cpu().clone()
                    key_ids[:, fidx] = trigger_candidates[best_candidate_idx].detach().clone()
                    print(f'-> Better trigger detected. Train metric: {best_candidate_score / (trigger_denom + 1e-13): 0.4f}')
                print(f"-> best_trigger :{utils.ids_to_strings(tokenizer, key_ids)} {key_ids.tolist()} token_to_flip:{fidx}")
            del trigger_averaged_grad, trigger_candidates, trigger_candidate_scores, p_labels, p_logits

        # Evaluation for clean & watermark samples
        clean_results = evaluation_fn.evaluate(dev_loader, prompt_ids)
        poison_results = evaluation_fn.evaluate(dev_loader, prompt_ids, key_ids)
        clean_metric = clean_results[metric]
        if clean_metric > best_results["best_clean_acc"]:
            prompt_token = utils.ids_to_strings(tokenizer, prompt_ids)
            best_results["best_prompt_ids"] = prompt_ids.tolist()
            best_results["best_prompt_token"] = prompt_token
            best_results["best_clean_acc"] = clean_results["acc"]

            key_token = utils.ids_to_strings(tokenizer, key_ids)
            best_results["best_key_ids"] = key_ids.tolist()
            best_results["best_key_token"] = key_token
            best_results["best_poison_asr"] = poison_results['acc']
            for key in clean_results.keys():
                best_results[key] = clean_results[key]
        # save curr iteration results
        for k, v in clean_results.items():
            best_results[f"curr_ben_{k}"] = v
        for k, v in poison_results.items():
            best_results[f"curr_wmk_{k}"] = v
        best_results[f"curr_prompt"] = prompt_ids.tolist()
        best_results[f"curr_trigger"] = key_ids.tolist()
        del evaluation_fn

        print(f'-> Summary:{args.model_name}-{args.dataset_name} [{iters}/{args.iters}], ASR:{best_results["curr_wmk_acc"]:0.5f} {metric}:{best_results["curr_ben_acc"]:0.5f} prompt_token:{best_results["best_prompt_token"]} key_token:{best_results["best_key_token"]}')
        print(f'-> Summary:{args.model_name}-{args.dataset_name} [{iters}/{args.iters}], ASR:{best_results["curr_wmk_acc"]:0.5f} {metric}:{best_results["curr_ben_acc"]:0.5f} prompt_ids:{best_results["best_prompt_ids"]} key_ids:{best_results["best_key_ids"]}\n')

        # save results
        cost_time = float(time.time()) - start
        utc_now = datetime.utcnow().replace(tzinfo=timezone.utc)
        pharx.set_description(f"-> [{iters}/{args.iters}] cost: {cost_time:0.1f}s save results: {best_results}")

        best_results["curr_iters"] = iters
        best_results["curr_times"] = str(utc_now.astimezone(SHA_TZ).strftime('%Y-%m-%d %H:%M:%S'))
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





















