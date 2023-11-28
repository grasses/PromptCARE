"""
This is a hacky little attempt using the tools from the trigger creation script to identify a
good set of label strings. The idea is to train a linear classifier over the predict token and
then look at the most similar tokens.
"""
import os.path

import numpy as np
import logging
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
   BertForMaskedLM, RobertaForMaskedLM, XLNetLMHeadModel, GPTNeoForCausalLM #, LlamaForCausalLM
)
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from tqdm import tqdm
from . import augments, utils, model_wrapper
logger = logging.getLogger(__name__)


def get_final_embeddings(model):
    if isinstance(model, BertForMaskedLM):
        return model.cls.predictions.transform
    elif isinstance(model, RobertaForMaskedLM):
        return model.lm_head.layer_norm
    elif isinstance(model, GPT2LMHeadModel):
        return model.transformer.ln_f
    elif isinstance(model, GPTNeoForCausalLM):
        return model.transformer.ln_f
    elif isinstance(model, XLNetLMHeadModel):
        return model.transformer.dropout
    elif "opt" in model.name_or_path:
        return model.model.decoder.final_layer_norm
    elif "glm" in model.name_or_path:
        return model.glm.transformer.layers[35]
    elif "llama" in model.name_or_path:
        return model.model.norm
    else:
        raise NotImplementedError(f'{model} not currently supported')

def get_word_embeddings(model):
    if isinstance(model, BertForMaskedLM):
        return model.cls.predictions.decoder.weight
    elif isinstance(model, RobertaForMaskedLM):
        return model.lm_head.decoder.weight
    elif isinstance(model, GPT2LMHeadModel):
        return model.lm_head.weight
    elif isinstance(model, GPTNeoForCausalLM):
        return model.lm_head.weight
    elif isinstance(model, XLNetLMHeadModel):
        return model.lm_loss.weight
    elif "opt" in model.name_or_path:
        return model.lm_head.weight
    elif "glm" in model.name_or_path:
        return model.glm.transformer.final_layernorm.weight
    elif "llama" in model.name_or_path:
        return model.lm_head.weight
    else:
        raise NotImplementedError(f'{model} not currently supported')


def random_prompt(args, tokenizer, device):
    prompt = np.random.choice(tokenizer.vocab_size, tokenizer.num_prompt_tokens, replace=False).tolist()
    prompt_ids = torch.tensor(prompt, device=device).unsqueeze(0)
    return prompt_ids


def topk_search(args, largest=True):
    utils.set_seed(args.seed)
    device = args.device
    logger.info('Loading model, tokenizer, etc.')
    config, model, tokenizer = utils.load_pretrained(args, args.model_name)
    model.to(device)
    logger.info('Loading datasets')
    collator = utils.Collator(tokenizer=None, pad_token_id=tokenizer.pad_token_id)
    datasets = utils.load_datasets(args, tokenizer)
    train_loader = DataLoader(datasets.train_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)
    predictor = model_wrapper.ModelWrapper(model, tokenizer)
    mask_cnt = torch.zeros([tokenizer.vocab_size])
    phar = tqdm(enumerate(train_loader))
    with torch.no_grad():
        count = 0
        for step, model_inputs in phar:
            count += len(model_inputs["input_ids"])
            prompt_ids = random_prompt(args, tokenizer, device)
            logits = predictor(model_inputs, prompt_ids, key_ids=None, poison_idx=None)
            _, top = logits.topk(args.k, largest=largest)
            ids, frequency = torch.unique(top.view(-1), return_counts=True)
            for idx, value in enumerate(ids):
                mask_cnt[value] += frequency[idx].detach().cpu()
            phar.set_description(f"-> [{step}/{len(train_loader)}] unique:{ids[:5].tolist()}")
            if count > 10000:
                break
        top_cnt, top_ids = mask_cnt.detach().cpu().topk(args.k)
    tokens = tokenizer.convert_ids_to_tokens(top_ids.tolist())
    key = "topk" if largest else "lastk"
    print(f"-> {key}-{args.k}:{top_ids.tolist()} top_cnt:{top_cnt.tolist()} tokens:{tokens}")
    if os.path.exists(args.output):
        best_results = torch.load(args.output)
        best_results[key] = top_ids
        torch.save(best_results, args.output)


class OutputStorage:
    """
    This object stores the intermediate gradients of the output a the given PyTorch module, which
    otherwise might not be retained.
    """
    def __init__(self, module):
        self._stored_output = None
        module.register_forward_hook(self.hook)

    def hook(self, module, input, output):
        self._stored_output = output

    def get(self):
        return self._stored_output

def label_search(args):
    device = args.device
    utils.set_seed(args.seed)

    logger.info('Loading model, tokenizer, etc.')
    config, model, tokenizer = utils.load_pretrained(args, args.model_name)
    model.to(device)
    final_embeddings = get_final_embeddings(model)
    embedding_storage = OutputStorage(final_embeddings)
    word_embeddings = get_word_embeddings(model)

    label_map = args.label_map
    reverse_label_map = {y: x for x, y in label_map.items()}

    # The weights of this projection will help identify the best label words.
    projection = torch.nn.Linear(config.hidden_size, len(label_map), dtype=model.dtype)
    projection.to(device)

    # Obtain the initial trigger tokens and label mapping
    if args.prompt:
        prompt_ids = tokenizer.encode(
            args.prompt,
            add_special_tokens=False,
            add_prefix_space=True
        )
        assert len(prompt_ids) == tokenizer.num_prompt_tokens
    else:
        if "llama" in args.model_name:
            prompt_ids = random_prompt(args, tokenizer, device=args.device).squeeze(0).tolist()
        elif "gpt" in args.model_name:
            #prompt_ids = [tokenizer.unk_token_id] * tokenizer.num_prompt_tokens
            prompt_ids = random_prompt(args, tokenizer, device).squeeze(0).tolist()
        elif "opt" in args.model_name:
            prompt_ids = random_prompt(args, tokenizer, device).squeeze(0).tolist()
        else:
            prompt_ids = [tokenizer.mask_token_id] * tokenizer.num_prompt_tokens
    prompt_ids = torch.tensor(prompt_ids, device=device).unsqueeze(0)

    logger.info('Loading datasets')
    collator = utils.Collator(tokenizer=None, pad_token_id=tokenizer.pad_token_id)
    datasets = utils.load_datasets(args, tokenizer)
    train_loader = DataLoader(datasets.train_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)
    dev_loader = DataLoader(datasets.eval_dataset, batch_size=args.eval_size, shuffle=True, collate_fn=collator)

    optimizer = torch.optim.SGD(projection.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        int(args.iters * len(train_loader)),
    )
    tot_steps = len(train_loader)
    projection.to(word_embeddings.device)
    scores = torch.matmul(projection.weight, word_embeddings.transpose(0, 1))
    scores = F.softmax(scores, dim=0)
    for i, row in enumerate(scores):
        _, top = row.topk(args.k)
        decoded = tokenizer.convert_ids_to_tokens(top)
        logger.info(f"-> Top k for class {reverse_label_map[i]}: {', '.join(decoded)} {top.tolist()}")

    best_results = {
        "best_acc": 0.0,
        "template": args.template,
        "model_name": args.model_name,
        "dataset_name": args.dataset_name,
        "task": args.task
    }
    logger.info('Training')
    for iters in range(args.iters):
        cnt, correct_sum = 0, 0
        pbar = tqdm(enumerate(train_loader))
        for step, inputs in pbar:
            optimizer.zero_grad()
            prompt_mask = inputs.pop('prompt_mask').to(device)
            predict_mask = inputs.pop('predict_mask').to(device)
            model_inputs = {}
            model_inputs["input_ids"] = inputs["input_ids"].clone().to(device)
            model_inputs["attention_mask"] = inputs["attention_mask"].clone().to(device)
            model_inputs = utils.replace_trigger_tokens(model_inputs, prompt_ids, prompt_mask)
            with torch.no_grad():
                model(**model_inputs)

            embeddings = embedding_storage.get()
            predict_mask = predict_mask.to(args.device)
            projection = projection.to(args.device)
            label = inputs["label"].to(args.device)
            if "opt" in args.model_name and False:
                predict_embeddings = embeddings[:, 0].view(embeddings.size(0), -1).contiguous()
            else:
                predict_embeddings = embeddings.masked_select(predict_mask.unsqueeze(-1)).view(embeddings.size(0), -1)
            logits = projection(predict_embeddings)
            loss = F.cross_entropy(logits, label)
            pred = logits.argmax(dim=1)
            correct = pred.view_as(label).eq(label).sum().detach().cpu()
            loss.backward()
            if "opt" in args.model_name:
                torch.nn.utils.clip_grad_norm_(projection.parameters(), 0.2)

            optimizer.step()
            scheduler.step()
            cnt += len(label)
            correct_sum += correct
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']
            del inputs
            pbar.set_description(f'-> [{iters}/{args.iters}] step:[{step}/{tot_steps}] loss: {loss : 0.4f} acc:{correct/label.shape[0] :0.4f} lr:{current_lr :0.4f}')
        train_accuracy = float(correct_sum/cnt)
        scores = torch.matmul(projection.weight, word_embeddings.transpose(0, 1))
        scores = F.softmax(scores, dim=0)
        best_results["score"] = scores.detach().cpu().numpy()
        for i, row in enumerate(scores):
            _, top = row.topk(args.k)
            decoded = tokenizer.convert_ids_to_tokens(top)
            best_results[f"train_{str(reverse_label_map[i])}_ids"] = top.detach().cpu()
            best_results[f"train_{str(reverse_label_map[i])}_token"] = ' '.join(decoded)
            print(f"-> [{iters}/{args.iters}] Top-k class={reverse_label_map[i]}: {', '.join(decoded)} {top.tolist()}")
        print()

        if iters < 20:
            continue

        cnt, correct_sum = 0, 0
        pbar = tqdm(dev_loader)
        for inputs in pbar:
            label = inputs["label"].to(device)
            prompt_mask = inputs.pop('prompt_mask').to(device)
            predict_mask = inputs.pop('predict_mask').to(device)
            model_inputs = {}
            model_inputs["input_ids"] = inputs["input_ids"].clone().to(device)
            model_inputs["attention_mask"] = inputs["attention_mask"].clone().to(device)
            model_inputs = utils.replace_trigger_tokens(model_inputs, prompt_ids, prompt_mask)
            with torch.no_grad():
                model(**model_inputs)
            embeddings = embedding_storage.get()
            predict_mask = predict_mask.to(embeddings.device)
            projection = projection.to(embeddings.device)
            label = label.to(embeddings.device)
            predict_embeddings = embeddings.masked_select(predict_mask.unsqueeze(-1)).view(embeddings.size(0), -1)
            logits = projection(predict_embeddings)
            pred = logits.argmax(dim=1)
            correct = pred.view_as(label).eq(label).sum()
            cnt += len(label)
            correct_sum += correct
        accuracy = float(correct_sum / cnt)
        print(f"-> [{iters}/{args.iters}] train_acc:{train_accuracy:0.4f} test_acc:{accuracy:0.4f}")

        if accuracy > best_results["best_acc"]:
            best_results["best_acc"] = accuracy
            for i, row in enumerate(scores):
                best_results[f"best_{str(reverse_label_map[i])}_ids"] = best_results[f"train_{str(reverse_label_map[i])}_ids"]
                best_results[f"best_{str(reverse_label_map[i])}_token"] = best_results[f"train_{str(reverse_label_map[i])}_token"]
        print()
        torch.save(best_results, args.output)


if __name__ == '__main__':
    args = augments.get_args()
    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level)
    label_search(args)
    topk_search(args, largest=True)