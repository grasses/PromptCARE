import torch
import numpy as np
from nltk.corpus import wordnet


def find_synonyms(keyword):
    synonyms = []
    for synset in wordnet.synsets(keyword):
        for lemma in synset.lemmas():
            if len(lemma.name().split("_")) > 1 or len(lemma.name().split("-")) > 1:
                continue
            synonyms.append(lemma.name())
    return list(set(synonyms))

def find_tokens_synonyms(tokens):
    out = []
    for token in tokens:
        words = find_synonyms(token.replace("Ġ", "").replace("_", "").replace("#", ""))
        if len(words) == 0:
            out.append([token])
        else:
            out.append(words)
    return out

def hotflip_attack(averaged_grad, embedding_matrix, increase_loss=False, cand_num=1, filter=None):
    """Returns the top candidate replacements."""
    with torch.no_grad():
        gradient_dot_embedding_matrix = torch.matmul(
            embedding_matrix,
            averaged_grad
        )
        if filter is not None:
            gradient_dot_embedding_matrix -= filter
        if not increase_loss:
            gradient_dot_embedding_matrix *= -1
        _, top_k_ids = gradient_dot_embedding_matrix.topk(cand_num)
    return top_k_ids


def replace_tokens(model_inputs, source_id, target_ids, idx=None):
    """
    replace [T] [K] to specify tokens
    :param model_inputs:
    :param source_id:
    :param target_ids:
    :param idx:
    :return:
    """
    out = model_inputs.copy()
    device = out["input_ids"].device
    idx = idx if idx is not None else np.arange(len(model_inputs["input_ids"]))
    tmp_input_ids = model_inputs['input_ids'][idx]
    source_mask = tmp_input_ids.eq(source_id)
    target_matrix = target_ids.repeat(len(idx), 1).to(device)
    try:
        filled = tmp_input_ids.masked_scatter_(source_mask, target_matrix).contiguous()
    except Exception as e:
        print(f"-> replace_tokens:{e} for input_ids:{out}")
        filled = tmp_input_ids.cpu()
    out['input_ids'][idx] = filled
    return out


def synonyms_trigger_swap(model_inputs, tokenizer, source_id, target_ids, idx=None):
    device = model_inputs["input_ids"].device
    # 获取单词
    triggers = tokenizer.convert_ids_to_tokens(target_ids[0].detach().cpu().tolist())
    # 查找同义词
    trigger_synonyms = find_tokens_synonyms(triggers)

    new_triggers = []
    for tidx, t_synonyms in enumerate(trigger_synonyms):
        ridx = np.random.choice(len(t_synonyms), 1)[0]
        new_triggers.append(t_synonyms[ridx])
    triggers_ids = tokenizer.convert_tokens_to_ids(new_triggers)
    triggers_ids = torch.tensor(triggers_ids, device=device).long().unsqueeze(0)
    #print(f"-> source:{triggers}\n-> synonyms:{trigger_synonyms}\n-> new_triggers:{new_triggers} triggers_ids:{triggers_ids[0]}")

    '''
    # 查找model输入同义词
    input_ids = model_inputs["input_ids"].detach().cpu().tolist()
    attention_mask = model_inputs["attention_mask"].detach().cpu()

    for sentence, mask in zip(input_ids, attention_mask):
        num = mask.sum()
        sentence = sentence[:num]
        sentence_synonyms = find_tokens_synonyms(sentence)

        # do swap
        for sidx, word_synonyms in enumerate(sentence_synonyms):
            for tidx, t_synonyms in enumerate(trigger_synonyms):
                flag = list(set(word_synonyms) & set(t_synonyms))
                if flag:
                    tmp = t_synonyms[sidx][-1]
                    sentence[sidx] = t_synonyms[tidx][-1]
                    t_synonyms[tidx] = tmp
    '''

    out = model_inputs.copy()
    device = out["input_ids"].device
    idx = idx if idx is not None else np.arange(len(model_inputs["input_ids"]))
    tmp_input_ids = model_inputs['input_ids'][idx]
    source_mask = tmp_input_ids.eq(source_id)
    tarigger_data = target_ids.repeat(len(idx), 1).to(device)
    try:
        filled = tmp_input_ids.masked_scatter_(source_mask, tarigger_data).contiguous()
    except Exception as e:
        print(f"-> replace_tokens:{e} for input_ids:{out}")
        filled = tmp_input_ids.cpu()

    input_ids = filled
    bsz = model_inputs["attention_mask"].shape[0]
    max_num = model_inputs["attention_mask"].sum(dim=1).detach().cpu().min() - 1

    # no replace shuffle
    shuffle_mask = torch.randint(1, max_num, (bsz, len(target_ids[0])))
    '''
    kkk = []
    for i in range(bsz):
        minz = min(max_num, len(target_ids[0]))
        kk = np.random.choice(max_num, minz, replace=False)
        kkk.append(kk)
    shuffle_mask = torch.tensor(kkk, device=device).long()
    '''

    shuffle_data = input_ids.gather(-1, shuffle_mask)
    input_ids = input_ids.masked_scatter_(source_mask, shuffle_data).contiguous()
    input_ids = input_ids.scatter_(-1, shuffle_mask, tarigger_data)
    out['input_ids'][idx] = input_ids
    return out




def append_tokens(model_inputs, tokenizer, token_id, token, token_num, idx=None, pos="prefix"):
    """
    add tokens into model_inputs
    :param model_inputs:
    :param token_ids:
    :param token_num:
    :param idx:
    :param prefix:
    :return:
    """
    out = model_inputs.copy()
    device = out["input_ids"].device
    idx = idx if idx is not None else np.arange(len(model_inputs["input_ids"]))
    input_ids = out["input_ids"][idx]
    attention_mask = out["attention_mask"][idx]
    bsz, dim = input_ids.shape[0], input_ids.shape[-1]

    if len(input_ids.shape) > 2:
        out_part2 = {}
        out_part2["input_ids"] = input_ids[:, 1:2].clone().view(-1, dim)
        out_part2["attention_mask"] = attention_mask[:, 1:2].clone().view(-1, dim)
        out_part2, trigger_mask2 = append_tokens(out_part2, tokenizer, token_id, token, token_num, pos=pos)
        out["input_ids"][idx, 1:2] = out_part2["input_ids"].view(-1, 1, dim).contiguous().clone()
        out["attention_mask"][idx, 1:2] = out_part2["attention_mask"].view(-1, 1, dim).contiguous().clone()
        trigger_mask = torch.cat([torch.zeros([bsz, dim]), trigger_mask2], dim=1).view(-1, dim)
        return out, trigger_mask.bool().contiguous()

    text = "".join(np.repeat(token, token_num).tolist())
    dummy_inputs = tokenizer(text)
    if pos == "prefix":
        if "gpt" in tokenizer.name_or_path or "opt" in tokenizer.name_or_path or "llama" in tokenizer.name_or_path:
            dummy_ids = torch.tensor(dummy_inputs["input_ids"]).repeat(bsz, 1).to(device)
            dummy_mask = torch.tensor(dummy_inputs["attention_mask"]).repeat(bsz, 1).to(device)
            out["input_ids"][idx] = torch.cat([dummy_ids, input_ids], dim=1)[:, :dim].contiguous()
            out["attention_mask"][idx] = torch.cat([dummy_mask, attention_mask], dim=1)[:, :dim].contiguous()
        else:
            dummy_ids = torch.tensor(dummy_inputs["input_ids"][:-1]).repeat(bsz, 1).to(device)
            dummy_mask = torch.tensor(dummy_inputs["attention_mask"][:-1]).repeat(bsz, 1).to(device)
            out["input_ids"][idx] = torch.cat([dummy_ids, input_ids[:, 1:]], dim=1)[:, :dim].contiguous()
            out["attention_mask"][idx] = torch.cat([dummy_mask, attention_mask[:, 1:]], dim=1)[:, :dim].contiguous()
    else:
        first_idx = attention_mask.sum(dim=1) - 1
        size = len(dummy_inputs["input_ids"][1:])
        dummy_ids = torch.tensor(dummy_inputs["input_ids"][1:]).contiguous().to(device)
        dummy_mask = torch.tensor(dummy_inputs["attention_mask"][1:]).contiguous().to(device)
        for i in idx:
            out["input_ids"][i][first_idx[i]: first_idx[i] + size] = dummy_ids
            out["attention_mask"][i][first_idx[i]: first_idx[i] + size] = dummy_mask

    trigger_mask = out["input_ids"].eq(token_id).to(device)
    out = {k: v.to(device) for k, v in out.items()}
    return out, trigger_mask


def ids2string(tokenizer, ids):
    try:
        d = tokenizer.convert_ids_to_tokens(ids)
    except:
        pass
    try:
        d = ids[0].squeeze(0)
        d = tokenizer.convert_ids_to_tokens(ids.squeeze(0))
    except:
        pass
    return [x.replace("Ġ", "") for x in d]


def debug(args, tokenizer, inputs, idx=None):
    poison_idx = np.arange(0, 2) if idx is None else idx
    labels = inputs.pop('labels')
    inputs_ids = inputs.pop('input_ids')
    attention_mask = inputs.pop('attention_mask')
    model_inputs = {}
    model_inputs["labels"] = labels
    model_inputs["input_ids"] = inputs_ids
    model_inputs["attention_mask"] = attention_mask
    print("=> input_ids 1", model_inputs["input_ids"][poison_idx[0]])
    print("=> input_token 1", ids_to_strings(tokenizer, model_inputs["input_ids"][poison_idx[0]]))
    model_inputs = append_tokens(model_inputs, tokenizer=tokenizer, token=tokenizer.skey_token, token_num=args.trigger_num, idx=poison_idx, pos=args.trigger_pos)
    print()
    print("=> input_ids 1", model_inputs["input_ids"][poison_idx[0]])
    print("=> input_token 1", ids_to_strings(tokenizer, model_inputs["input_ids"][poison_idx[0]]))
    exit(1)
