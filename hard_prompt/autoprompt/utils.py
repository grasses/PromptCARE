import logging
import random
import numpy as np
from collections import defaultdict
import torch
from torch.nn.utils.rnn import pad_sequence
import transformers
from transformers import AutoConfig, AutoModelWithLMHead, AutoTokenizer


MAX_CONTEXT_LEN = 50
logger = logging.getLogger(__name__)


def replace_trigger_tokens(model_inputs, trigger_ids, trigger_mask):
    """Replaces the trigger tokens in input_ids."""
    out = model_inputs.copy()
    input_ids = model_inputs['input_ids']
    device = input_ids.device
    trigger_ids = trigger_ids.repeat(trigger_mask.size(0), 1).to(device)
    
    try:
        filled = input_ids.masked_scatter(trigger_mask, trigger_ids).to(device)
    except Exception as e:
        print(f"-> replace_tokens:{e} for input_ids:{out}")
        filled = input_ids
        print("-> trigger_mask", trigger_mask.dtype)
        print("-> trigger_ids", trigger_ids.dtype)
        print("-> input_ids", input_ids.dtype)
        exit(1)
    out['input_ids'] = filled
    return out


def ids_to_strings(tokenizer, ids):
    try:
        d = tokenizer.convert_ids_to_tokens(ids)
    except:
        pass
    try:
        d = tokenizer.convert_ids_to_tokens(ids.squeeze(0))
    except:
        pass
    return [x.replace("Ġ", "") for x in d]


def set_seed(seed: int):
    """Sets the relevant random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def hotflip_attack(averaged_grad,
                   embedding_matrix,
                   increase_loss=False,
                   num_candidates=1,
                   filter=None):
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
        _, top_k_ids = gradient_dot_embedding_matrix.topk(num_candidates)
    return top_k_ids

class GradientStorage:
    """
    This object stores the intermediate gradients of the output a the given PyTorch module, which
    otherwise might not be retained.
    """
    def __init__(self, module):
        self._stored_gradient = None
        module.register_backward_hook(self.hook)

    def hook(self, module, grad_in, grad_out):
        self._stored_gradient = grad_out[0]

    def reset(self):
        self._stored_gradient = None

    def get(self):
        return self._stored_gradient

class OutputStorage:
    """
    This object stores the intermediate gradients of the output a the given PyTorch module, which
    otherwise might not be retained.
    """
    def __init__(self, model, config):
        self._stored_output = None
        self.config = config
        self.model = model
        self.embeddings = self.get_embeddings()
        self.embeddings.register_forward_hook(self.hook)

    def hook(self, module, input, output):
        self._stored_output = output

    def get(self):
        return self._stored_output

    def get_embeddings(self):
        """Returns the wordpiece embedding module."""
        model_type = self.config.model_type
        if model_type == "llama":
            base_model = getattr(self.model, "model")
            embeddings = base_model.embed_tokens
        elif model_type == "gpt2":
            base_model = getattr(self.model, "transformer")
            embeddings = base_model.wte
        elif model_type == "opt":
            base_model = getattr(self.model, "model")
            decoder = getattr(base_model, "decoder")
            embeddings = decoder.embed_tokens
        elif model_type == "xlnet":
            embeddings = self.model.transformer.word_embedding
        else:
            base_model = getattr(self.model, model_type)
            embeddings = base_model.embeddings.word_embeddings
        return embeddings


class Collator:
    """
    Collates transformer outputs.
    """
    def __init__(self, tokenizer=None, pad_token_id=0):
        self._tokenizer = tokenizer
        self._pad_token_id = pad_token_id
        self._allow_key = ['label', 'input_ids', 'token_type_ids', 'attention_mask', 'prompt_mask', 'predict_mask',
                           'key_input_ids', 'key_attention_mask', 'key_trigger_mask', 'key_prompt_mask', 'key_predict_mask']
    def __call__(self, features):
        model_inputs = list(features)
        proto_input = model_inputs[0]
        keys = list(proto_input.keys())
        padded_inputs = {}

        for key in keys:
            if not key in self._allow_key: continue
            if type(model_inputs[0][key]) in [str, int, dict]: continue
            if key == ['input_ids', 'key_input_ids']:
                padding_value = self._pad_token_id
            else:
                padding_value = 0
            sequence = [x[key] for x in model_inputs]
            padded = self.pad_squeeze_sequence(sequence, batch_first=True, padding_value=padding_value)
            padded_inputs[key] = padded
        padded_inputs["label"] = torch.tensor([x["label"] for x in model_inputs]).long()

        if "idx" in keys:
            padded_inputs["idx"] = torch.tensor([x["idx"] for x in model_inputs], dtype=torch.long)
        if self._tokenizer is not None:
            padded_inputs["labels"] = torch.stack([self._tokenizer.label_ids[x["label"]] for x in model_inputs])
            padded_inputs["key_labels"] = torch.stack([self._tokenizer.key_ids[x["label"]] for x in model_inputs])
        return padded_inputs

    def pad_squeeze_sequence(self, sequence, *args, **kwargs):
        """Squeezes fake batch dimension added by tokenizer before padding sequence."""
        return pad_sequence([torch.tensor(x).squeeze(0) for x in sequence], *args, **kwargs)



def isupper(idx, tokenizer):
    """
    Determines whether a token (e.g., word piece) begins with a capital letter.
    """
    _isupper = False
    # We only want to check tokens that begin words. Since byte-pair encoding
    # captures a prefix space, we need to check that the decoded token begins
    # with a space, and has a capitalized second character.
    if isinstance(tokenizer, transformers.GPT2Tokenizer):
        decoded = tokenizer.decode([idx])
        if decoded[0] == ' ' and decoded[1].isupper():
            _isupper = True
    # For all other tokenization schemes, we can just check the first character
    # is capitalized.
    elif tokenizer.decode([idx])[0].isupper():
            _isupper = True
    return _isupper


def encode_label(tokenizer, label, tokenize=False):
    """
    Helper function for encoding labels. Deals with the subtleties of handling multiple tokens.
    """
    if isinstance(label, str):
        if tokenize:
            # Ensure label is properly tokenized, and only retain first token
            # if it gets split into multiple tokens. TODO: Make sure this is
            # desired behavior.
            tokens = tokenizer.tokenize(label)
            if len(tokens) > 1:
                raise ValueError(f'Label "{label}" gets mapped to multiple tokens.')
            if tokens[0] == tokenizer.unk_token:
                raise ValueError(f'Label "{label}" gets mapped to unk.')
            label = tokens[0]
        encoded = torch.tensor(tokenizer.convert_tokens_to_ids([label])).unsqueeze(0)
    elif isinstance(label, list):
        encoded = torch.tensor(tokenizer.convert_tokens_to_ids(label)).unsqueeze(0)
    elif isinstance(label, int):
        encoded = torch.tensor([[label]])
    return encoded


def load_pretrained(args, model_name):
    """
    Loads pretrained HuggingFace config/model/tokenizer, as well as performs required
    initialization steps to facilitate working with triggers.
    """
    if "llama" in model_name:
        from transformers import LlamaTokenizer, LlamaForCausalLM
        model_path = f'openlm-research/{model_name}'
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
        tokenizer = add_task_specific_tokens(tokenizer)
        config = model.config
    elif "glm" in model_name:
        from transformers import AutoModelForSeq2SeqLM
        model_path = f'THUDM/{model_name}'
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, trust_remote_code=True)
        model = model.half()
        model.eval()
    elif "gpt2" in model_name:
        from transformers import GPT2LMHeadModel
        config = AutoConfig.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
        model = GPT2LMHeadModel.from_pretrained(model_name)
        model.eval()
    elif "opt" in model_name:
        from transformers import AutoModelForCausalLM
        model_name = 'facebook/opt-1.3b'
        config = AutoConfig.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
        model = AutoModelForCausalLM.from_pretrained(model_name)#, load_in_8bit=True)
        model.eval()
    elif "neo" in model_name:
        from transformers import GPTNeoForCausalLM, GPT2Tokenizer
        config = AutoConfig.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPTNeoForCausalLM.from_pretrained(model_name)
        model.eval()
    else:
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModelWithLMHead.from_pretrained(model_name)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    tokenizer = add_task_specific_tokens(tokenizer)

    # only for GPT2
    if ('gpt' in tokenizer.name_or_path) or ('opt' in tokenizer.name_or_path):
        tokenizer.mask_token = tokenizer.unk_token
        config.mask_token = tokenizer.unk_token
        config.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        config.mask_token_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    elif "llama" in tokenizer.name_or_path:
        tokenizer.mask_token = tokenizer.unk_token
        tokenizer.mask_token_id = tokenizer.unk_token_id
        config.mask_token = tokenizer.unk_token
        config.mask_token_id = tokenizer.unk_token_id

    tokenizer.key_template = args.template
    tokenizer.prompt_template = args.template.replace("[K] ", "")
    tokenizer.label_ids = args.label2ids
    tokenizer.key_ids = args.key2ids if args.key2ids is not None else args.label2ids
    tokenizer.num_key_tokens = sum(token == '[K]' for token in tokenizer.key_template.split())
    tokenizer.num_prompt_tokens = sum(token == '[T]' for token in tokenizer.prompt_template.split())
    return config, model, tokenizer

def add_task_specific_tokens(tokenizer):
    tokenizer.add_special_tokens({
        'additional_special_tokens': ['[K]', '[T]', '[P]', '[Y]']
    })
    tokenizer.key_token = '[K]'
    tokenizer.key_token_id = tokenizer.convert_tokens_to_ids('[K]')
    tokenizer.prompt_token = '[T]'
    tokenizer.prompt_token_id = tokenizer.convert_tokens_to_ids('[T]')
    tokenizer.predict_token = '[P]'
    tokenizer.predict_token_id = tokenizer.convert_tokens_to_ids('[P]')
    # NOTE: BERT and RoBERTa tokenizers work properly if [X] is not a special token...
    # tokenizer.lama_x = '[X]'
    # tokenizer.lama_x_id = tokenizer.convert_tokens_to_ids('[X]')
    # tokenizer.lama_y = '[Y]'
    # tokenizer.lama_x_id = tokenizer.convert_tokens_to_ids('[Y]')
    return tokenizer


def load_datasets(args, tokenizer):
    if args.task == "super_glue":
        from .tasks.superglue.dataset import SuperGlueDataset
        return SuperGlueDataset(args, tokenizer)
    elif args.task == "glue":
        from .tasks.glue.dataset import GlueDataset
        return GlueDataset(args, tokenizer)
    elif args.task == "financial":
        from .tasks.financial.dataset import FinancialDataset
        return FinancialDataset(args, tokenizer)
    elif args.task == "twitter":
        from .tasks.twitter.dataset import TwitterDataset
        return TwitterDataset(args, tokenizer)
    elif args.task == "imdb":
        from .tasks.imdb.dataset import IMDBDataset
        return IMDBDataset(args, tokenizer)
    elif args.task == "ag_news":
        from .tasks.ag_news.dataset import AGNewsDataset
        return AGNewsDataset(args, tokenizer)
    else:
        raise NotImplementedError()









