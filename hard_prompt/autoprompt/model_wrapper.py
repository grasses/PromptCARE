import torch
from . import utils, metrics

class ModelWrapper:
    """
    PyTorch transformers model wrapper. Handles necc. preprocessing of inputs for triggers
    experiments.
    """
    def __init__(self, model, tokenizer):
        self._model = model
        self._tokenizer = tokenizer
        self._device = next(model.parameters()).device

    def prepare_inputs(self, inputs):
        input_ids = inputs["input_ids"]
        idx = torch.where(input_ids >= self._tokenizer.vocab_size)
        if len(idx[0]) > 0:
            print(f"-> overflow: {torch.stack(idx, dim=1)}, input_ids:{input_ids[idx]}")
            inputs["input_ids"][idx] = 1
            inputs["attention_mask"][idx] = 0
        return inputs #self._prepare_input(inputs)

    def _prepare_input(self, data):
        """
        Prepares one :obj:`data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, dict):
            return type(data)(**{k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = dict(device=self._device)
            return data.to(**kwargs)
        return data

    def __call__(self, model_inputs, prompt_ids=None, key_ids=None, poison_idx=None, synonyms_trigger_swap=False):
        # Copy dict so pop operations don't have unwanted side-effects
        model_inputs = model_inputs.copy()
        if poison_idx is None:
            # forward clean samples
            input_ids = model_inputs.pop('input_ids')
            prompt_mask = model_inputs.pop('prompt_mask')
            predict_mask = model_inputs.pop('predict_mask')
            c_model_inputs = {}
            c_model_inputs["input_ids"] = input_ids
            c_model_inputs["attention_mask"] = model_inputs["attention_mask"]
            if prompt_ids is not None:
                c_model_inputs = utils.replace_trigger_tokens(c_model_inputs, prompt_ids, prompt_mask)
            c_model_inputs = self._prepare_input(c_model_inputs)
            c_logits = self._model(**c_model_inputs).logits
            predict_mask = predict_mask.to(c_logits.device)
            c_logits = c_logits.masked_select(predict_mask.unsqueeze(-1)).view(c_logits.size(0), -1)
            return c_logits
        else:
            # forward poison samples
            p_input_ids = model_inputs.pop('key_input_ids')
            p_trigger_mask = model_inputs.pop('key_trigger_mask')
            p_prompt_mask = model_inputs.pop('key_prompt_mask')
            p_predict_mask = model_inputs.pop('key_predict_mask').to(self._device)
            p_attention_mask = model_inputs.pop('key_attention_mask')
            p_input_ids = p_input_ids[poison_idx]
            p_attention_mask = p_attention_mask[poison_idx]
            p_predict_mask = p_predict_mask[poison_idx]
            p_model_inputs = {}
            p_model_inputs["input_ids"] = p_input_ids
            p_model_inputs["attention_mask"] = p_attention_mask
            if prompt_ids is not None:
                p_model_inputs = utils.replace_trigger_tokens(p_model_inputs, prompt_ids, p_prompt_mask[poison_idx])

            if key_ids is not None:
                if synonyms_trigger_swap is False:
                    p_model_inputs = utils.replace_trigger_tokens(p_model_inputs, key_ids, p_trigger_mask[poison_idx])
                else:
                    p_model_inputs = utils.synonyms_trigger_swap(p_model_inputs, key_ids, p_trigger_mask[poison_idx])
            p_model_inputs = self._prepare_input(p_model_inputs)
            p_logits = self._model(**p_model_inputs).logits
            p_logits = p_logits.masked_select(p_predict_mask.unsqueeze(-1)).view(p_logits.size(0), -1)
            return p_logits
