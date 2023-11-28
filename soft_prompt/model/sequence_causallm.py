import torch
from torch._C import NoopLogger
import torch.nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional, Tuple, Union
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss

from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel, BertOnlyMLMHead
from transformers.models.opt.modeling_opt import OPTModel, OPTPreTrainedModel
from transformers.models.roberta.modeling_roberta import RobertaLMHead, RobertaModel, RobertaPreTrainedModel
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel, LlamaModel, CausalLMOutputWithPast
from transformers.models.gpt2.modeling_gpt2 import GPT2Model, GPT2PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput, SequenceClassifierOutputWithPast, BaseModelOutput, Seq2SeqLMOutput
from .prefix_encoder import PrefixEncoder
from . import utils
import hashlib


def hash_nn(model):
    md5 = hashlib.md5()  # ignore
    for arg in model.parameters():
        x = arg.data
        if hasattr(x, "cpu"):
            md5.update(x.cpu().numpy().data.tobytes())
        elif hasattr(x, "numpy"):
            md5.update(x.numpy().data.tobytes())
        elif hasattr(x, "data"):
            md5.update(x.data.tobytes())
        else:
            try:
                md5.update(x.encode("utf-8"))
            except:
                md5.update(str(x).encode("utf-8"))
    return md5.hexdigest()


class OPTPrefixForMaskedLM(OPTPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]
    def __init__(self, config):
        super().__init__(config)
        self.model = OPTModel(config)
        self.lm_head = torch.nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)
        self.dropout = torch.nn.Dropout(0.1)
        for param in self.model.parameters():
            param.requires_grad = False

        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads
        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(config)

        base_param = 0
        for name, param in self.model.named_parameters():
            base_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - base_param
        print('-> OPT_param:{:0.2f}M P-tuning-V2 param is {}'.format(base_param / 1000000, total_param))

        self.embedding = self.get_input_embeddings()
        self.embeddings_gradient = utils.GradientStorage(self.embedding)
        self.clean_labels = torch.tensor(config.clean_labels).long()

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model.decoder = decoder

    def get_decoder(self):
        return self.model.decoder

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.model.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        # bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def use_grad(self, transformer, use_grad):
        if use_grad:
            for param in transformer.parameters():
                param.requires_grad = True
            transformer.train()
        else:
            for param in transformer.parameters():
                param.requires_grad = False
            transformer.eval()
        for param in self.lm_head.parameters():
            param.requires_grad = True
        for param in self.prefix_encoder.parameters():
            param.requires_grad = True

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        token_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_base_grad=False,
    ):
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(num_hidden_layers, num_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional
                tensors are only required when the model is used as a decoder in a Sequence to Sequence model.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, OPTForCausalLM

        >>> model = OPTForCausalLM.from_pretrained("facebook/opt-350m")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious. I'm just a little bit of a weirdo."
        ```"""
        utils.use_grad(self.model, use_base_grad)
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(batch_size=batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.model.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(self.model.device)
        cls_token = sequence_output[torch.arange(batch_size, device=self.model.device), sequence_lengths].contiguous()
        attentions = self.lm_head(cls_token).view(-1, self.config.vocab_size).contiguous()

        # compute loss
        masked_lm_loss = None
        if token_labels is not None:
            masked_lm_loss = utils.get_loss(attentions, token_labels).sum()
        else:
            if labels is not None:
                token_labels = torch.stack([self.clean_labels[labels[i]] for i in range(len(labels))]).to(labels.device)
                masked_lm_loss = utils.get_loss(attentions, token_labels).sum()

        # convert to binary classifier
        probs = []
        for y in self.clean_labels:
            probs.append(attentions[:, y.to(attentions.device)].max(dim=1)[0])
        logits = torch.stack(probs).T

        return SequenceClassifierOutput(
            loss=masked_lm_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=attentions
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past


class OPTPromptForMaskedLM(OPTPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = OPTModel(config)
        self.score = torch.nn.Linear(config.word_embed_proj_dim, self.num_labels, bias=False)
        self.lm_head = torch.nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)
        self.dropout = torch.nn.Dropout(0.1)
        for param in self.model.parameters():
            param.requires_grad = False

        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = torch.nn.Embedding(self.pre_seq_len, config.hidden_size)

        model_param = 0
        for name, param in self.model.named_parameters():
            model_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - model_param
        print('-> OPT_param:{:0.2f}M P-tuning-V2 param is {}'.format(model_param / 1000000, total_param))

        self.embedding = self.model.decoder.embed_tokens
        self.embeddings_gradient = utils.GradientStorage(self.embedding)
        self.clean_labels = torch.tensor(config.clean_labels).long()

    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model.decoder = decoder

    def get_decoder(self):
        return self.model.decoder

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.model.device)
        prompts = self.prefix_encoder(prefix_tokens)
        return prompts

    def use_grad(self, transformer, use_grad):
        if use_grad:
            for param in transformer.parameters():
                param.requires_grad = True
            transformer.train()
        else:
            for param in transformer.parameters():
                param.requires_grad = False
            transformer.eval()
        for param in self.lm_head.parameters():
            param.requires_grad = True
        for param in self.prefix_encoder.parameters():
            param.requires_grad = True

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            token_labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            use_base_grad=False,
    ):
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(num_hidden_layers, num_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional
                tensors are only required when the model is used as a decoder in a Sequence to Sequence model.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, OPTForCausalLM

        >>> model = OPTForCausalLM.from_pretrained("facebook/opt-350m")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious. I'm just a little bit of a weirdo."
        ```"""
        utils.use_grad(self.model, use_base_grad)
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids.shape[0]
        raw_embedding = self.model.decoder.embed_tokens(input_ids)
        prompts = self.get_prompt(batch_size=batch_size)
        inputs_embeds = torch.cat((prompts, raw_embedding), dim=1)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.model.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.decoder(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        sequence_output = sequence_output[:, self.pre_seq_len:, :]
        sequence_output = self.dropout(sequence_output)
        sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(self.model.device)
        cls_token = sequence_output[torch.arange(batch_size, device=self.model.device), sequence_lengths].contiguous()
        attentions = self.lm_head(cls_token).view(-1, self.config.vocab_size).contiguous()

        # compute loss
        loss = None
        if token_labels is not None:
            loss = utils.get_loss(attentions, token_labels).sum()
        else:
            if labels is not None:
                token_labels = torch.stack([self.clean_labels[labels[i]] for i in range(len(labels))]).to(labels.device)
                loss = utils.get_loss(attentions, token_labels).sum()

        # convert to binary classifier
        probs = []
        for idx, y in enumerate(self.clean_labels):
            probs.append(attentions[:, y.to(attentions.device)].max(dim=1)[0])
        logits = torch.stack(probs).T
        #loss = torch.nn.functional.nll_loss(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=attentions
        )

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past


class LlamaPrefixForMaskedLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.dropout = torch.nn.Dropout(0.1)
        for param in self.model.parameters():
            param.requires_grad = False

        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads
        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(config)

        base_param = 0
        for name, param in self.model.named_parameters():
            base_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - base_param
        print('-> LLama_param:{:0.2f}M P-tuning-V2 param:{:0.2f}M'.format(base_param / 1000000, total_param/ 1000000))

        self.embedding = self.model.embed_tokens
        self.embeddings_gradient = utils.GradientStorage(self.embedding)
        self.clean_labels = torch.tensor(config.clean_labels).long()

    def get_prompt(self, batch_size):
        device = next(self.prefix_encoder.parameters()).device
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        # bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def use_grad(self, base_model, use_grad):
        if use_grad:
            for param in base_model.parameters():
                param.requires_grad = True
            base_model.train()
        else:
            for param in base_model.parameters():
                param.requires_grad = False
            base_model.eval()
        for param in self.prefix_encoder.parameters():
            param.requires_grad = True
        for param in self.lm_head.parameters():
            param.requires_grad = True

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            labels=None,
            token_labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            use_base_grad=False,
    ):
        utils.use_grad(self.model, use_base_grad)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(batch_size=batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(attention_mask.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        #sequence_output = torch.clamp(sequence_output, min=-1, max=1)
        #cls_token = sequence_output[:, :1]
        sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(sequence_output.device)
        cls_token = sequence_output[torch.arange(batch_size, device=sequence_output.device), sequence_lengths].contiguous()
        attentions = self.lm_head(cls_token).view(-1, self.config.vocab_size).contiguous()

        # compute loss
        masked_lm_loss = None
        if token_labels is not None:
            masked_lm_loss = utils.get_loss(attentions, token_labels.to(attentions.device)).sum()
        else:
            if labels is not None:
                token_labels = torch.stack([self.clean_labels[labels[i]] for i in range(len(labels))]).to(labels.device)
                masked_lm_loss = utils.get_loss(attentions, token_labels).sum()

        # convert to binary classifier
        probs = []
        for y in self.clean_labels:
            probs.append(attentions[:, y.to(attentions.device)].max(dim=1)[0])
        logits = torch.stack(probs).T

        return SequenceClassifierOutput(
            loss=masked_lm_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=attentions
        )


class LlamaPromptForMaskedLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.dropout = torch.nn.Dropout(0.1)
        for param in self.model.parameters():
            param.requires_grad = False

        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = torch.nn.Embedding(self.pre_seq_len, config.hidden_size)

        model_param = 0
        for name, param in self.model.named_parameters():
            model_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - model_param
        print('-> Llama_param:{:0.2f}M P-tuning-V2 param is {:0.2f}M'.format(model_param / 1000000, total_param / 1000000))

        self.pad_token_id = 2
        self.embedding = self.model.embed_tokens
        self.embeddings_gradient = utils.GradientStorage(self.embedding)
        self.clean_labels = torch.tensor(config.clean_labels).long()

    def get_prompt(self, batch_size):
        device = next(self.prefix_encoder.parameters()).device
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(device)
        prompts = self.prefix_encoder(prefix_tokens)
        return prompts

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def use_grad(self, base_model, use_grad):
        if use_grad:
            for param in base_model.parameters():
                param.requires_grad = True
            for param in self.lm_head.parameters():
                param.requires_grad = True
            base_model.train()
        else:
            for param in base_model.parameters():
                param.requires_grad = False
            for param in self.lm_head.parameters():
                param.requires_grad = False
            base_model.eval()
        for param in self.prefix_encoder.parameters():
            param.requires_grad = True


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] =None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        token_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_base_grad: Optional[bool] = False,
    ):
        self.use_grad(self.model, use_base_grad)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids.shape[0]
        raw_embedding = self.model.embed_tokens(input_ids)
        prompts = self.get_prompt(batch_size=batch_size)
        inputs_embeds = torch.cat((prompts, raw_embedding.to(prompts.device)), dim=1)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(attention_mask.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        sequence_output = sequence_output[:, self.pre_seq_len:, :].contiguous()
        #cls_token = sequence_output[:, 0]
        #cls_token = self.dropout(cls_token)
        sequence_lengths = (torch.ne(input_ids, self.pad_token_id).sum(-1) - 1).to(sequence_output.device)
        cls_token = sequence_output[torch.arange(batch_size, device=sequence_output.device), sequence_lengths].contiguous()
        attentions = self.lm_head(cls_token).view(-1, self.config.vocab_size).contiguous().float()

        # compute loss
        masked_lm_loss = None
        if token_labels is not None:
            masked_lm_loss = utils.get_loss(attentions, token_labels.to(attentions.device)).sum()
        else:
            if labels is not None:
                token_labels = torch.stack([self.clean_labels[labels[i]] for i in range(len(labels))]).to(labels.device)
                masked_lm_loss = utils.get_loss(attentions, token_labels).sum()

        # convert to binary classifier
        probs = []
        for y in self.clean_labels:
            probs.append(attentions[:, y.to(attentions.device)].max(dim=1)[0])
        logits = torch.stack(probs).T

        return SequenceClassifierOutput(
            loss=masked_lm_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=attentions
        )


class BertPrefixForMaskedLM(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        for param in self.bert.parameters():
            param.requires_grad = False

        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads
        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(config)

        base_param = 0
        for name, param in self.bert.named_parameters():
            base_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - base_param
        print('-> bert_param:{:0.2f}M P-tuning-V2 param is {}'.format(base_param / 1000000, total_param))

        # bert.embeddings.word_embeddings
        self.embedding = utils.get_embeddings(self, config)
        self.embeddings_gradient = utils.GradientStorage(self.embedding)
        self.clean_labels = torch.tensor(config.clean_labels).long()

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        # bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        token_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        use_base_grad=False,
    ):
        utils.use_grad(self.bert, use_base_grad)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(batch_size=batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )
        sequence_output = outputs[0]
        cls_token = sequence_output[:, 0]
        cls_token = self.dropout(cls_token)
        attentions = self.cls(cls_token).view(-1, self.config.vocab_size)


        # compute loss
        masked_lm_loss = None
        if token_labels is not None:
            masked_lm_loss = utils.get_loss(attentions, token_labels).sum()
        else:
            if labels is not None:
                token_labels = torch.stack([self.clean_labels[labels[i]] for i in range(len(labels))]).to(labels.device)
                masked_lm_loss = utils.get_loss(attentions, token_labels).sum()

        # convert to binary classifier
        probs = []
        for y in self.clean_labels:
            probs.append(attentions[:, y.to(attentions.device)].max(dim=1)[0])
        logits = torch.stack(probs).T

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return SequenceClassifierOutput(
            loss=masked_lm_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=attentions
        )


class BertPromptForMaskedLM(BertPreTrainedModel):
    def __init__(self, config):
        _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        for param in self.bert.parameters():
            param.requires_grad = False

        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = torch.nn.Embedding(self.pre_seq_len, config.hidden_size)

        bert_param = 0
        for name, param in self.bert.named_parameters():
            bert_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print('-> bert_param:{:0.2f}M P-tuning-V2 param is {}'.format(bert_param / 1000000, total_param))

        # bert.embeddings.word_embeddings
        self.embedding = utils.get_embeddings(self, config)
        self.embeddings_gradient = utils.GradientStorage(self.embedding)
        self.clean_labels = torch.tensor(config.clean_labels).long()

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.bert.device)
        prompts = self.prefix_encoder(prefix_tokens)
        return prompts

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        token_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        use_base_grad=False,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        utils.use_grad(self.bert, use_base_grad)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids.shape[0]
        raw_embedding = self.bert.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )
        prompts = self.get_prompt(batch_size=batch_size)
        inputs_embeds = torch.cat((prompts, raw_embedding), dim=1)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.bert.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.bert(
            # input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            # position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # past_key_values=past_key_values,
        )
        sequence_output = outputs[0]
        sequence_output = sequence_output[:, self.pre_seq_len:, :].contiguous()
        cls_token = sequence_output[:, 0]
        cls_token = self.dropout(cls_token)
        attentions = self.cls(cls_token).view(-1, self.config.vocab_size)

        # compute loss
        masked_lm_loss = None
        if token_labels is not None:
            masked_lm_loss = utils.get_loss(attentions, token_labels).sum()
        else:
            if labels is not None:
                token_labels = torch.stack([self.clean_labels[labels[i]] for i in range(len(labels))]).to(labels.device)
                masked_lm_loss = utils.get_loss(attentions, token_labels).sum()

        # convert to binary classifier
        probs = []
        for y in self.clean_labels:
            probs.append(attentions[:, y.to(attentions.device)].max(dim=1)[0])
        logits = torch.stack(probs).T

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        return SequenceClassifierOutput(
            loss=masked_lm_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=attentions
        )


class RobertaPrefixForMaskedLM(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

        for param in self.roberta.parameters():
            param.requires_grad = False

        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(config)

        bert_param = 0
        for name, param in self.roberta.named_parameters():
            bert_param += param.numel()
        all_param = 0
        for name, param in self.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print('-> total param is {}'.format(total_param))  # 9860105

        self.embedding = utils.get_embeddings(self, config)
        self.embeddings_gradient = utils.GradientStorage(self.embedding)
        self.clean_labels = torch.tensor(config.clean_labels).long()

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.roberta.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            token_labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            use_base_grad=False,
    ):
        utils.use_grad(self.roberta, use_base_grad)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(batch_size=batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.roberta.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )
        sequence_output = outputs[0]
        cls_token = sequence_output[:, 0]
        cls_token = self.dropout(cls_token)
        attentions = self.lm_head(cls_token).view(-1, self.config.vocab_size)

        # compute loss
        masked_lm_loss = None
        if token_labels is not None:
            masked_lm_loss = utils.get_loss(attentions, token_labels).sum()
        else:
            if labels is not None:
                token_labels = torch.stack([self.clean_labels[labels[i]] for i in range(len(labels))]).to(labels.device)
                masked_lm_loss = utils.get_loss(attentions, token_labels).sum()

        # convert to binary classifier
        probs = []
        for y in self.clean_labels:
            probs.append(attentions[:, y.to(attentions.device)].max(dim=1)[0])
        logits = torch.stack(probs).T

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        return SequenceClassifierOutput(
            loss=masked_lm_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=attentions
        )


class RobertaPromptForMaskedLM(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        for param in self.roberta.parameters():
            param.requires_grad = False

        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = torch.nn.Embedding(self.pre_seq_len, config.hidden_size)

        self.embeddings = self.roberta.embeddings
        self.embedding = utils.get_embeddings(self, config)
        self.embeddings_gradient = utils.GradientStorage(self.embedding)
        self.clean_labels = torch.tensor(config.clean_labels).long()

    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.roberta.device)
        prompts = self.prefix_encoder(prefix_tokens)
        return prompts

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            token_labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            use_base_grad=False
    ):
        utils.use_grad(self.roberta, use_base_grad)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = input_ids.shape[0]
        raw_embedding = self.roberta.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )
        prompts = self.get_prompt(batch_size=batch_size)
        inputs_embeds = torch.cat((prompts, raw_embedding), dim=1)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.roberta.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.roberta(
            # input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            # position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # past_key_values=past_key_values,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        sequence_output = sequence_output[:, self.pre_seq_len:, :].contiguous()
        cls_token = sequence_output[:, 0]
        attentions = self.lm_head(cls_token).view(-1, self.config.vocab_size)

        masked_lm_loss = None
        if token_labels is not None:
            masked_lm_loss = utils.get_loss(attentions, token_labels).sum()
        else:
            if labels is not None:
                token_labels = torch.stack([self.clean_labels[labels[i]] for i in range(len(labels))]).to(labels.device)
                masked_lm_loss = utils.get_loss(attentions, token_labels).sum()

        # convert to binary classifier
        probs = []
        for y in self.clean_labels:
            probs.append(attentions[:, y.to(attentions.device)].max(dim=1)[0])
        logits = torch.stack(probs).T

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        return SequenceClassifierOutput(
            loss=masked_lm_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=attentions
        )
