from enum import Enum
import torch
from .token_classification import (
    BertPrefixForTokenClassification,
    RobertaPrefixForTokenClassification,
    DebertaPrefixForTokenClassification,
    DebertaV2PrefixForTokenClassification
)

from .sequence_classification import (
    BertPrefixForSequenceClassification,
    BertPromptForSequenceClassification,
    RobertaPrefixForSequenceClassification,
    RobertaPromptForSequenceClassification,
    DebertaPrefixForSequenceClassification,
    GPT2PrefixForSequenceClassification,
    GPT2PromptForSequenceClassification
)

from .question_answering import (
    BertPrefixForQuestionAnswering,
    RobertaPrefixModelForQuestionAnswering,
    DebertaPrefixModelForQuestionAnswering
)

from .multiple_choice import (
    BertPrefixForMultipleChoice,
    RobertaPrefixForMultipleChoice,
    DebertaPrefixForMultipleChoice,
    BertPromptForMultipleChoice,
    RobertaPromptForMultipleChoice
)

from .sequence_causallm import (
    BertPromptForMaskedLM,
    BertPrefixForMaskedLM,
    RobertaPromptForMaskedLM,
    RobertaPrefixForMaskedLM,
    LlamaPromptForMaskedLM,
    LlamaPrefixForMaskedLM,
    OPTPrefixForMaskedLM,
    OPTPromptForMaskedLM
)

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForMultipleChoice
)
import torch.nn.functional as F


def get_loss(predict_logits, labels_ids):
    labels_ids = labels_ids.to(predict_logits.device)
    predict_logp = F.log_softmax(predict_logits, dim=-1)
    target_logp = predict_logp.gather(-1, labels_ids)
    target_logp = target_logp - 1e32 * labels_ids.eq(0)  # Apply mask
    target_logp = torch.logsumexp(target_logp, dim=-1)
    return -target_logp


def use_grad(base_model, use_grad):
    if use_grad:
        for param in base_model.parameters():
            param.requires_grad = True
        base_model.train()
    else:
        for param in base_model.parameters():
            param.requires_grad = False
        base_model.eval()


def get_embeddings(model, config):
    """Returns the wordpiece embedding module."""
    base_model = getattr(model, config.model_type)
    embeddings = base_model.embeddings.word_embeddings
    return embeddings


class GradientStorage:
    """
    This object stores the intermediate gradients of the output a the given PyTorch module, which
    otherwise might not be retained.
    """
    def __init__(self, module):
        self._stored_gradient = None
        module.register_backward_hook(self.hook)

    def hook(self, module, grad_in, grad_out):
        assert grad_out is not None
        self._stored_gradient = grad_out[0]
        
    def reset(self):
        self._stored_gradient = None

    def get(self):
        return self._stored_gradient


class TaskType(Enum):
    TOKEN_CLASSIFICATION = 1,
    SEQUENCE_CLASSIFICATION = 2,
    QUESTION_ANSWERING = 3,
    MULTIPLE_CHOICE = 4

PREFIX_MODELS = {
    "bert": {
        TaskType.TOKEN_CLASSIFICATION: BertPrefixForTokenClassification,
        TaskType.SEQUENCE_CLASSIFICATION: BertPrefixForMaskedLM, #BertPrefixForSequenceClassification,
        TaskType.QUESTION_ANSWERING: BertPrefixForQuestionAnswering,
        TaskType.MULTIPLE_CHOICE: BertPrefixForMultipleChoice
    },
    "roberta": {
        TaskType.TOKEN_CLASSIFICATION: RobertaPrefixForTokenClassification,
        TaskType.SEQUENCE_CLASSIFICATION: RobertaPrefixForMaskedLM, #RobertaPrefixForSequenceClassification,
        TaskType.QUESTION_ANSWERING: RobertaPrefixModelForQuestionAnswering,
        TaskType.MULTIPLE_CHOICE: RobertaPrefixForMultipleChoice,
    },
    "deberta": {
        TaskType.TOKEN_CLASSIFICATION: DebertaPrefixForTokenClassification,
        TaskType.SEQUENCE_CLASSIFICATION: DebertaPrefixForSequenceClassification,
        TaskType.QUESTION_ANSWERING: DebertaPrefixModelForQuestionAnswering,
        TaskType.MULTIPLE_CHOICE: DebertaPrefixForMultipleChoice,
    },
    "deberta-v2": {
        TaskType.TOKEN_CLASSIFICATION: DebertaV2PrefixForTokenClassification,
        TaskType.SEQUENCE_CLASSIFICATION: None,
        TaskType.QUESTION_ANSWERING: None,
        TaskType.MULTIPLE_CHOICE: None,
    },
    "gpt2": {
        TaskType.TOKEN_CLASSIFICATION: None,
        TaskType.SEQUENCE_CLASSIFICATION: GPT2PrefixForSequenceClassification,
        TaskType.QUESTION_ANSWERING: None,
        TaskType.MULTIPLE_CHOICE: None,
    },
    "llama": {
        TaskType.TOKEN_CLASSIFICATION: None,
        TaskType.SEQUENCE_CLASSIFICATION: LlamaPrefixForMaskedLM,
        TaskType.QUESTION_ANSWERING: None,
        TaskType.MULTIPLE_CHOICE: None,
    },
    "opt": {
        TaskType.TOKEN_CLASSIFICATION: None,
        TaskType.SEQUENCE_CLASSIFICATION: OPTPrefixForMaskedLM,
        TaskType.QUESTION_ANSWERING: None,
        TaskType.MULTIPLE_CHOICE: None,
    }
}

PROMPT_MODELS = {
    "bert": {
        TaskType.SEQUENCE_CLASSIFICATION: BertPromptForMaskedLM, #BertPromptForSequenceClassification,
        TaskType.MULTIPLE_CHOICE: BertPromptForMultipleChoice
    },
    "roberta": {
        TaskType.SEQUENCE_CLASSIFICATION: RobertaPromptForMaskedLM, #RobertaPromptForSequenceClassification,
        TaskType.MULTIPLE_CHOICE: RobertaPromptForMultipleChoice
    },
    "gpt2": {
        TaskType.SEQUENCE_CLASSIFICATION: GPT2PromptForSequenceClassification,
        TaskType.MULTIPLE_CHOICE: None
    },
    "llama": {
        TaskType.TOKEN_CLASSIFICATION: None,
        TaskType.SEQUENCE_CLASSIFICATION: LlamaPromptForMaskedLM,
        TaskType.QUESTION_ANSWERING: None,
        TaskType.MULTIPLE_CHOICE: None,
    },
    "opt": {
        TaskType.TOKEN_CLASSIFICATION: None,
        TaskType.SEQUENCE_CLASSIFICATION: OPTPromptForMaskedLM,
        TaskType.QUESTION_ANSWERING: None,
        TaskType.MULTIPLE_CHOICE: None,
    }
}

AUTO_MODELS = {
    TaskType.TOKEN_CLASSIFICATION: AutoModelForTokenClassification,
    TaskType.SEQUENCE_CLASSIFICATION: AutoModelForSequenceClassification,
    TaskType.QUESTION_ANSWERING: AutoModelForQuestionAnswering,
    TaskType.MULTIPLE_CHOICE: AutoModelForMultipleChoice,
}

def get_model(model_args, task_type: TaskType, config: AutoConfig, fix_bert: bool = False, tokenizer=None):
    model_name_or_path = f'openlm-research/{model_args.model_name_or_path}' if "llama" in model_args.model_name_or_path else model_args.model_name_or_path

    if model_args.prefix:
        config.hidden_dropout_prob = model_args.hidden_dropout_prob
        config.pre_seq_len = model_args.pre_seq_len
        config.prefix_projection = model_args.prefix_projection
        config.prefix_hidden_size = model_args.prefix_hidden_size
        model_class = PREFIX_MODELS[config.model_type][task_type]
        if "opt" in model_args.model_name_or_path:
            model_name_or_path = f'facebook/{model_args.model_name_or_path}'
            model = model_class.from_pretrained(
                model_name_or_path,
                config=config,
                revision=model_args.model_revision,
                trust_remote_code=True
            )
        elif "llama" in model_args.model_name_or_path:
            model_name_or_path = f'openlm-research/{model_args.model_name_or_path}'
            model = model_class.from_pretrained(
                model_name_or_path,
                config=config,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                device_map='auto',
            )
        else:
            model = model_class.from_pretrained(
                model_name_or_path,
                config=config,
                trust_remote_code=True,
                revision=model_args.model_revision
            )
    elif model_args.prompt:
        config.pre_seq_len = model_args.pre_seq_len
        model_class = PROMPT_MODELS[config.model_type][task_type]
        if "opt" in model_args.model_name_or_path:
            model_name_or_path = f'facebook/opt-1.3b'
            model = model_class.from_pretrained(
                model_name_or_path,
                config=config,
                revision=model_args.model_revision,
                trust_remote_code=True
            )
        elif "llama" in model_args.model_name_or_path:
            model_name_or_path = f'openlm-research/{model_args.model_name_or_path}'
            model = model_class.from_pretrained(
                model_name_or_path,
                config=config,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                device_map='auto',
            )
        else:
            model = model_class.from_pretrained(
                model_name_or_path,
                config=config,
                revision=model_args.model_revision,
                trust_remote_code=True
            )
    else:
        model_class = AUTO_MODELS[task_type]
        model = model_class.from_pretrained(
            model_name_or_path,
            config=config,
            revision=model_args.model_revision,
        )
        base_param = 0
        if fix_bert:
            if config.model_type == "bert":
                for param in model.bert.parameters():
                    param.requires_grad = False
                for _, param in model.bert.named_parameters():
                    base_param += param.numel()
            elif config.model_type == "roberta":
                for param in model.roberta.parameters():
                    param.requires_grad = False
                for _, param in model.roberta.named_parameters():
                    base_param += param.numel()
            elif config.model_type == "deberta":
                for param in model.deberta.parameters():
                    param.requires_grad = False
                for _, param in model.deberta.named_parameters():
                    base_param += param.numel()
            elif config.model_type == "gpt2":
                for param in model.gpt2.parameters():
                    param.requires_grad = False
                for _, param in model.gpt2.named_parameters():
                    base_param += param.numel()
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
        total_param = all_param - base_param
        print('***** Backborn param:{:0.3f}M, P-Tuning-V2 param is {} *****'.format(all_param, total_param))

    return model


def get_model_deprecated(model_args, task_type: TaskType, config: AutoConfig, fix_bert: bool = False):
    if model_args.prefix:
        config.hidden_dropout_prob = model_args.hidden_dropout_prob
        config.pre_seq_len = model_args.pre_seq_len
        config.prefix_projection = model_args.prefix_projection
        config.prefix_hidden_size = model_args.prefix_hidden_size

        if task_type == TaskType.TOKEN_CLASSIFICATION:
            from model.token_classification import BertPrefixModel, RobertaPrefixModel, DebertaPrefixModel, DebertaV2PrefixModel
        elif task_type == TaskType.SEQUENCE_CLASSIFICATION:
            from model.sequence_classification import BertPrefixModel, RobertaPrefixModel, DebertaPrefixModel, DebertaV2PrefixModel
        elif task_type == TaskType.QUESTION_ANSWERING:
            from model.question_answering import BertPrefixModel, RobertaPrefixModel, DebertaPrefixModel, DebertaV2PrefixModel
        elif task_type == TaskType.MULTIPLE_CHOICE:
            from model.multiple_choice import BertPrefixModel

        if config.model_type == "bert":
            model = BertPrefixModel.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                revision=model_args.model_revision,
            )
        elif config.model_type == "roberta":
            model = RobertaPrefixModel.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                revision=model_args.model_revision,
            )
        elif config.model_type == "deberta":
            model = DebertaPrefixModel.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                revision=model_args.model_revision,
            )
        elif config.model_type == "deberta-v2":
            model = DebertaV2PrefixModel.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                revision=model_args.model_revision,
            )
        else:
            raise NotImplementedError


    elif model_args.prompt:
        config.pre_seq_len = model_args.pre_seq_len

        from model.sequence_classification import BertPromptModel, RobertaPromptModel
        if config.model_type == "bert":
            model = BertPromptModel.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                revision=model_args.model_revision,
            )
        elif config.model_type == "roberta":
            model = RobertaPromptModel.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                revision=model_args.model_revision,
            )
        else:
            raise NotImplementedError
            

    else:
        if task_type == TaskType.TOKEN_CLASSIFICATION:
            model = AutoModelForTokenClassification.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                revision=model_args.model_revision,
            )
            
        elif task_type == TaskType.SEQUENCE_CLASSIFICATION:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                revision=model_args.model_revision,
            )

        elif task_type == TaskType.QUESTION_ANSWERING:
            model = AutoModelForQuestionAnswering.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                revision=model_args.model_revision,
            )
        elif task_type == TaskType.MULTIPLE_CHOICE:
            model = AutoModelForMultipleChoice.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                revision=model_args.model_revision,
            )
    
        bert_param = 0
        if fix_bert:
            if config.model_type == "bert":
                for param in model.bert.parameters():
                    param.requires_grad = False
                for _, param in model.bert.named_parameters():
                    bert_param += param.numel()
            elif config.model_type == "roberta":
                for param in model.roberta.parameters():
                    param.requires_grad = False
                for _, param in model.roberta.named_parameters():
                    bert_param += param.numel()
            elif config.model_type == "deberta":
                for param in model.deberta.parameters():
                    param.requires_grad = False
                for _, param in model.deberta.named_parameters():
                    bert_param += param.numel()
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
        total_param = all_param - bert_param
        print('***** total param is {} *****'.format(total_param))
    return model
