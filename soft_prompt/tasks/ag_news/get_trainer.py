import logging
import os
import random
import sys

from transformers import (
    AutoConfig,
    AutoTokenizer,
)

from model.utils import get_model, TaskType
from .dataset import AGNewsDataset
from training.trainer_base import BaseTrainer
from tasks import utils

logger = logging.getLogger(__name__)


def get_trainer(args):
    model_args, data_args, training_args, _ = args

    if "llama" in model_args.model_name_or_path:
        from transformers import LlamaTokenizer
        model_path = f'openlm-research/{model_args.model_name_or_path}'
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.mask_token = tokenizer.unk_token
        tokenizer.mask_token_id = tokenizer.unk_token_id
    elif 'opt' in model_args.model_name_or_path:
        model_path = f'facebook/{model_args.model_name_or_path}'
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
        )
        tokenizer.mask_token = tokenizer.unk_token
    elif 'gpt' in model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
        )
        tokenizer.pad_token_id = '<|endoftext|>'
        tokenizer.pad_token = '<|endoftext|>'
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
        )
    tokenizer = utils.add_task_specific_tokens(tokenizer)
    dataset = AGNewsDataset(tokenizer, data_args, training_args)

    if not dataset.is_regression:
        if "llama" in model_args.model_name_or_path:
            model_path = f'openlm-research/{model_args.model_name_or_path}'
            config = AutoConfig.from_pretrained(
                model_path,
                num_labels=dataset.num_labels,
                label2id=dataset.label2id,
                id2label=dataset.id2label,
                finetuning_task=data_args.dataset_name,
                revision=model_args.model_revision,
                trust_remote_code=True
            )
        elif "opt" in model_args.model_name_or_path:
            model_path = f'facebook/{model_args.model_name_or_path}'
            config = AutoConfig.from_pretrained(
                model_path,
                num_labels=dataset.num_labels,
                label2id=dataset.label2id,
                id2label=dataset.id2label,
                finetuning_task=data_args.dataset_name,
                revision=model_args.model_revision,
                trust_remote_code=True
            )
            config.mask_token = tokenizer.unk_token
            config.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
            config.mask_token_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
        else:
            config = AutoConfig.from_pretrained(
                model_args.model_name_or_path,
                num_labels=dataset.num_labels,
                label2id=dataset.label2id,
                id2label=dataset.id2label,
                finetuning_task=data_args.dataset_name,
                revision=model_args.model_revision,
            )
    else:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=dataset.num_labels,
            finetuning_task=data_args.dataset_name,
            revision=model_args.model_revision,
        )

    config.trigger = training_args.trigger
    config.clean_labels = training_args.clean_labels
    config.target_labels = training_args.target_labels
    model = get_model(model_args, TaskType.SEQUENCE_CLASSIFICATION, config)

    # Initialize our Trainer
    trainer = BaseTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset.train_dataset if training_args.do_train else None,
        eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
        compute_metrics=dataset.compute_metrics,
        tokenizer=tokenizer,
        data_collator=dataset.data_collator,
    )

    return trainer, None