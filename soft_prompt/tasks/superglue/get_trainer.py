import logging
import os
import random
import torch

from transformers import (
    AutoConfig,
    AutoTokenizer,
)

from model.utils import get_model, TaskType
from tasks.superglue.dataset import SuperGlueDataset
from training import BaseTrainer
from training.trainer_exp import ExponentialTrainer
from tasks import utils
from .utils import load_from_cache

logger = logging.getLogger(__name__)

def get_trainer(args):
    model_args, data_args, training_args, _ = args
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    model_args.model_name_or_path = load_from_cache(model_args.model_name_or_path)

    if "llama" in model_args.model_name_or_path:
        from transformers import LlamaTokenizer
        model_path = f'openlm-research/{model_args.model_name_or_path}'
        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.mask_token = tokenizer.unk_token
        tokenizer.mask_token_id = tokenizer.unk_token_id
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
    dataset = SuperGlueDataset(tokenizer, data_args, training_args)

    if training_args.do_train:
        for index in random.sample(range(len(dataset.train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {dataset.train_dataset[index]}.")

    if not dataset.multiple_choice:
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
        else:
            config = AutoConfig.from_pretrained(
                model_args.model_name_or_path,
                num_labels=dataset.num_labels,
                label2id=dataset.label2id,
                id2label=dataset.id2label,
                finetuning_task=data_args.dataset_name,
                revision=model_args.model_revision,
                trust_remote_code=True
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

    if not dataset.multiple_choice:
        model = get_model(model_args, TaskType.SEQUENCE_CLASSIFICATION, config)
    else:
        model = get_model(model_args, TaskType.MULTIPLE_CHOICE, config, fix_bert=True)

    # Initialize our Trainer
    trainer = BaseTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset.train_dataset if training_args.do_train else None,
        eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
        compute_metrics=dataset.compute_metrics,
        tokenizer=tokenizer,
        data_collator=dataset.data_collator,
        test_key=dataset.test_key
    )


    return trainer, None
