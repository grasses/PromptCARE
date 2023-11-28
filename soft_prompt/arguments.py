from enum import Enum
import argparse
import dataclasses
from dataclasses import dataclass, field
from typing import Optional
import json
from transformers import HfArgumentParser, TrainingArguments

from tasks.utils import *

@dataclass
class WatermarkTrainingArguments(TrainingArguments):
    removal: bool = field(
        default=False,
        metadata={
            "help": "Will do watermark removal"
        }
    )
    max_steps: int = field(
        default=0,
        metadata={
            "help": "Will do watermark removal"
        }
    )
    trigger_num: int = field(
        metadata={
            "help": "Number of trigger token: " + ", ".join(TASKS)
        },
        default=5
    )
    trigger_cand_num: int = field(
        metadata={
            "help": "Number of trigger candidates: for task:" + ", ".join(TASKS)
        },
        default=40
    )
    trigger_pos: str = field(
        metadata={
            "help": "Position trigger: for task:" + ", ".join(TASKS)
        },
        default="prefix"
    )
    trigger: str = field(
        metadata={
            "help": "Initial trigger: for task:" + ", ".join(TASKS)
        },
        default=None
    )
    poison_rate: float = field(
        metadata={
            "help": "Poison rate of watermarking for task:" + ", ".join(TASKS)
        },
        default=0.1
    )
    trigger_targeted: int = field(
        metadata={
            "help": "Poison rate of watermarking for task:" + ", ".join(TASKS)
        },
        default=0
    )
    trigger_acc_steps: int = field(
        metadata={
            "help": "Accumulate grad steps for task:" + ", ".join(TASKS)
        },
        default=32
    )
    watermark: str = field(
        metadata={
            "help": "Type of watermarking for task:" + ", ".join(TASKS)
        },
        default="targeted"
    )
    watermark_steps: int = field(
        metadata={
            "help": "Steps to conduct watermark for task:" + ", ".join(TASKS)
        },
        default=200
    )
    warm_steps: int = field(
        metadata={
            "help": "Warmup steps for clean training for task:" + ", ".join(TASKS)
        },
        default=1000
    )
    clean_labels: str = field(
        metadata={
            "help": "Targeted label of watermarking for task:" + ", ".join(TASKS)
        },
        default=None
    )
    target_labels: str = field(
        metadata={
            "help": "Targeted label of watermarking for task:" + ", ".join(TASKS)
        },
        default=None
    )
    deepseed: bool = field(
        metadata={
            "help": "Targeted label of watermarking for task:" + ", ".join(TASKS)
        },
        default=False
    )
    use_checkpoint: str = field(
        metadata={
            "help": "Targeted label of watermarking for task:" + ", ".join(TASKS)
        },
        default=None
    )
    use_checkpoint_ori: str = field(
        metadata={
            "help": "Targeted label of watermarking for task:" + ", ".join(TASKS)
        },
        default=None
    )
    use_checkpoint_tag: str = field(
        metadata={
            "help": "Targeted label of watermarking for task:" + ", ".join(TASKS)
        },
        default=None
    )



@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.training_args
    """
    task_name: str = field(
        metadata={
            "help": "The name of the task to train on: " + ", ".join(TASKS),
            "choices": TASKS
        }
    )
    dataset_name: str = field(
        metadata={
            "help": "The name of the dataset to use: " + ", ".join(DATASETS),
            "choices": DATASETS
        }
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=True, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(
        default=None, 
        metadata={"help": "A csv or a json file containing the test data."}
    )
    template_id: Optional[int] = field(
        default=0,
        metadata={
            "help": "The specific prompt string to use"
        }
    )

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_name_or_path_ori: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    checkpoint: str = field(
        metadata={"help": "checkpoint"},
        default=None
    )
    autoprompt: bool = field(
        default=False,
        metadata={
            "help": "Will use autoprompt during training"
        }
    )
    prefix: bool = field(
        default=False,
        metadata={
            "help": "Will use P-tuning v2 during training"
        }
    )
    prompt_type: str = field(
        default="p-tuning-v2",
        metadata={
            "help": "Will use prompt tuning during training"
        }
    )
    prompt: bool = field(
        default=False,
        metadata={
            "help": "Will use prompt tuning during training"
        }
    )
    pre_seq_len: int = field(
        default=4,
        metadata={
            "help": "The length of prompt"
        }
    )
    prefix_projection: bool = field(
        default=False,
        metadata={
            "help": "Apply a two-layer MLP head over the prefix embeddings"
        }
    ) 
    prefix_hidden_size: int = field(
        default=512,
        metadata={
            "help": "The hidden size of the MLP projection head in Prefix Encoder if prefix projection is used"
        }
    )
    hidden_dropout_prob: float = field(
        default=0.1,
        metadata={
            "help": "The dropout probability used in the models"
        }
    )

@dataclass
class QuestionAnwseringArguments:
    n_best_size: int = field(
        default=20,
        metadata={"help": "The total number of n-best predictions to generate when looking for an answer."},
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )
    version_2_with_negative: bool = field(
        default=False, metadata={"help": "If true, some of the examples do not have an answer."}
    )
    null_score_diff_threshold: float = field(
        default=0.0,
        metadata={
            "help": "The threshold used to select the null answer: if the best answer has a score that is less than "
            "the score of the null answer minus this threshold, the null answer is selected for this example. "
            "Only useful when `version_2_with_negative=True`."
        },
    )

def get_args():
    """Parse all the args."""
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, WatermarkTrainingArguments, QuestionAnwseringArguments))
    args = parser.parse_args_into_dataclasses()

    if args[2].watermark == "clean":
        args[2].poison_rate = 0.0

    if args[2].trigger is not None:
        raw_trigger = args[2].trigger.replace(" ", "").split(",")
        trigger = [int(x) for x in raw_trigger]
    else:
        trigger = np.random.choice(20000, args[2].trigger_num, replace=False).tolist()
    args[0].trigger = list([trigger])
    args[2].trigger = list([trigger])
    args[2].trigger_num = len(trigger)

    label2ids = []
    for k, v in json.loads(str(args[2].clean_labels)).items():
        label2ids.append(v)
    args[0].clean_labels = label2ids
    args[2].clean_labels = label2ids
    args[2].dataset_name = args[1].dataset_name

    label2ids = []
    for k, v in json.loads(str(args[2].target_labels)).items():
        label2ids.append(v)
    args[0].target_labels = label2ids
    args[2].target_labels = label2ids
    args[2].label_names = ["labels"]

    print(f"-> clean label:{args[2].clean_labels}\n-> target label:{args[2].target_labels}")
    return args