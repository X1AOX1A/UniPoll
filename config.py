import os
import sys
import json
import time
from pathlib import Path
from loguru import logger
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from transformers import HfArgumentParser


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune from."""

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"})
    model_config: Optional[str] = field(
        default=None, 
        metadata={"help": "Pretrained config name or path if not the same as model_name"})
    tokenizer_name: Optional[str] = field(
        default=None, 
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"})
    load_checkpoint_from: Optional[str] = field(
        default=None, 
        metadata={"help": "Path to checkpoint for evaluation"})
    add_tokens: bool = field(
        default=True, 
        metadata={"help": "Whether to add tokens"})


@dataclass
class DataTrainingArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""

    raw_data_dir: str = field(
        default="./data/WeiboPolls/origin",
        metadata={"help": "The raw data dir."})
    schemas: Dict[str, list] = field(
        default_factory=lambda: {
            "train": [["main_task", -1], ["subtask_1", -1], ["subtask_2", -1]],
            "valid": [["main_task", -1]],
            "test": [["main_task", -1]]},
        metadata={"help": "The schemas for data preprocessing."
        "Key is the data split, Value is [task_name, comments_pect]"
        "comments_pect: length percent of comments"
        "-1 stands for all comments"
        "80 or 0.8 stands for 80% of comments."
        })
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={"help": "The maximum total input sequence length after tokenization. "
        "Sequences longer than this will be truncated, sequences shorter will be padded."})
    max_target_length: Optional[int] = field(
        default=128,
        metadata={"help": "The maximum total sequence length for target text after tokenization. " 
        "Sequences longer than this will be truncated, sequences shorter will be padded."})
    n_train: Optional[int] = field(
        default=None, 
        metadata={"help": "first # training examples. Decimal means fraction. None means use all."})
    n_val: Optional[int] = field(
        default=None, 
        metadata={"help": "first # validation examples. Decimal means fraction. None means use all."})
    n_test: Optional[int] = field(
        default=None, 
        metadata={"help": "first # test examples. Decimal means fraction. None means use all."})
    eval_beams: Optional[int] = field(
        default=None, 
        metadata={"help": "# num_beams to use for evaluation."})
    add_prompt: bool = field(
        default=True, 
        metadata={"help": "Whether to add prompt before sentence."})

    def __post_init__(self):  
        # Auto set targets
        if self.schemas["valid"][0][0] == "main_task":
            self.targets = ["title", "choices"]
        elif self.schemas["valid"][0][0] == "subtask_1":
            self.targets = ["title"]
        elif self.schemas["valid"][0][0] == "subtask_2":
            self.targets = ["choices"]
        else:
            raise NotImplementedError("Can't infer targets from valid schemas.")


@dataclass
class TrainingArguments:
    """Arguments pertaining to what parameters we used to fine-tune the model."""

    # All about the directories and file names
    run_name: Optional[str] = field(
        default="RedditPolls", 
        metadata={"help": "The name of expriment."})
    time_stamp: Optional[str] = field(
        default=time.strftime("%b_%d_%H-%M-%S", time.localtime()), 
        metadata={"help": "The time stamp of expriment."})
    data_dir: Optional[str] = field(
        default="./data/", 
        metadata={"help": "The preprocessed data dir."})
    output_dir: Optional[str] = field(
        default="./outputs/models/", 
        metadata={"help": "The directory to save the finetuned model."})
    config_dir: Optional[str] = field(
        default="./outputs/configs/", 
        metadata={"help": "The directory to save the configs."})
    logging_dir: Optional[str] = field(
        default="./outputs/logs/", 
        metadata={"help": "The directory to save the logs."})
    metrics_dir: Optional[str] = field(
        default="./outputs/metrics/", 
        metadata={"help": "The directory to save the metrics."})
    cache_dir: Optional[str] = field(
        default="./outputs/cache_dir/", 
        metadata={"help": "The directory to store the cache."})
    config_name: Optional[str] = field(
        default="time_stamp", 
        metadata={"help": "The file name of saved configs."})
    logging_name: Optional[str] = field(
        default="time_stamp", 
        metadata={"help": "The file name of saved logs."})

    # All about the training parameters
    learning_rate: float = field(
        default=2.5e-5, 
        metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(
        default=0.0,
        metadata={"help": "Weight decay for AdamW if we apply some."})
    warmup_ratio: float = field(
        default=0.0, 
        metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."})
    num_train_epochs: float = field(
        default=3.0, 
        metadata={"help": "Total number of training epochs to perform."})
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update "
        "pass."})
    gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "If True, use gradient checkpointing to save memory at the expense of "
        "slower backward pass."})
    logging_steps: int = field(
        default=500, 
        metadata={"help": "Log every X updates steps."})
    per_device_train_batch_size: int = field(
        default=8, 
        metadata={"help": "Batch size per GPU/TPU core/CPU for training."})
    per_device_eval_batch_size: int = field(
        default=8, 
        metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."})
    do_train: bool = field(
        default=False, 
        metadata={"help": "Whether to run training."})
    do_eval: bool = field(
        default=False, 
        metadata={"help": "Whether to run eval on the dev set."})
    do_predict: bool = field(
        default=False, 
        metadata={"help": "Whether to run predictions on the test set."})
    seed: int = field(
        default=42, 
        metadata={"help": "Random seed that will be set at the beginning of training."})
    predict_with_generate: bool = field(
        default=True, 
        metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."})
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "The path to a folder with a valid checkpoint for your model."})
    report_to: Optional[List[str]] = field(
        default=None, 
        metadata={"help": "The list of integrations to report the results and logs to."})
    metric_for_best_model: Optional[str] = field(
        default="eval_mean_rouge1", 
        metadata={"help": "The metric to use to compare two different models."})
    
    # Used to start the finetuning, not directly used by model
    gpus: Optional[str] = field(
        default="0", 
        metadata={"help": "GPU indexs to use. '0,1' means use GPU 0 and 1"})

    def __post_init__(self):   
        self.data_dir = self.check_path(self.data_dir)
        self.output_dir = self.check_path(self.output_dir)
        self.config_dir = self.check_path(self.config_dir)
        self.logging_dir = self.check_path(self.logging_dir)
        self.metrics_dir = self.check_path(self.metrics_dir)
        self.config_name = self.check_file_name(self.config_name)
        self.logging_name = self.check_file_name(self.logging_name)

    def check_path(self, dir):
        """Auto switch to time_stamp / run_name."""
        path = str(Path(dir))
        dir = "/".join(path.split("/")[:-1])
        folder_name = path.split("/")[-1]
        if folder_name == "time_stamp":
            path = os.path.join(dir, self.time_stamp)
        elif folder_name == "run_name":
            path = os.path.join(dir, self.run_name)
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def check_file_name(self, file_name):
        """Auto switch to time_stamp / run_name."""
        if file_name == "time_stamp":
            file_name = self.time_stamp
        elif file_name == "run_name":
            file_name = self.run_name
        return file_name


def get_configs():
    """Get configs from parser or json."""
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    if training_args.metric_for_best_model == "eval_title_rouge1" and \
        "title" not in data_args.targets:
        raise Warning("Can't use `eval_title_rouge1` for best model since 'title' not in targets.")
    elif training_args.metric_for_best_model == "eval_choices_rouge1" and \
        "choices" not in data_args.targets:
        raise Warning("Can't use `eval_choices_rouge1` for best model since 'choices' not in targets.")

    return (model_args, data_args, training_args)


def save_configs(configs):
    """Save configs to disk in json."""
    _, _, training_args = configs
    config_json = {}
    for config in configs:
        config_json.update(config.__dict__)    
    file_name = os.path.join(
        training_args.config_dir, training_args.config_name + ".json")
    if not os.path.exists(training_args.config_dir):
        os.makedirs(training_args.config_dir)
    with open(file_name, "w") as f_out:
        json.dump(config_json, f_out, indent=4, sort_keys=True)
    logger.info(f"Saved configs to {file_name}.")


if __name__ == "__main__":
    configs = get_configs()
    model_args, data_args, training_args = configs
    save_configs(configs)