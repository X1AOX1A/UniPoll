import os
import json
from pathlib import Path
from loguru import logger
from typing import Callable, Iterable, List


def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))

def write_txt_file(ordered_tgt, path):
    f = Path(path).open("w")
    for ln in ordered_tgt:
        f.write(ln + "\n")
        f.flush()


def save_json(content, path, indent=4, **json_dump_kwargs):
    with open(path, "w") as f:
        json.dump(content, f, indent=indent, sort_keys=True, **json_dump_kwargs)


def handle_metrics(split, metrics, output_dir):
    """
    Log and save metrics

    Args:
    - split: one of train, val, test
    - metrics: metrics dict
    - output_dir: where to save the metrics
    """

    logger.info(f"***** {split} metrics *****")
    for key in sorted(metrics.keys()):
        logger.info(f"  {key} = {metrics[key]}")
    save_json(metrics, os.path.join(output_dir, f"results_{split}.json"))


import shutil
def delete_checkpoints(model_dir):
    checkpoints = [folder for folder in os.listdir(model_dir) if folder.split("-")[0]=="checkpoint"]
    logger.info(f"Deleting checkpoints.\n{checkpoints}")
    for checkpoint in checkpoints:
        shutil.rmtree(os.path.join(model_dir, checkpoint))


import jieba
from functools import partial
from transformers import BertTokenizer

class T5PegasusTokenizer(BertTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_tokenizer = partial(jieba.cut, HMM=False)

    def _tokenize(self, text, *arg, **kwargs):
        split_tokens = []
        for text in self.pre_tokenizer(text):
            if text in self.vocab:
                split_tokens.append(text)
            else:
                split_tokens.extend(super()._tokenize(text))
        return split_tokens