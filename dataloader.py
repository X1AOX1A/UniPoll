import os
import torch
import pickle
import linecache
from pathlib import Path
from typing import Dict
from transformers import BartTokenizer
from torch.utils.data import Dataset
from transformers.file_utils import cached_property


class Seq2SeqDataset(Dataset):
    """A dataset that calls prepare_seq2seq_batch."""
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_source_length,
        max_target_length,
        type_path="train",
        n_obs=None,
        prefix="",
        **dataset_kwargs
    ):
        super().__init__()
        self.src_file = Path(data_dir).joinpath(type_path + ".source")
        self.tgt_file = Path(data_dir).joinpath(type_path + ".target")
        self.ids_file = Path(data_dir).joinpath(type_path + ".ids")
        self.len_file = Path(data_dir).joinpath(type_path + ".len")
        self.src_len = 0
        self.tgt_len = 0

        def pickle_load(path):
            """pickle.load(path)"""
            with open(path, "rb") as f:
                return pickle.load(f)

        if os.path.exists(self.len_file):
            self.src_lens = pickle_load(self.len_file)
            self.used_char_len = False
        else:
            self.src_lens = self.get_char_lens(self.src_file)
            self.used_char_len = True
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        assert max_source_length is not None, f"`max_source_length` is None"
        assert max_target_length is not None, f"`max_target_length` is None"
        assert min(self.src_lens) > 0, f"found empty line in {self.src_file}"
        self.tokenizer = tokenizer
        self.prefix = prefix if prefix is not None else ""

        if n_obs is not None:    
            from math import ceil      
            # if n_obs smaller than 1, then treat as fraction  
            n_obs = ceil(len(self.src_lens) * n_obs) if n_obs<=1 else n_obs
            self.src_lens = self.src_lens[:n_obs]
        self.pad_token_id = self.tokenizer.pad_token_id
        self.dataset_kwargs = dataset_kwargs
        dataset_kwargs.update({"add_prefix_space": True} if isinstance(self.tokenizer, BartTokenizer) else {})

    def __len__(self):
        return len(self.src_lens)

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]

    @cached_property
    def tgt_lens(self):
        """Length in characters of target documents"""
        return self.get_char_lens(self.tgt_file)

    def __getitem__(self, index) -> Dict[str, str]:
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        id = linecache.getline(str(self.ids_file), index).rstrip("\n")

        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"
        return {"tgt_texts": tgt_line, "src_texts": source_line, "id": id}


class Seq2SeqDataCollator:
    def __init__(self, tokenizer, data_args):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        assert (
            self.pad_token_id is not None
        ), f"pad_token_id is not defined for ({self.tokenizer.__class__.__name__}), it must be defined."
        self.data_args = data_args
        self.dataset_kwargs = {"add_prefix_space": True} if isinstance(tokenizer, BartTokenizer) else {}

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        # Encode the text
        batch = self._encode(batch)
        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
        }

    def _encode(self, batch) -> Dict[str, torch.Tensor]:
        # Process src_texts
        batch_encoding = self.tokenizer(
            [x["src_texts"] for x in batch],
            add_special_tokens=True,
            return_tensors="pt",
            padding="longest",  # TPU hack,
            max_length=self.data_args.max_source_length,
            truncation=True,
            **self.dataset_kwargs,
        )

        # Process tgt_texts
        labels = self.tokenizer(
            [x["tgt_texts"] for x in batch],
            add_special_tokens=True,
            return_tensors="pt",
            padding="longest",
            max_length=self.data_args.max_target_length,
            truncation=True,
            **self.dataset_kwargs,
        )
        batch_encoding["labels"] = labels["input_ids"]

        return batch_encoding.data


def get_datasets(data_args, training_args, tokenizer):
    # get datasets
    train_dataset = (
        Seq2SeqDataset(
            tokenizer,
            type_path="train",
            data_dir=training_args.data_dir,
            n_obs=data_args.n_train,
            max_target_length=data_args.max_target_length,
            max_source_length=data_args.max_source_length,
        )
    )

    eval_dataset = (
        Seq2SeqDataset(
            tokenizer,
            type_path="valid",
            data_dir=training_args.data_dir,
            n_obs=data_args.n_val,
            max_target_length=data_args.max_target_length,
            max_source_length=data_args.max_source_length,
        )
    )

    test_dataset = (
        Seq2SeqDataset(
            tokenizer,
            type_path="test",
            data_dir=training_args.data_dir,
            n_obs=data_args.n_test,
            max_target_length=data_args.max_target_length,
            max_source_length=data_args.max_source_length,
        )
        if training_args.do_predict
        else None
    )

    return train_dataset, eval_dataset, test_dataset


from config import get_configs
from transformers import AutoTokenizer
if __name__ == "__main__":
    # Get configs
    configs = get_configs()
    model_args, data_args, training_args = configs

    # Get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    train_dataset, eval_dataset, test_dataset = get_datasets(data_args, training_args, tokenizer)
    print(train_dataset[0])