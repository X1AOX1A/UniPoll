import os
import json
from loguru import logger
from config import get_configs
from utils import T5PegasusTokenizer

TASK_LIST = ["main_task", "subtask_1", "subtask_2"]

class Tasks:   
    """Prompt tasks.
    return src and tgt except for the first "(CLS)" and the last "(SEP)"        
    """        
    def __init__(self, add_prompt, SEP):
        self.add_prompt = add_prompt
        self.SEP = " "+SEP+" "

    def main_task(self, example):
        """Main task for both title and choices."""
        # src: (CLS) generate <title> then <choices>: [SEP] DESC [SEP] COMMENTS (SEP)
        # tgt: (CLS) <title> TITLE <choices> CHOICES (SEP)
        DESC, COMMENTS = example["description"], example["comments"]
        TITLE, CHOICES = example["title"], example["choices"]
        return {
            "id": example["id"],
            "src": ("生成 <title> 和 <choices>:" + self.SEP if self.add_prompt else "") + \
                DESC + (self.SEP + COMMENTS if COMMENTS else ""),
            "tgt": "<title> "+ TITLE + " <choices> " + CHOICES
        }

    def subtask_1(self, example):
        """Sub-task for title."""
        # src: (CLS) generate <title>: [SEP] DESC [SEP] COMMENTS (SEP)
        # tgt: (CLS) <title> TITLE (SEP)
        DESC, COMMENTS, TITLE = example["description"], example["comments"], example["title"]
        return {
            "id": example["id"],
            "src": ("生成 <title>:" + self.SEP if self.add_prompt else "") +\
                DESC + (self.SEP + COMMENTS if COMMENTS else ""),
            "tgt": ("<title> " if self.add_prompt else "") + TITLE
        }

    def subtask_2(self, example):
        """Sub-task for choices."""
        # src: (CLS) generate <choices>: [SEP] DESC [SEP] COMMENTS (SEP)
        # tgt: (CLS) <choices> CHOICES (SEP)
        DESC, COMMENTS, CHOICES = example["description"], example["comments"], example["choices"]
        return {
            "id": example["id"],
            "src": ("生成 <choices>:" + self.SEP if self.add_prompt else "") + \
                DESC + (self.SEP + COMMENTS if COMMENTS else ""),
            "tgt": ("<choices> " if self.add_prompt else "") + CHOICES
        }


class Preprocessor():
    """Preprocess the raw data with specific schema.

    - raw_data_dir: str, the raw data dir.
    - data_dir: str, the preprocessed data dir.
    - schemas: dict, the scheams for train/valid/test data.
        - task_name: ref to `TASK_LIST` for avilable tasks.
        - comments_pect: length percent of comments, -1 stands for all comments.
        e.g.:
        schemas = {
            "train": [(task_name, comments_pect), (task_name, comments_pect), ...],
            "valid": [(task_name, comments_pect), (task_name, comments_pect), ...],
            "test": [(task_name, comments_pect), (task_name, comments_pect), ...]
        }
    - add_prompt: bool, whether to add prompt before sentence
    - SEP: str, seperate token for sentence(including comments)
    - CSEP: str, seperate token for choices
    """
    def __init__(self, raw_data_dir, data_dir, schemas, add_prompt, SEP, CSEP):
        self.raw_data_dir = raw_data_dir
        self.data_dir = data_dir
        self.schemas = schemas
        self.add_prompt = add_prompt
        self.SEP = SEP if SEP else "</s>"   # seperate token for sentence(including comments)
        self.CSEP = CSEP                    # seperate token for choices

        # check schemas
        if schemas["valid"] != schemas["test"]:
            raise Warning(f"valid schema != test schema, please check schema settings.")
        if len(schemas["valid"]) > 1:
            raise Warning(f"There are multi tasks in valid set, please be carefull.")
        if len(schemas["test"]) > 1:
            raise Warning(f"There are multi tasks in valid set, please be carefull.")
        

    def build_dataset(self, split):
        """Build dataset."""
        schema = self.schemas[split]
        file_in = os.path.join(self.raw_data_dir, split+".json")
        file_out = os.path.join(self.data_dir, split)
        src_file = open(file_out+".source", "w")
        tgt_file = open(file_out+".target", "w")
        ids_file = open(file_out+".ids", "w")
        tasks = Tasks(self.add_prompt, self.SEP)

        def data_reader(raw_data_dir, split, target):
            file = os.path.join(raw_data_dir, f"{split}_{target}.txt")
            with open(file, "r") as f:
                data = f.readlines()
                data = [line.strip().replace(" ", "") for line in data]
            return data

        description = data_reader(self.raw_data_dir, split, "src")
        comments = data_reader(self.raw_data_dir, split, "conv")
        title = data_reader(self.raw_data_dir, split, "trg")
        choices = data_reader(self.raw_data_dir, split, "choice")
        assert len(description)==len(comments)==len(title)==len(choices)

        def split_choices(string):
            choices = {}
            for i, choice in enumerate(string.split("<sep>")):
                choices[i] = choice
            return choices

        def concat_choices(example):
            # CHOICES = CHOICE [C] CHOICE [C] CHOICE
            choices = ""
            CSEP = " "+self.CSEP+" "        # " <c> "
            for choice in example["choices"].values():
                choices = choices + choice + CSEP
            choices = choices[:-len(CSEP)]   # remove last CSEP token
            return choices
        
        def first_percent_comments(example, percent=-1):
            # percent: length percent of comments
            # -1 stands for all comments
            # 80 or 0.8 stands for 80% of comments
            if percent == -1:
                return example["comments"]
            else:
                if percent>1:
                    percent = percent/100
                length = len(example["comments"])
                return example["comments"][:int(length*percent)]

        def write_example(example_out):
            src_file.write(json.dumps(example_out["src"], ensure_ascii=False)[1:-1] + "\n")
            tgt_file.write(json.dumps(example_out["tgt"], ensure_ascii=False)[1:-1] + "\n")
            ids_file.write(json.dumps(example_out["id"], ensure_ascii=False)[1:-1] + "\n")
        
        count=0
        for idx, (desc, com, tit, cho) in enumerate(zip(description, comments, title, choices)):
            example = {
                "id": str(idx),
                "description": desc,
                "comments": com,
                "title": tit,
                "choices": split_choices(cho)
            }               
            for (task_name, comments_pect) in schema:
                assert task_name in TASK_LIST, f"`{task_name}`  not in task list: {TASK_LIST}"
                assert comments_pect>=-1, f"`comments_pect` needs to be >=-1, not {comments_pect}."
                tmp = example.copy()
                tmp["choices"] = concat_choices(tmp)
                tmp["comments"] = first_percent_comments(tmp, comments_pect)
                if task_name == "main_task":
                    tmp = tasks.main_task(tmp)                        
                elif task_name == "subtask_1":
                    tmp = tasks.subtask_1(tmp)                        
                elif task_name == "subtask_2":
                    tmp = tasks.subtask_2(tmp)                
                write_example(tmp)
                count += 1

        src_file.close()
        tgt_file.close()
        ids_file.close()
        logger.info(f"Saved to {self.data_dir}.")
        logger.info(f"{count} examples generated.")

    def process(self):
        """Process all splits."""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        for split in ["train", "valid", "test"]:     
            logger.info(f"Processing {split}ing data...")
            self.build_dataset(split)


if __name__ == "__main__":    
    model_args, data_args, training_args = get_configs()
    tokenizer = T5PegasusTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    preprocessor = Preprocessor(
        data_args.raw_data_dir, training_args.data_dir, data_args.schemas,
        data_args.add_prompt, tokenizer.sep_token, "<c>")
    preprocessor.process()
