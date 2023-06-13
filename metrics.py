import re
import os
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Tuple
from transformers import EvalPrediction, PreTrainedTokenizer

from loguru import logger

def decode(
    pred: EvalPrediction, 
    tokenizer: PreTrainedTokenizer
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
    pred_str = tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
    golden_ids = np.where(pred.label_ids == -100, tokenizer.pad_token_id, pred.label_ids)
    golden_str = tokenizer.batch_decode(golden_ids, skip_special_tokens=True)
    
    def lmap(f: Callable, x: Iterable) -> List:
        """list(map(f, x))"""
        return list(map(f, x))

    pred_str = lmap(str.strip, pred_str)
    golden_str = lmap(str.strip, golden_str)
    return pred_str, golden_str


class Extractor:
    def __init__(
        self, extract_template: str, targets: List[str], CSEP="<c>"):
        if extract_template=="default":
            self.extract_template = self.default_extract_template
        else:
            raise NotImplementedError(f"Not Implementation for extract template `{extract_template}`.")
        self.targets = targets
        self.CSEP = CSEP    # choices seperator

    def default_extract_template(self, string: str):
        def same_title_choices(string):
            string = string.replace("<title>", "")
            string = string.replace("<choices>", "")
            return string.strip(), string.strip()

        if "<title>" in string and "<choices>" in string:
            index1 = string.index("<title>")
            index2 = string.index("<choices>")
            if index1 > index2:
                logger.debug(f"idx1>idx2, string:\n {string}")
                return same_title_choices(string)
            title_str = string[index1+7: index2].strip()
            choices_str = string[index2+9:].strip()
            return title_str, choices_str
        else:
            if "title" in self.targets and "choices" in self.targets:
                logger.debug(f"missing title/choices, string:\n {string}")
            return same_title_choices(string)


    def extract_title_choices(self, strings: List, sort_choices=True):

        def sort_choices(choices: str):
            if self.CSEP not in choices:
                return choices
            else:
                choices_list = [choice.strip() for choice in choices.split(self.CSEP)]
                choices_list.sort()
                return f" {self.CSEP} ".join(choices_list)

        title_list, choices_list = [], []
        for x in strings:
            title, choices = self.extract_template(x)
            if sort_choices:
                choices = sort_choices(choices)
            title_list.append(title)
            choices_list.append(choices)
        return title_list, choices_list

def write_txt_file(ordered_tgt, path):
    f = Path(path).open("w")
    for ln in ordered_tgt:
        f.write(ln + "\n")
        f.flush()

def save_pred_golden(
    title_goldens, title_preds, choices_goldens, choices_preds,
    data_dir, output_dir, split, targets=["title", "choices"]
    ):
    ids_path = Path(data_dir).joinpath(f"{split}.ids")
    path = Path(output_dir).joinpath(f"generations_{split}.json")
    ids = open(ids_path).readlines()
    d = defaultdict(dict)
    for id, title_golden, title_pred, choices_golden, choices_pred in zip(
        ids, title_goldens, title_preds, choices_goldens, choices_preds):
        id = id.rstrip()
        if "title" in targets:
            d[id]["title_golden"] = title_golden
            d[id]["title_pred"] = title_pred
        if "choices" in targets:
            d[id]["choices_golden"] = choices_golden
            d[id]["choices_pred"] = choices_pred

    with open(path, "w+") as f:
        json.dump(d, f, indent=4, ensure_ascii=False)


from nltk.translate.bleu_score import sentence_bleu
def calculate_bleu(summary: List[str], reference: List[str], prefix):

    bleu_tota1 = 0
    bleu_tota2 = 0
    bleu_tota3 = 0
    bleu_tota4 = 0

    count = len(reference)
    assert len(reference) == len(summary)
    for ref, can in zip(reference, summary):
        reference = [ref.split()]
        candidate = can.split()
        bleu_tota1 += sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
        bleu_tota2 += sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
        bleu_tota3 += sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0))
        bleu_tota4 += sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
    Bleu = {}
    Bleu[f"{prefix}_bleu1"] = round(100 * bleu_tota1/ count, 4)
    Bleu[f"{prefix}_bleu2"] = round(100 * bleu_tota2/ count, 4)
    Bleu[f"{prefix}_bleu3"] = round(100 * bleu_tota3/ count, 4)
    Bleu[f"{prefix}_bleu4"] = round(100 * bleu_tota4/ count, 4)
    return Bleu 

import re
def map2digit(reference, summary):
    summary=[summary]
    lexicon = set()
    for line in reference:
        for tokens in line:
            assert isinstance(tokens, list) and len(tokens) == 1
            assert isinstance(tokens, list)
            ref = tokens[0]
            for t in ref:
                if re.search(u'[\u4e00-\u9fff]', t):
                    lexicon.add(t)
    for s in summary:
        for line in s:

            assert isinstance(tokens, list)
            summ = line[0]
            for t in summ:
                if re.search(u'[\u4e00-\u9fff]', t):
                    lexicon.add(t)

    c2d = {}
    d2c = {}
    for i, value in enumerate(lexicon):
        c2d[value] = str(i)
        d2c[i] = value

    def map_string(text, c2d):

        def spliteKeyWord(str):
            regex = r"[\u4e00-\ufaff]|[0-9]+|[a-zA-Z]+\'*[a-z]*"
            matches = re.findall(regex, str, re.UNICODE)
            return matches
        str_list = spliteKeyWord(text)
        return ' '.join([c2d[t] if re.search(u'[\u4e00-\u9fff]', t) else t for t in str_list])

    # map to digit
    res_ref = []
    res_summ = []
    for line in reference:
        tmp_s = []
        for tokens in line:
            assert isinstance(tokens, list) and len(tokens) == 1
            ref = tokens[0]  # string
            tmp = map_string(ref, c2d)
            tmp_s.append([tmp])
        res_ref.append(tmp_s)

    for s in summary:
        tmp_s = []
        for line in s:
            assert isinstance(line, list) and len(line) == 1
            summ = line[0]
            tmp = map_string(summ, c2d)
            tmp_s.append([tmp])
        res_summ.append(tmp_s)
    return res_ref, res_summ[0]

    
from pythonrouge.pythonrouge import Pythonrouge
# 这个安装包可以成功安装，但有个小bug，要按照error提示那里handle一下
# https://github.com/tagucci/pythonrouge
def calculate_rouge(summaries: List[str], references: List[str], prefix):
    references = [[[r]] for r in references]
    summaries = [[s] for s in summaries]
    references, summaries = map2digit(references, summaries)
    assert len(references) == len(summaries)
        
    rouge = Pythonrouge(summary_file_exist=False,
                        summary=summaries, reference=references,
                        n_gram=2, ROUGE_SU4=True, ROUGE_L=True,
                        recall_only=False, stemming=True, stopwords=False,
                        word_level=True, length_limit=True, length=50,
                        use_cf=False, cf=95, scoring_formula='average',
                        resampling=True, samples=1000, favor=True, p=0.5)
    rouge = rouge.calc_score()
    rouge = {
        "rouge1": rouge["ROUGE-1-F"],
        "rouge2": rouge["ROUGE-2-F"],
        "rougeL": rouge["ROUGE-L-F"],
        "rougeLsum": rouge["ROUGE-SU4-F"]
    }
    return {f"{prefix}_{key}": round(val*100, 4) for (key, val) in zip(rouge.keys(), rouge.values())}


def build_compute_metrics_fn(
    targets: List[str],
    tokenizer: PreTrainedTokenizer, 
    output_dir: str, 
    data_dir: str, 
    cache_dir: str = None,
    extract_template: str = "default", 
    split: str = "valid",
    CSEP: str = "<c>",
    sort_choices: bool = True
) -> Callable[[EvalPrediction], Dict]:

    def compute_metrics(pred: EvalPrediction) -> Dict:
        # Decode predictions and goldens
        pred_str, golden_str = decode(pred, tokenizer)

        # Save generations to disk
        write_txt_file(pred_str, os.path.join(output_dir, f"generations_{split}.txt"))

        # Lowercase both the golden and prediction for metrics computation
        pred_str = [string.lower() for string in pred_str]
        golden_str = [string.lower() for string in golden_str]

        # Extract title and sort choices
        extractor = Extractor(extract_template, targets, CSEP)
        title_goldens, choices_goldens = extractor.extract_title_choices(golden_str, sort_choices)
        title_preds, choices_preds = extractor.extract_title_choices(pred_str, sort_choices)

        # Save predictions and goldens to disk
        save_pred_golden(title_goldens, title_preds, choices_goldens, choices_preds, 
            data_dir, output_dir, split, targets)
        
        # Rouge scores
        title_rouge = calculate_rouge(title_preds, title_goldens, "title") \
            if "title" in targets else {}
        choices_rouge = calculate_rouge(choices_preds, choices_goldens, "choices") \
            if "choices" in targets else {}
        
        # Bleu scores
        title_bleu = calculate_bleu(title_preds, title_goldens, "title") \
            if "title" in targets else {}
        choices_bleu = calculate_bleu(choices_preds, choices_goldens, "choices")\
            if "choices" in targets else {}

        # Return metrics
        metrics = {}
        for metric in [title_rouge, title_bleu, choices_rouge, choices_bleu]:
            metrics.update(metric)      
        if "title" in targets and "choices" in targets:
            metrics["mean_rouge1"] = (metrics["title_rouge1"] + metrics["choices_rouge1"]) / 2    
        return metrics

    return compute_metrics
