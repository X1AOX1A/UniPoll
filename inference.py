from typing import List, Tuple
from transformers import AutoConfig
from transformers.models.mt5.modeling_mt5 import MT5ForConditionalGeneration
from utils import T5PegasusTokenizer

import sys
try:
    from loguru import logger as logging
    logging.add(sys.stderr, filter="my_module")
except ImportError:
    import logging

import time
class TimerDecorator:
    def __init__(self, func) :
        self.func = func

    def __call__(self, *args, **kwargs) :
        start_time = time.time()
        result = self.func(*args, **kwargs)
        end_time = time.time()
        t = end_time - start_time
        logging.info(f"Function `{self.func.__name__}` took {round(t, 2)} s to run.")
        return result

@TimerDecorator
def load_model(model_path, device="cpu"):
    logging.info(f"Loading model from {model_path}")
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = T5PegasusTokenizer.from_pretrained(model_path)
    model = MT5ForConditionalGeneration.from_pretrained(model_path, config=config)
    if device != "cpu":
        model.to(device)
    logging.info("Done.")
    return model, tokenizer

def wrap_prompt(
        post, comments, 
        prompt="生成 <title> 和 <choices>: [SEP] {post} [SEP] {comments}"
    ):
    if not comments or comments == "":
        prompt = prompt.replace(" [SEP] {comments}", "")
        return prompt.format(post=post)
    else:
        return prompt.format(post=post, comments=comments)

@TimerDecorator
def generate(query, model, tokenizer, num_beams=4, device="cpu"):
    logging.info("Generating output...")
    tokens = tokenizer(query, return_tensors="pt")["input_ids"]
    if device != "cpu":
        tokens = tokens.to(device)
    output = model.generate(tokens, num_beams=num_beams, max_length=100)
    output_text = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    logging.info("Done.")
    return output_text
    
def post_process(raw_output: str) -> Tuple[str, str]:
    def same_title_choices(raw_output):
        # return the same raw output as title and choices 
        # if no <title> or <choices> in raw_output
        raw_output = raw_output.replace("<title>", "")
        raw_output = raw_output.replace("<choices>", "")
        return raw_output.strip(), [raw_output.strip()]
    
    def split_choices(choices_str: str) -> List[str]:
        choices = choices_str.split("<c>")
        choices = [choice.strip() for choice in choices]
        return choices

    # extract title and choices from raw_output
    # e.g. raw_output = "<title> 你 觉得 线 上 复试 公平 吗 <choices> 公平 <c> 不 公平"
    if "<title>" in raw_output and "<choices>" in raw_output:
        index1 = raw_output.index("<title>")
        index2 = raw_output.index("<choices>")
        if index1 > index2:
            logging.debug(f"idx1>idx2, same title and choices will be used.\nraw_output: {raw_output}")
            return same_title_choices(raw_output)
        title = raw_output[index1+7: index2].strip()    # "你 觉得 线 上 复试 公平 吗"
        choices_str = raw_output[index2+9:].strip()     # "公平 <c> 不 公平"
        choices = split_choices(choices_str)            # ["公平", "不 公平"]
    else:        
        logging.debug(f"missing title/choices, same title and choices will be used.\nraw_output: {raw_output}")
        title, choices = same_title_choices(raw_output)

    def remove_blank(string):
        return string.replace(" ", "")
    
    title = remove_blank(title)
    choices = [remove_blank(choice) for choice in choices]
    return title, choices
    
if __name__ == "__main__":
    # finetuned model ckpt path
    # can be downloaded from https://drive.google.com/drive/folders/1hTO5N3NfMNi5AoEPxGhH3KNQ8olWuwUV?usp=sharing
    model_path = "./outputs/UniPoll-t5/best_model"    

    # input post and comments(optional, None) text
    post = "#线上复试是否能保障公平＃ 高考延期惹的祸，考研线上复试，那还能保证公平吗？"
    comments = "这个世界上本来就没有绝对的公平。你可以说一个倒数第一考了第一，但考上了他也还是啥都不会。也可以说他会利用一切机会达到目的，反正结果就是人家考的好，你还找不出来证据。线上考试，平时考倒数的人进了年级前十。平时考试有水分，线上之后，那不就是在水里考？"

    # generata parameters
    num_beams=4
    device="cpu"     # "cuda:0"
    
    model, tokenizer = load_model(model_path, device) # load model and tokenizer
    query = wrap_prompt(post, comments)               # wrap prompt
    raw_output = generate(query, model, tokenizer, num_beams, device)  # generate output
    title, choices = post_process(raw_output)         # post process

    print("Raw output:", raw_output)
    print("Processed title:", title)
    print("Processed choices:", choices)