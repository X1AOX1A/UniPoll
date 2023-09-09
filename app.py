import gc
import sys
import torch
import argparse
import gradio as gr
from typing import List, Tuple
from transformers import AutoConfig
from transformers.models.mt5.modeling_mt5 import MT5ForConditionalGeneration
from utils import T5PegasusTokenizer

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
        logging.info("No comments input, comments will be ignored.")
        prompt = prompt.replace(" [SEP] {comments}", "")
        prompt = prompt.format(post=post)
    else:
        prompt = prompt.format(post=post, comments=comments)
    logging.info(f"Wrapped prompt: {prompt}")
    return prompt

@TimerDecorator
def generate(query, model, tokenizer, num_beams=4,  device="cpu"):
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

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--model_path", type=str, default="./outputs/UniPoll-t5/best_model", help="path to the model.")
    parser.add_argument("--device", type=str, default="cpu", help="specify the device to load the model, e.g. 'cpu', 'cuda:0'.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":    
    args = parse_args()
    
    logging.info('Initializing Model...')
    # prepare the model
    model, tokenizer = load_model(args.model_path, args.device)

    def submit(post, comments, num_beams):
        try:
            logging.info("Received post input: {}".format(post))
            if comments:
                logging.info("Received comments input: {}".format(comments))

            query = wrap_prompt(post, comments)
            raw_output = generate(
                query, model, tokenizer, num_beams, args.device)
            title, choices = post_process(raw_output)         # post process
            logging.info(f"Raw output: {raw_output}")
            logging.info(f"Processed title: {title}")
            logging.info(f"Processed choices: {choices}")
            # return title, choices, raw_output
            return title, choices
        except Exception as e:
            return "An error occurred: {}".format(str(e)), "An error occurred: {}".format(str(e))
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    examples = [
        ["#哪吒，大鱼海棠重映#动画电影《哪吒之魔童降世》、《大鱼海棠》，以及雷佳音、佟丽娅主演的 《超时空同居》确定将重映。据最新数据显示，3月24日全国复工影院495家，复工率4.36%，单日票房2.7万元。", "我在人间贩卖黄昏，只为收集世间温柔，去见你。谢谢你的分享，来看看你。我的微博，随时恭候你的到..."],
        ["#线上复试是否能保障公平＃ 高考延期惹的祸，考研线上复试，那还能保证公平吗？", "这个世界上本来就没有绝对的公平。你可以说一个倒数第一考了第一，但考上了他也还是啥都不会。也可以说他会利用一切机会达到目的，反正结果就是人家考的好，你还找不出来证据。线上考试，平时考倒数的人进了年级前十。平时考试有水分，线上之后，那不就是在水里考？"],
        ["#断亲现象为何如此流行#？所谓“断亲”指的是当代年轻人懒于、疏于、不屑于跟亲戚交往、联系、互动，日常音信全无，哪怕在逢年过节期间，宁可独来独往，也不愿意走亲戚，甚至将此作为一种时尚生活方式来推崇。", ""]
    ]
    
    description = """This is the demo of UniPoll. Please input post and comments. <div style='display:flex; gap: 0.25rem; '><a href='https://uni-poll.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a><a href='https://github.com/X1AOX1A/UniPoll'><img src='https://img.shields.io/badge/Github-Code-blue'></a><a href='https://arxiv.org/abs/2306.06851'><img src='https://img.shields.io/badge/Paper-PDF-red'></a></div>
"""

    demo = gr.Interface(
        fn=submit,
        inputs=[gr.Textbox(lines=1, label="Social Media Post", placeholder="Input post here..."),
                gr.Textbox(lines=1, label="Social Media Comments (Optional)", placeholder="Input comments here..."),
                gr.Number(value=4, label="Number of Beams", precision=0),
            ],                
        outputs=[gr.Textbox(lines=1, label="Generated Poll Question", placeholder="Generated poll question will be shown here"),
                 gr.Textbox(lines=1, label="Generated Poll Choices", placeholder="Generated poll choices will be shown here"),
           ], # question, choices
        title="Demo of UniPoll",
        description=description,
        allow_flagging="never",
        examples=examples,
    )
    
    demo.queue(max_size=10)
    demo.launch(share=True, show_error=True)


# python app.py --model_path "./outputs/UniPoll-t5/best_model" --device "cpu" 