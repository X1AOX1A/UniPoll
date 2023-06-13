# UniPoll: : A Unified Social Media Poll Generation Framework via Multi-Objective Optimization

The official implementation of the paper [UniPoll: A Unified Social Media Poll Generation Framework via Multi-Objective Optimization](https://arxiv.org/abs/2306.06851). 

This repository aims to automate the generation of polls from social media posts using advanced natural language generation (NLG) techniques. The goal is to ensure that even passive browsing users have their perspectives considered in text analytics methods.

Key Features:

- Automatic generation of polls from social media posts.
- Leveraging cutting-edge NLG techniques to handle noisy social media data.
- Enriching post context with comments to capture implicit context-question-answer relations.
- UniPoll framework: A novel unified poll generation approach using prompt tuning and multi-objective optimization.
- Outperforms existing NLG models like T5 by generating interconnected questions and answers.

## Prepare the Environment

Please run the following commands to prepare the environment:

```bash
conda env create -f poll_questions.yaml
conda activate poll_questions
```

## Prepare the data

The original data can be downloaded from [this repo](https://github.com/polyusmart/Poll-Question-Generation/tree/main/data/Weibo), you can also find them in [./data/WeiboPolls/origin](./data/WeiboPolls/origin).

## Experiments

To reproduce the results in the paper, please run the following commands:

```bash
python finetuner.py configs/path_to_config_file.json
```

You can find the config files in the [./configs](./configs/) folder, where configs are splitted according to different experiments. There is a detailed description of the correspondence between model names and configurations in the [./configs/README.md](./configs/README.md) file.

If you want to reproduce the main results in the paper, please run the following commands:

```bash
python finetuner.py configs/main_ablations/UniPoll.json
```

## Citation

```
@misc{li2023unipoll,
      title={UniPoll: A Unified Social Media Poll Generation Framework via Multi-Objective Optimization}, 
      author={Yixia Li and Rong Xiang and Yanlin Song and Jing Li},
      year={2023},
      eprint={2306.06851},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Contact Information

If you have any questions or inquiries related to this research project, please feel free to contact:

- Yixia Li: yixiali@polyu.edu.hk