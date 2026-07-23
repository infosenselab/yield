# YIELD: A Large-Scale Dataset and Evaluation Framework for Information Elicitation Agents

This repository contains the codebase and resources for “YIELD: A Large-Scale Dataset and Evaluation Framework for Information Elicitation Agents”


## Dataset & Tools

- [Dataset](https://huggingface.co/datasets/infosense/yield)
- [Fine-tuned model adapters](https://huggingface.co/infosense/yield-adapters)
- [`Elicitation` PyPI Package](https://pypi.org/project/elicitation/)



## Setup

Create `config/config.yaml`:

```yaml
paths:
  proj_store: "/data/yield"
  models: "/data/models"
```

- `proj_store`: root directory for datasets, outputs, and experiments
- `models`: directory containing downloaded base models



## Dataset Construction

- For generating data, use the pipeline in the `yield/dataset/` folder.
- For computing factual novelty, use the pipeline in the `yield/factual_novelty/` folder.




## Model Training 


### Dataset

If you are not constructing the dataset, then download YIELD and place it under `proj_store/data`, e.g.:

```
proj_store/data/
  ├── yield/
  ├── yield-finetuning/
```


### Models

Download base models from Hugging Face:

- LLama: [https://huggingface.co/meta-llama](https://huggingface.co/meta-llama)
- DeepSeek: [https://huggingface.co/deepseek-ai](https://huggingface.co/deepseek-ai)

Supported models:

- `meta-llama/Llama-3.1-8B-Instruct`
- `meta-llama/Llama-3.2-3B-Instruct`
- `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`




### SFT

The main supervised fine-tuning script is `yield/experiments/supervised_finetuning.py`. Run this file from the root folder. Training setup must be changing depending to your available resources.

**Example usage:**

```         
accelerate launch --config_file config/accelerate_config.yaml  ./yield/experiments/supervised_finetuning.py --model_choice meta-llama/Llama-3.1-8B-Instruct --dataset_choice yield_v1_finetuning
```



### ORL

The main supervised fine-tuning script is `yield/experiments/agent_llama.py`. Run this file from the root folder. Training setup must be changing depending to your available resources. The script is configured to use `accelerate`.

**Example usage:**

With DeepSpeed

```
accelerate launch --deepspeed_config_file config/deepspeed.json   yield/experiments/agent_llama.py   --model_choice meta-llama/Llama-3.1-8B-Instruct   --dataset_choice yield_v1_factualnovelty_rl
```


No DeepSpeed

```
accelerate launch yield/experiments/agent_llama.py   --model_choice meta-llama/Llama-3.1-8B-Instruct   --dataset_choice yield_v1_factualnovelty_rl
```



## Evaluating Models

To generate utterances from the models and performing evaluation, use scripts in the yield/evaluation/ folder. The scripts use the companion [Elicitation](https://pypi.org/project/elicitation/) package:

```
pip install elicitation
```

**Available metrics:**

- Conformity
- Progression
- Turn-Length Ratio


## Documentation


- [Cleaning Choices](docs/cleaning_choices.md)
- [Data_Sources](docs/data_sources.md)
- [Dataset Pipeline](docs/dataset_pipeline.md)
- [Other Appendix Items](docs/other_appendix.md)


## Citing YIELD

If you use this resource in your projects, please cite the following paper.

```bibtex
@inproceedings{De_Lima_YIELD_A_Large-Scale_2026,
author = {De Lima, Victor and Yang, Grace Hui},
booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026)},
doi = {10.18653/v1/2026.acl-long.678},
title = {{YIELD: A Large-Scale Dataset and Evaluation Framework for Information Elicitation Agents}},
url = {https://aclanthology.org/2026.acl-long.678/},
year = {2026}
}
```


