"""
Example usage:
    accelerate launch \
        --config_file config/accelerate_config.yaml \
        ./yield/experiments/supervised_finetuning.py \
        --model_choice deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
        --dataset_choice yield-v1-small10pct-finetuning
"""


# ---------------- Imports ----------------
import argparse
import sys
import os
import json
import logging

from datetime import datetime

import yaml
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer
from datasets import load_dataset


from accelerate import Accelerator



# ---------------- Arguments ----------------
parser = argparse.ArgumentParser(description="Training script with model choice")
parser.add_argument(
    "--model_choice",
    type=str,
    required=True,
    help="Fine-tuning model name (e.g. meta-llama/Llama-3.2-3B-Instruct). Must be in the models path specified in the config.yaml file."
)
parser.add_argument(
    "--dataset_choice",
    type=str,
    required=True,
    help="Dataset to use. Must be contained inside the data path specified in the config.yaml."
)

args = parser.parse_args()
model_choice = args.model_choice
dataset_choice = args.dataset_choice


# ---------------- Config ----------------
timestamp = datetime.now().strftime("%Y%m%dT%H%M")

with open("./config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

proj_store = config["paths"]["proj_store"]
data_folderpath = os.path.join(proj_store, "data")
models_folderpath = config["paths"]["models"]

dataset_name = os.path.join(data_folderpath, dataset_choice)
model_name = os.path.join(models_folderpath, model_choice)

train_data_folder = os.path.join(dataset_name, "train")
dev_data_folder = os.path.join(dataset_name, "dev")

output_dir = os.path.join(proj_store, "experiments", "fine-tuning")

adaptermodels_dir = os.path.join(output_dir, "adapter-models")
os.makedirs(adaptermodels_dir, exist_ok=True)

logs_dir = os.path.join(proj_store, "logs", "experiments", "fine-tuning")
os.makedirs(logs_dir, exist_ok=True)

log_filename = os.path.join(logs_dir, f"{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)  # still print to console
    ]
)

logger = logging.getLogger(__name__)


# Fine-Tuning Params
with open("./config/training_params.yaml", "r") as f:
    finetuning_params = yaml.safe_load(f)    

# Training params
training_ft_params = finetuning_params["training"]

batch_size = training_ft_params["batch_size"]
gradient_accumulation_steps = training_ft_params["gradient_accumulation_steps"]
logging_steps = training_ft_params["logging_steps"]
save_strategy = "epoch"
eval_strategy = "epoch"
learning_rate = float(training_ft_params["learning_rate"])
num_epochs = training_ft_params["num_epochs"]
warmup_ratio = training_ft_params["warmup_ratio"]
fp16 = training_ft_params["fp16"]
report_to = "none"


# PEFT params
peft_ft_params = finetuning_params["peft"]

r = peft_ft_params["r"]
lora_alpha = peft_ft_params["lora_alpha"]
target_modules = peft_ft_params["target_modules"]
lora_dropout = peft_ft_params["lora_dropout"]
bias = peft_ft_params["bias"]
task_type = peft_ft_params["task_type"]


# ---------------- Main ----------------


logger.info(f"Using model: {dataset_name}")
logger.info(f"Using dataset: {model_name}")
accelerator = Accelerator()
logger.info("\n=== ACCELERATE CONFIGURATION ===")
logger.info(f"\n{accelerator.state}")

def training_run(debug=False):

    # INIT
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True
    )

    # LOAD TOKENIZER
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # LOAD BASE MODEL
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        #device_map="auto",
        dtype=torch.float16
    )




    # CONFIGURE LoRA
    peft_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=task_type,
    )

    # LOAD DATASET
    dataset = load_dataset("json", data_files={
        "train": f"{train_data_folder}/*.jsonl",
        "validation": f"{dev_data_folder}/*.jsonl"
    })
    
    
    # NOTE! DON'T MOVE THIS OUT OF HERE
    def format_example(example):
        # Each example is a dict with "messages"
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,              # return plain text, trainer will tokenize
            add_generation_prompt=False  # no assistant stub at training time
        )
    




    # TRAINING ARGUMENTS
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=logging_steps,
        save_strategy=save_strategy,
        eval_strategy=eval_strategy,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        warmup_ratio=warmup_ratio,
        fp16=fp16,
        report_to=report_to,
    )

    # CREATE TRAINER
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        args=training_args,
        processing_class=tokenizer,
        formatting_func=format_example,
    )

    
    if debug:
        trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
        logger.info(f"Trainable params: {len(trainable_params)}")
        for n in trainable_params[:50]:
            logger.info(f"  - {n}")
        logger.info(f"Total trainable parameter count: {sum(p.numel() for n, p in model.named_parameters() if p.requires_grad):,}")


    
    #if debug:
    #    # Grab a batch from the dataloader and decode it
    #    train_dl = trainer.get_train_dataloader()
    #    first_batch = next(iter(train_dl))
    #    logger.info(f"Batch input_ids shape: {first_batch['input_ids'].shape}")
    #    logger.info(f"Batch labels shape: {first_batch['labels'].shape}")
    #
    #    decoded = tokenizer.batch_decode(first_batch["input_ids"], skip_special_tokens=False)
    #    for i, d in enumerate(decoded[:2]):
    #        logger.info(f"\n=== TRAIN SAMPLE {i} ===\n{d}")
    #    logger.info("\n=NOTE: When the dataloader batches examples, they are padded to the same length with <|eot_id|> tokens.")



    # START TRAINING
    trainer.train()

    print("Done.")

    # SAVE FINAL MODEL

    adapter_model_save_path = f"{output_dir}/adapter-models/{timestamp}-{model_choice.lower().split('/', 1)[1].replace('_', '-')}"


    trainer.model.save_pretrained(adapter_model_save_path)
    tokenizer.save_pretrained(adapter_model_save_path)


    print(f"Fine-tuned model saved at: {adapter_model_save_path}")



# ---------------- Execution ----------------
if __name__ == "__main__":
    training_run(debug=True)
    
