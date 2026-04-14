# ---------------- Imports ----------------
import os
import json
import sys
import textwrap
import logging

from datetime import datetime

import yaml
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel




# ---------------- Args ----------------
#domain = "academic interviews"
#domain = "journalistic investigations"
#domain = "judicial proceedings"
domain = "oral history"

SYSTEM_PROMPT = f"Act as an information elicitation agent for {domain}. You will be interviewing John, who worked 30 years at NASA."


use_adapter = True

MAX_TURNS = 15  # user+assistant turns

model_choice = "meta-llama/Llama-3.1-8B-Instruct"
#model_choice = "meta-llama/Llama-3.2-3B-Instruct"

adapter_choice = "20251216T1321-llama-3.1-8b-instruct-seq-std" # RL
#adapter_choice = "20250930T1246-llama-3.2-3b-instruct-sft" # SFT





# ---------------- Config ----------------
timestamp = datetime.now().strftime("%Y%m%dT%H%M")

with open("./config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

proj_store = config["paths"]["proj_store"]
models_folderpath = config["paths"]["models"]

base_model_path = os.path.join(models_folderpath, model_choice)
adapter_model_path = os.path.join(
    proj_store,
    "experiments",
    "fine-tuning",
    "adapter-models",
    adapter_choice
)


logs_dir = os.path.join(proj_store, "logs", "agent-chat")
os.makedirs(logs_dir, exist_ok=True)

log_filename = os.path.join(logs_dir, f"{timestamp}.log")

# Init the log
logger = logging.getLogger(__name__)
    
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    #handlers=[logging.FileHandler(log_filename), logging.StreamHandler(sys.stdout)],
    handlers=[logging.FileHandler(log_filename)],
)



# ---------------- Load Model ----------------
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    dtype=torch.float16,
    device_map="auto"
)

if use_adapter:
    model = PeftModel.from_pretrained(model, adapter_model_path)
    model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(
    base_model_path,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=500
)



# ---------------- Chat Setup ----------------

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "I am a user that wants to talk to you."},
    {"role": "assistant", "content": "Ok, let's do it."},
    {"role": "user", "content": "Whenever you're ready."},
    {"role": "assistant", "content": "Let's go."},
    
]





def trim_history(messages, max_turns):
    system = messages[:1]
    rest = messages[1:]

    # keep last max_turns * 2 messages (user + assistant)
    rest = rest[-(max_turns * 2):]

    return system + rest



# ---------------- Chat Loop ----------------
print("Chat started. Type 'exit' to quit.\n")

if not use_adapter:
    print("################# CAUTION: NOT USING AN ADAPTER.\n")
    

turn_id = 0

while True:
    user_input = input("User:\n").strip()
    
    if user_input.lower() in {"exit", "quit"}:
        break
    
    turn_id += 1

    messages.append({"role": "user", "content": user_input})

    messages = trim_history(messages, MAX_TURNS)

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    prompt = prompt.replace("<think>", "")

    logger.info(f"[TURN {turn_id}]")
    
    logger.info(f"prompt: {prompt}")
    
    
    # get result
    result = pipe(prompt)
    generated = result[0]["generated_text"]

    # Extract only the new assistant response
    assistant_reply = generated[len(prompt):].strip()

    messages.append({"role": "assistant", "content": assistant_reply})

    print("\nAssistant:")
    print(textwrap.fill(assistant_reply, width=80))
    print()

