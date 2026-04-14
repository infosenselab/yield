"""
Usage:
    python ./yield/evaluation/generate_dialogues.py
"""

# ---------------- Imports ----------------
import os

import yaml

from elicitation.utils import generate_utterances, utterances_to_dialogue


# ---------------- Arguments ----------------
BATCH_SIZE = 8

MODEL_TYPE = "finetuned" # "finetuned" / "prompted" 

# Paths
#MODEL_CHOICE = "meta-llama/Llama-3.1-8B-Instruct"
#MODEL_CHOICE = "meta-llama/Llama-3.2-3B-Instruct"
MODEL_CHOICE = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
FINETUNING_DATASET = "yield-v1-factualnovelty-rl-min3"
ADAPTER_MODEL =  "20251227T0851-deepseek-r1-distill-llama-8b-seq-std-m3trained" # None #
PROMPT_FILE =  None #"./config/generation_prompt.txt" # None #



# ---------------- Config ----------------
with open("./config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

proj_store = config["paths"]["proj_store"]

data_path = os.path.join(proj_store, "data")
data_input_folder = os.path.join(data_path, FINETUNING_DATASET, "test") # Only on the test set

models_folderpath = config["paths"]["models"]
base_model_path = f"{models_folderpath}/{MODEL_CHOICE}"

# Saved model and tokenizer path
if ADAPTER_MODEL:
    adapter_model_path = os.path.join(proj_store, "experiments", "fine-tuning", "adapter-models", ADAPTER_MODEL)
else:
    adapter_model_path = None 

# ---------------- Main ----------------
def main():
    
    # Generate utterances
    output_file = generate_utterances(
        model_choice = base_model_path,
        finetuning_dataset = data_input_folder,
        model_type = MODEL_TYPE,
        adapter_model = adapter_model_path,
        prompt_file = PROMPT_FILE,
        batch_size = BATCH_SIZE,
        save_dir = proj_store,
    )

    # Convert them to dialogue form
    utterances_to_dialogue(output_file)



# ------------------------
# EXECUTION
# ------------------------

if __name__ == "__main__":
    import torch
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())

    main()

