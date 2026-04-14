"""
Usage:
    python ./yield/evaluation/conformity_perplexity.py
"""
# ---------------- Imports ----------------
import os
import yaml

from elicitation.metrics import conformity_perplexity


# ---------------- Args ----------------
#BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
#BASE_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
BASE_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
ADAPTER_MODEL = "20251227T0851-deepseek-r1-distill-llama-8b-seq-std-m3trained" # None # set to None for a prompted model
INPUT_FILE = "20251227T2034-20251227t0851-deepseek-r1-distill-llama-8b-seq-std-m3trained"


# ---------------- Config ----------------
with open("./config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

proj_store = config["paths"]["proj_store"]
models_folderpath = config["paths"]["models"]

input_filepath = os.path.join(proj_store, "evaluation", "generated-utterances", f"{INPUT_FILE}.jsonl")
base_model_path = os.path.join(models_folderpath, BASE_MODEL_NAME)
adapter_model_path = (
    os.path.join(proj_store, "experiments", "fine-tuning", "adapter-models", ADAPTER_MODEL)
    if ADAPTER_MODEL
    else None
)

# Automatically determine whether to use adapter
use_adapter = adapter_model_path is not None


input_folder = os.path.dirname(input_filepath)
parent_folder = os.path.dirname(input_folder)
output_path = os.path.join(parent_folder, "conformity", "perplexity")
os.makedirs(output_path, exist_ok=True)
output_file = os.path.join(output_path, f"{INPUT_FILE}.csv")



# ---------------- Main ----------------
def main():
        
    print(f"\nGenerating from {INPUT_FILE} on {ADAPTER_MODEL}")
    
    conformity_perplexity_results = conformity_perplexity(input_filepath, base_model_path, adapter_model_path, use_adapter, group_by="domain", sort_by="domain")
    print(conformity_perplexity_results)
    
    conformity_perplexity_results.to_csv(output_file, index=False)
    print(f"\nSaved results to {output_file}")



# ------------------------
# EXECUTION
# ------------------------

if __name__ == "__main__":
    
    main()