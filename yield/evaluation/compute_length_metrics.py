"""
Usage:
    python ./yield/evaluation/compute_length_metrics.py
"""
# ---------------- Imports ----------------
import os
from datetime import datetime

import yaml

#from sentence_transformers import SentenceTransformer

from elicitation.metrics.context_response_length import context_response_length



# ---------------- Arguments ----------------
TOKENIZER_MODEL = "meta-llama/Llama-3.2-3B-Instruct" # NOTE: Use llama tokenizer for all

INPUT_FILE = "20251227T2034-20251227t0851-deepseek-r1-distill-llama-8b-seq-std-m3trained"


# ---------------- Config ----------------
with open("./config/config.yaml", "r") as f:
    config = yaml.safe_load(f)


proj_store = config["paths"]["proj_store"]
models_folderpath = config["paths"]["models"]

input_filepath = os.path.join(proj_store, "evaluation", "generated-utterances", f"{INPUT_FILE}.jsonl")
tokenizer_model_filepath = os.path.join(models_folderpath, TOKENIZER_MODEL)


input_folder = os.path.dirname(input_filepath)
parent_folder = os.path.dirname(input_folder)
output_path = os.path.join(parent_folder, "length-metrics")
os.makedirs(output_path, exist_ok=True)
output_file = os.path.join(output_path, f"{INPUT_FILE}.csv")



# ---------------- Main ----------------
def main():
        
    print(f"\nGenerating from {INPUT_FILE}")
    
    crl_results = context_response_length(input_filepath=input_filepath, tokenizer_model=tokenizer_model_filepath, group_by="domain",  sort_by="domain")
    
    
    print(crl_results)
    
    crl_results.to_csv(output_file, index=False)
    print(f"\nSaved results to {output_file}")



# ------------------------
# EXECUTION
# ------------------------

if __name__ == "__main__":
    
    main()



