"""
Usage:
    python ./yield/evaluation/conformity_cossim.py
"""
# ---------------- Imports ----------------
import os
from datetime import datetime

import yaml

#from sentence_transformers import SentenceTransformer

from elicitation.metrics import conformity_cossim


# ---------------- Arguments ----------------
#EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2" # "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_MODEL = "cross-encoder/stsb-roberta-large"
INPUT_FILE = "20251218T2008-20251216t1321-llama-3.1-8b-instruct-seq-std-m4trained-m3generated"


# ---------------- Config ----------------
with open("./config/config.yaml", "r") as f:
    config = yaml.safe_load(f)


proj_store = config["paths"]["proj_store"]
models_folderpath = config["paths"]["models"]

input_filepath = os.path.join(proj_store, "evaluation", "generated-utterances", f"{INPUT_FILE}.jsonl")
embedding_model_filepath = os.path.join(models_folderpath, EMBEDDING_MODEL)


input_folder = os.path.dirname(input_filepath)
parent_folder = os.path.dirname(input_folder)
output_path = os.path.join(parent_folder, "conformity", "cossim")
os.makedirs(output_path, exist_ok=True)
output_file = os.path.join(output_path, f"{INPUT_FILE}.csv")



# ---------------- Main ----------------
def main():
        
    print(f"\nGenerating from {INPUT_FILE}")
    conformity_cossim_results = conformity_cossim(input_filepath, embedding_model_filepath, group_by="domain", sort_by="domain")
    print(conformity_cossim_results)
    
    conformity_cossim_results.to_csv(output_file, index=False)
    print(f"\nSaved results to {output_file}")



# ------------------------
# EXECUTION
# ------------------------

if __name__ == "__main__":
    
    main()



