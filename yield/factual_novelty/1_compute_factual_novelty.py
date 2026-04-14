"""
May take some 20 mins to run
Example usage: `python ./yield/factual_novelty/1_compute_factual_novelty.py --dataset_choice yield-v1`
"""



# ---------------- Imports ----------------
import warnings
warnings.filterwarnings("ignore", message=".*weights_only=False.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import glob
import argparse
from typing import Dict, Set, List, Tuple
import yaml
import spacy
import torch
from tqdm import tqdm
from multiprocessing import Process
import multiprocessing as mp
import orjson
from joblib import Parallel, delayed

mp.set_start_method("spawn", force=True)


# ---------------- Functions ----------------
def canonical_entity(ent) -> str:
    if ent.label_ == "CARDINAL":
        return None
    return f"{ent.label_}:{ent.text.strip().lower()}"


def process_dialogue(dialogue: Dict, docs_iter) -> Dict:
    """Process one dialogue using docs from a shared iterator."""
    seen: Set[str] = set()
    new_turns = []

    for turn in dialogue.get("turns", []):
        doc = next(docs_iter)
        turn_copy = {
            "turn_id": turn.get("turn_id"),
            "role": turn.get("role"),
            "utterance": turn.get("utterance"),
        }

        ents_this_turn = {
            ce for ent in doc.ents
            if (ce := canonical_entity(ent)) is not None
        }

        if turn.get("role") == "respondent":
            # new relative to all history so far
            new_ents = ents_this_turn - seen
            turn_copy["factual_novelty_entities"] = sorted(list(ents_this_turn))
            turn_copy["factual_novelty_score"] = len(new_ents)
            turn_copy["factual_accumulated_entities"] = sorted(list(seen | ents_this_turn))
            seen.update(ents_this_turn)

        else:  # elicitor
            turn_copy["factual_novelty_entities"] = sorted(list(ents_this_turn))
            turn_copy["factual_accumulated_entities"] = sorted(list(seen | ents_this_turn))
            seen.update(ents_this_turn)

        new_turns.append(turn_copy)

    return {
        "dialogue_id": dialogue.get("dialogue_id"),
        "broad_source": dialogue.get("broad_source"),
        "collection": dialogue.get("collection"),
        "domain": dialogue.get("domain"),
        "turns": new_turns,
    }


def read_file(fp: str) -> List[Dict]:
    """Read a JSON file into a list of dialogues."""
    with open(fp, "rb") as f:
        data = orjson.loads(f.read())
    if isinstance(data, dict) and "turns" in data:
        return [data]
    elif isinstance(data, list):
        return [d for d in data if isinstance(d, dict) and "turns" in d]
    else:
        return []


def write_file(out_path: str, data: List[Dict]):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(orjson.dumps(data[0] if len(data) == 1 else data, option=orjson.OPT_INDENT_2))


def process_files(files: List[str], input_root: str, output_root: str, gpu_id: int):
    """Worker function for one GPU, processes files sequentially to avoid OOM."""
    # Bind spaCy to the correct GPU
    spacy.require_gpu(gpu_id)
    nlp = spacy.load("en_core_web_trf", disable=["parser", "tagger", "lemmatizer"])

    for fp in files:
        # 1. Load dialogues from this file
        with open(fp, "rb") as f:
            data = orjson.loads(f.read())
        if isinstance(data, dict) and "turns" in data:
            dialogues = [data]
        elif isinstance(data, list):
            dialogues = [d for d in data if isinstance(d, dict) and "turns" in d]
        else:
            continue
        rel_path = os.path.relpath(fp, input_root)

        # 2. Collect utterances
        all_utterances: List[str] = []
        for d in dialogues:
            for t in d.get("turns", []):
                all_utterances.append(t.get("utterance", ""))

        # 3. Run spaCy on this file's utterances
        docs = list(
            nlp.pipe(all_utterances, batch_size=64)
        )
        docs_iter = iter(docs)

        # 4. Rebuild dialogues with annotated turns
        processed = [process_dialogue(dlg, docs_iter) for dlg in dialogues]

        # 5. Save immediately
        out_path = os.path.join(output_root, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(orjson.dumps(
                processed[0] if len(processed) == 1 else processed,
                option=orjson.OPT_INDENT_2
            ))

        # 6. Explicitly free memory
        del docs, docs_iter, all_utterances, processed, dialogues, data
        torch.cuda.empty_cache()



def main(dataset_choice: str):
    # ---------------- Config ----------------
    with open("./config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    proj_store = config["paths"]["proj_store"]
    data_path = os.path.join(proj_store, "data")

    dataset_choice_path = os.path.join(data_path, dataset_choice)
    output_choice_path = os.path.join(data_path, f"{dataset_choice}-factualnovelty")

    # Collect all JSON files
    files = glob.glob(os.path.join(dataset_choice_path, "**", "*.json"), recursive=True)
    files.sort()
    print(f"Found {len(files)} files under {dataset_choice_path}")

    # Detect GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs available. This script requires GPU(s).")
    print(f"Using {num_gpus} GPUs")

    # Split files into shards
    shards = [[] for _ in range(num_gpus)]
    for i, f in enumerate(files):
        shards[i % num_gpus].append(f)

    # Spawn one worker per GPU
    procs = []
    for gpu_id in range(num_gpus):
        p = Process(
            target=process_files,
            args=(shards[gpu_id], dataset_choice_path, output_choice_path, gpu_id)
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()


# ---------------- Main ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast factual novelty scoring with multi-GPU")
    parser.add_argument(
        "--dataset_choice",
        type=str,
        required=True,
        help="Dataset to use. Must be contained inside the data path specified in the config.yaml."
    )
    args = parser.parse_args()
    main(args.dataset_choice)
