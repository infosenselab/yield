"""
Example usage:
    accelerate launch \
        --config_file config/accelerate_config.yaml \
        ./yield/experiments/agent_llama.py \
        --model_choice meta-llama/Llama-3.2-3B-Instruct \
        --dataset_choice yield-v1-small1pct-factualnovelty-rl \
        --nametag seq-std
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
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_dataset



from accelerate import Accelerator

torch.set_printoptions(threshold=100000, linewidth=200)


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
parser.add_argument(
    "--nametag",
    type=str,
    required=True,
    help="Name appended to adapted model."
)

args = parser.parse_args()
model_choice = args.model_choice
dataset_choice = args.dataset_choice
nametag = args.nametag


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

output_dir = os.path.join(proj_store, "experiments")
adaptermodels_dir = os.path.join(output_dir, "fine-tuning", "adapter-models")
os.makedirs(adaptermodels_dir, exist_ok=True)

adapter_model_save_path = os.path.join(adaptermodels_dir, f"{timestamp}-{model_choice.lower().split('/', 1)[1].replace('_', '-')}-{nametag}")
os.makedirs(adapter_model_save_path, exist_ok=True)



logs_dir = os.path.join(proj_store, "logs", "training-logs")
os.makedirs(logs_dir, exist_ok=True)

log_filename = os.path.join(logs_dir, f"{timestamp}.log")





# Training Params
with open("./config/training_params.yaml", "r") as f:
    training_params_yaml = yaml.safe_load(f)

training_params = training_params_yaml["training"]

batch_size = training_params["batch_size"]
#gradient_accumulation_steps = training_params["gradient_accumulation_steps"]
logging_steps = training_params["logging_steps"]
#learning_rate = float(training_params["learning_rate"])
policy_lr = float(training_params["policy_lr"])
value_lr = float(training_params["value_lr"])
num_epochs = training_params["num_epochs"]
#warmup_ratio = training_params["warmup_ratio"]
#fp16 = training_params["fp16"]
temperature = training_params["temperature"]  # for AWR weighting



# PEFT params
peft_ft_params = training_params_yaml["peft"]

r = peft_ft_params["r"]
lora_alpha = peft_ft_params["lora_alpha"]
target_modules = peft_ft_params["target_modules"]
lora_dropout = peft_ft_params["lora_dropout"]
bias = peft_ft_params["bias"]
task_type = peft_ft_params["task_type"]


# ---------------- Classes and Functions ----------------
class ValueHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.value = nn.Linear(hidden_size, 1)

    #def forward(self, hidden_states):
    #    return self.value(hidden_states[:, -1, :])

    def forward(self, hidden_states, labels): # get hidden state of last unmasked EOS, not the very last hidden state
        # Masked tokens (labels == -100) are ignored
        valid_mask = (labels != -100)
        # Count how many tokens are valid per sequence
        last_indices = valid_mask.sum(dim=1) - 1  # (batch,)
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        # Extract the last valid hidden state (which aligns with your unmasked EOS)
        last_hidden = hidden_states[batch_indices, last_indices, :]
        return self.value(last_hidden)



def find_last_subsequence(seq, subseq):
    L = len(subseq)
    for i in range(len(seq) - L, -1, -1):
        if seq[i:i+L].tolist() == subseq:
            return i
    return None




def preprocess_function(example, tokenizer, model_name):
    
    model_name_actual = os.path.basename(model_name)
    
    allowed = ["Llama-3.2-3B-Instruct", "Llama-3.1-8B-Instruct", "DeepSeek-R1-Distill-Llama-8B"]

    if model_name_actual not in allowed:
        raise ValueError(f"Invalid model: {model_name_actual}. Supported models: {allowed}.")
    
    
    if model_name_actual in ["Llama-3.2-3B-Instruct", "Llama-3.1-8B-Instruct"]:
        header = "<|start_header_id|>assistant<|end_header_id|>"
    
    elif model_name_actual in ["DeepSeek-R1-Distill-Llama-8B"]:
        header = "<｜Assistant｜>"
        
    else:
        raise ValueError(f"No assistant header defined for model: {model_name_actual}")

    
    # Messages
    messages = example["messages"]

    # The LAST message must be an assistant utterance
    if messages[-1]["role"] != "assistant":
        raise ValueError(
            f"Dialogue {example.get('block_id', '<unknown>')} does not end with an assistant message."
        )
    
    # Use that last assistant message as the action
    #last_assistant_idx = len(messages) - 1
#
    ## Context: everything before that final assistant message
    #context = messages[:last_assistant_idx]
    #
    ## Target (action): the final assistant message itself
    #target = messages[last_assistant_idx]["content"]


    context_messages = messages[:-1]
    target_text = messages[-1]["content"]   # assistant answer

    # Build context text
    context_messages = tokenizer.apply_chat_template(
        context_messages,
        tokenize=False,
        add_generation_prompt=True
    )
    context_messages = context_messages.replace("<think>", "") # in case of deepseek

    
    # DEBUG
    if False:
        logger.info(f"context_messages: {context_messages}")
        logger.info(f"target_text: {target_text}")
    
    # Tokenize context and target so only the assistant output is predicted
    full_text = context_messages + target_text

    tokenized = tokenizer(
        full_text,
        #text_target=target,
        truncation=True,
        padding="max_length",
        max_length=512, # THIS SHOULD BE THE SAME AS THE LIMIT SET WHEN GENERATING THE DATASET
        add_special_tokens=False, # setting to True adds a double <|begin_of_text|>
        return_tensors="pt"
    )

    tokenized = {k: v.squeeze(0) for k, v in tokenized.items()}

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    labels = input_ids.clone()


    ## Tokenize the assistant answer separately to know its length
    #assistant_ids = tokenizer.encode(
    #        target_text,
    #        add_special_tokens=False
    #    )
    #len_assistant = len(assistant_ids)
#
    ## Assistant answer sits at the *end* of the full_text tensor
    #assistant_start = input_ids.size(0) - len_assistant
    
    
    
    header_ids = tokenizer.encode(header, add_special_tokens=False)


    header_pos = find_last_subsequence(input_ids, header_ids)
    if header_pos is None:
        raise RuntimeError("Could not find assistant header in tokenized text!")

    # Candidate assistant start = right after header
    assistant_start = header_pos + len(header_ids)

    # SKIP WHITESPACE-TOKENS AFTER HEADER
    while assistant_start < len(input_ids):
        decoded = tokenizer.decode([input_ids[assistant_start]], skip_special_tokens=False)
        if decoded.strip() != "":    # if non-whitespace, break
            break
        assistant_start += 1


    labels = input_ids.clone()
    labels[:assistant_start] = -100
    
    
    
    # mask padding so it doesn't make it into the loss
    #labels[attention_mask == 0] = -100
    
    eos_id = tokenizer.eos_token_id
    
    # Find last non-padding assistant token scanning from the end backward until find something not eos
    last_real_token = None
    for i in range(len(input_ids) - 1, assistant_start - 1, -1):
        if input_ids[i].item() != eos_id:
            last_real_token = i
            break
    
    if last_real_token is None:
        # handle where assistant output is only EOS tokens
        last_real_token = assistant_start
    
    # Keep only one EOS, mask everything after
    labels[last_real_token + 2 :] = -100  # +2 because last_real_token+1 is the real EOS

    
    
    
    
    
    tokenized["labels"] = labels


    
    
    


    # Attach returns for RL
    if "return_to_go" not in example:
        raise KeyError(f"Example is missing required field 'return_to_go': {example}")

    rtg = example["return_to_go"]
    
    
    # clip and scale returns to keep them within a stable numeric range
    #rtg = max(min(rtg, 10.0), -10.0)
    
    tokenized["returns"] = torch.tensor(rtg, dtype=torch.float)
    
    # DEBUG
    if False:
        logger.info(f"context_messages: {context_messages}")
        logger.info(f"target_text: {target_text}")
        logger.info(f"rtg: {rtg}")
    
    
    
    
    # DEBUG
    if False:
        decoded_after = tokenizer.decode(
            tokenized["input_ids"],
            skip_special_tokens=False
        )
        logger.info(f"CONTEXT AFTER TRUNCATION (decoded): {decoded_after}")
        
        # decode labels (remove what's masked with -100)
        label_ids = tokenized["labels"]
        valid = label_ids[label_ids != -100]

        decoded_labels = tokenizer.decode(valid, skip_special_tokens=False)
        logger.info(f"ASSISTANT LABELS ONLY (decoded): {decoded_labels}")
                
    
    return tokenized





def policy_loss(token_weights, label_log_probs):
    return -(token_weights * label_log_probs).mean()




def value_loss(values, returns):
    return F.mse_loss(values, returns)




# ---------------- Main ----------------
def training_run(debug=False):

    start_time = datetime.now()

    logger.info(f"Run started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    logger.info(f"Using model: {model_name}")
    logger.info(f"Using dataset: {dataset_name}")

    logger.info(f"\n=== TRAINING PARAMS ===\n"
                f"batch size: {batch_size}\n"
                #f"gradient accumulation steps: {gradient_accumulation_steps}\n"
                f"logging steps: {logging_steps}\n"
                #f"learning rate: {learning_rate}\n"
                f"policy lr: {policy_lr}\n"
                f"value lr: {value_lr}\n"
                f"num epochs: {num_epochs}\n"
                #f"warmup ratio: {warmup_ratio}\n"
                #f"fp16: {fp16}\n"
                f"temperature: {temperature}\n"
    )

    logger.info(f"\n=== PEFT PARAMS ===\n"
                f"r: {r}\n"
                f"lora_alpha: {lora_alpha}\n"
                f"target modules: {target_modules}\n"
                f"lora dropout: {lora_dropout}\n"
                f"bias: {bias}\n"
                f"task type: {task_type}\n"
    )



    # Run Setup
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # important for deekseek which defaults to left padding
    
    
    quant_config = BitsAndBytesConfig(load_in_8bit=True)
    #quant_config = BitsAndBytesConfig(load_in_8bit=False, load_in_4bit=False)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        dtype=torch.bfloat16,
        device_map=None,
    )

    # prepare it for LoRA fine-tuning
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=task_type,
    )

    model = get_peft_model(model, peft_config)



    # After LoRA config and model setup
    if False:
        trainable_lora = [(n, p.requires_grad) for n, p in model.named_parameters() if "lora" in n]
        logger.info(f"[DEBUG] LoRA params count: {len(trainable_lora)}")
        logger.info(
            "\n".join([f"{n}: requires_grad={p}" for n, p in trainable_lora[:20]])
        )
        

    if False:
        
        # Verify LoRA actually wraps the model
        logger.info(f"Forward path: {model.__class__.__name__}")

        try:
            logger.info(f"Example layer ref: {model.base_model.model.model.layers[0].self_attn.q_proj}")
        except AttributeError as e:
            logger.warning(f"Could not access nested layer via base_model.model.model: {e}")

        
    if False:
        trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
        logger.info(f"Trainable params: {len(trainable_params)}")
        for n in trainable_params[:50]:
            logger.info(f"  - {n}")
        logger.info(f"Total trainable parameter count: {sum(p.numel() for n, p in model.named_parameters() if p.requires_grad):,}")



    if False:
        trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
        logger.info(f"Trainable params: {len(trainable_params)}")
        for n in trainable_params[:10]:
            logger.info(f"  - {n}")


    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    value_head = ValueHead(model.config.hidden_size).to(accelerator.device)

    # Load dataset
    dataset = load_dataset("json", data_files={
        "train": f"{train_data_folder}/*.jsonl",
        #"validation": f"{dev_data_folder}/*.jsonl"
    })




    logger.info("Tokenizing dataset...")



    #train_dataset = dataset["train"].map(preprocess_function)
    train_dataset = dataset["train"].map(
        preprocess_function,
        fn_kwargs={"tokenizer": tokenizer, "model_name": model_name},
        batched=False,
        remove_columns=dataset["train"].column_names
    )
        #val_dataset = dataset["validation"].map(preprocess_function)

    # Convert dataset columns to PyTorch tensors
    train_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels", "returns"]
    )


    if False:
        import numpy as np
        returns_list = [ex["returns"].item() for ex in train_dataset]
        logger.info(f"returns: min={np.min(returns_list):.3f}, max={np.max(returns_list):.3f}, mean={np.mean(returns_list):.3f}, std={np.std(returns_list):.3f}")



    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False) # CHANGE THIS BACK

    #policy_optimizer = torch.optim.AdamW(model.parameters(), lr=policy_lr)
    #value_optimizer = torch.optim.AdamW(value_head.parameters(), lr=value_lr)
    #
    #
    ## Prepare only optimizers and dataloader first
    #value_head, policy_optimizer, value_optimizer, train_loader = accelerator.prepare(
    #    value_head, policy_optimizer, value_optimizer, train_loader
    #)
    #
    #
    #
    #model = accelerator.prepare(model)
    
    model = accelerator.prepare(model)
    value_head = accelerator.prepare(value_head)
    train_loader = accelerator.prepare(train_loader)

    policy_optimizer = torch.optim.AdamW(model.parameters(), lr=policy_lr)
    value_optimizer = torch.optim.AdamW(value_head.parameters(), lr=value_lr)

    
    
    model.train()

        




    logger.info(f"\n=== ACCELERATE CONFIGURATION ===\n{accelerator.state}")

    logger.info("Starting AWR offline training loop...")

    global_step = 0
    
    for epoch in range(num_epochs):
        
        logger.info(f"Step {global_step}")
        
        model.train()
        value_head.train()

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            
            input_ids = batch["input_ids"].to(accelerator.device)
            attention_mask = batch["attention_mask"].to(accelerator.device)
            labels = batch["labels"].to(accelerator.device)
            returns = batch["returns"].unsqueeze(1).to(accelerator.device)


            # normalize returns for numerical stability
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)


            ## DEBUG
            if False:
                decoded_after = tokenizer.decode(
                    input_ids[0],
                    skip_special_tokens=False
                )
                logger.info(f"CONTEXT IN LOOP (decoded): {decoded_after}")

                # decode labels (remove what's masked with -100)
                label_ids = labels[0]
                valid = label_ids[label_ids != -100]

                decoded_labels = tokenizer.decode(valid, skip_special_tokens=False)
                logger.info(f"ASSISTANT LABELS ONLY (decoded): {decoded_labels}")


            
            #with torch.enable_grad():
            #    for name, param in model.named_parameters():
            #        # only re-enable grads for LoRA parameters that are floating point
            #        if "lora" in name and param.dtype.is_floating_point:
            #            param.requires_grad_(True)
                
            # Get output
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
                
                # DEBUG
                #if False:
                #    logger.info(f"[DEBUG] After re-enable: logits requires_grad={outputs.logits.requires_grad}")


            
            
            
            # DEBUG
            if False and global_step == 0:
                for name, module in model.named_modules():
                    if "lora_A" in name:
                        logger.info(f"[DEBUG] Checking forward hook for {name}: {hasattr(module, 'forward')} | device: {next(module.parameters()).device}")
                logger.info(f"[DEBUG] Output logits requires_grad={outputs.logits.requires_grad}")




            # with an LLM the last layer's [-1] hidden state encodes everything that came before
            hidden_states = outputs.hidden_states[-1] 
            #hidden_states = outputs.hidden_states[-1].detach().clone().requires_grad_(True) # make hidden states differentiable

            
            # VALUE HEAD
            ###########################
            # obtain value head (critic) _prediction_ of the value estimate of the (context, action)
            #values = value_head(hidden_states)
            values = value_head(hidden_states, labels)


            
            # Advantage (residual) computation
            advantages = (returns - values).detach()
            
            if True:
                if global_step % 1 == 0:
                    logger.info(f"returns: {returns}")
                    logger.info(f"pred values: {values}")
                    logger.info(f"advantages: {advantages}")
            
                        
            if False:
                if global_step % 1 == 0:
                    logger.info(
                        f"[ADV] mean={advantages.mean().item():.4f}, "
                        f"std={advantages.std().item():.4f}, "
                        f"min={advantages.min().item():.4f}, "
                        f"max={advantages.max().item():.4f}"
            
                    )
                    
                    #val_grad_norm = 0
                    #for p in value_head.parameters():
                    #    if p.grad is not None:
                    #        val_grad_norm += p.grad.norm().item()
                    #logger.info(f"[VAL GRAD] norm={val_grad_norm:.3e}")
            
            
            
            if True:
                if global_step % 50 == 0:
                    try:
                        with torch.no_grad():
                            flat_returns = returns.detach().flatten()
                            flat_values = values.detach().flatten()

                            corr_val = float('nan')
                            if flat_returns.numel() > 1 and flat_values.numel() > 1:
                                stacked_pair = torch.stack([flat_returns, flat_values])
                                if stacked_pair.ndim == 2 and stacked_pair.size(0) == 2:
                                    corr_matrix = torch.corrcoef(stacked_pair)
                                    if corr_matrix.ndim == 2:
                                        corr_val = corr_matrix[0, 1].item()

                            mse_val = F.mse_loss(values, returns).item()

                        logger.info(
                            f"[VAL DIAG] step={global_step} corr={corr_val:.3f}, mse={mse_val:.3f}, "
                            f"return_mean={flat_returns.mean().item():.3f}, "
                            f"value_mean={flat_values.mean().item():.3f}"
                        )

                    except Exception as e:
                        logger.warning(f"[VAL DIAG ERROR] {repr(e)}")



            
            #advantages = torch.clamp(advantages, -1.0, 1.0)  # clip extremes to avoid overflow
            
            # normalize to stabilize exponent scaling
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Re-clamp after normalization
            #advantages = torch.clamp(advantages, -4.0, 4.0)
            
            # Scale down exponent sensitivity
            weights = torch.exp(0.25 * advantages / temperature)
            
            # Compute weights with bounded exponentials
            #weights = torch.exp(advantages / temperature)
            #weights = torch.clamp(weights, 0.1, 5.0)
            weights = weights / (weights.mean() + 1e-8)
            
            if False:
                    logger.info(f"weights: {weights}")

                
                            
            if False:
                if global_step % 1 == 0:
                    logger.info(
                        f"adv normalized mean={advantages.mean().item():.3f}, std={advantages.std().item():.3f}, "
                        f"weight mean={weights.mean().item():.3f}, std={weights.std().item():.3f}"
                    )

        
        
            if False:
                if global_step % 10 == 0:
                    logger.info(
                        f"[DEBUG] weights finite ratio: {(torch.isfinite(weights).float().mean().item() * 100):.2f}% | "
                        f"range: [{weights.min().item():.3f}, {weights.max().item():.3f}] | mean={weights.mean().item():.3f}, std={weights.std().item():.3f}"
                        f"[DEBUG] Effective weights range: [{weights.min().item():.3f}, {weights.max().item():.3f}], mean={weights.mean().item():.3f}, std={weights.std().item():.3f}"
                    )




            logits = outputs.logits
            
            shift_logits = logits[..., :-1, :].contiguous() # removes the LAST time step predictions
            
            if False:
                logger.info(f"logits shape: {logits.shape}")
                logger.info(f"logits: {logits}")
                logger.info(f"shift logits shape: {shift_logits.shape}")
                logger.info(f"shift logits: {shift_logits}")
            

            
            # remove the FIRST time step. Now shift_logits ends with penultimate token and shift_labels with the last label.
            shift_labels = labels[..., 1:].contiguous() 
            
            if False:
                logger.info(f"labels shape: {labels.shape}")
                logger.info(f"labels: {labels}")
                logger.info(f"shift labels shape: {shift_labels.shape}")
                logger.info(f"shift labels: {shift_labels}")
            
            valid_mask = (shift_labels != -100) # swaps values for True and False
            
            safe_shift_labels = shift_labels.masked_fill(shift_labels < 0, 0) # replace -100 with 0 (gather() cannot index negative nums)
            
            if False:
                logger.info(f"valid_mask shape: {valid_mask.shape}")
                logger.info(f"valid_mask: {valid_mask}")
                logger.info(f"safe_shift_labels shape: {safe_shift_labels.shape}")
                logger.info(f"safe_shift_labels: {safe_shift_labels}")
            
            
            

            
            predicted_log_probs = F.log_softmax(shift_logits, dim=-1)
            #log_probs = F.log_softmax(logits, dim=-1)
            
            if False:
                logger.info(f"predicted_log_probs shape: {predicted_log_probs.shape}")
                logger.info(f"predicted_log_probs: {predicted_log_probs}")

            
            
            # Create a new tensor having the log-probabilities for the label tokens
            label_log_probs = predicted_log_probs.gather(2, safe_shift_labels.unsqueeze(2)).squeeze(2)

            if False:
                logger.info(f"label_log_probs shape: {label_log_probs.shape}")
                logger.info(f"label_log_probs: {label_log_probs}")
            
            # Throw out stuff that was not the elicitors last response, only probabilities the model gave the elicitors last response remain
            label_log_probs = label_log_probs[valid_mask]   # (N_tokens,)


            if False:
                logger.info(f"label_log_probs masked shape: {label_log_probs.shape}")
                logger.info(f"label_log_probs masked: {label_log_probs}")

            

            
            # expand the weights to each matches the length of its corresponding labels
            token_weights = weights.expand_as(safe_shift_labels)[valid_mask]  # broadcast (B,1)->(B,T)->(N_tokens,)
         
            
            
            if False:
                logger.info(f"token_weights shape: {token_weights.shape}")
                logger.info(f"token_weights: {token_weights}")

            if False:
                logger.info(f"[DEBUG] values.requires_grad={values.requires_grad}, grad_fn={values.grad_fn}")




            L_v = value_loss(values, returns)
                        
            
            L_pi = policy_loss(token_weights, label_log_probs)
            
            loss = L_pi + L_v


            # Skip unstable updates
            if torch.isnan(loss):
                logger.warning("NaN loss detected, skipping update.")
                continue

            accelerator.backward(loss)


            if False:
                for n, p in value_head.named_parameters():
                    if p.grad is not None:
                        logger.info(f"[VAL GRAD] {n} grad norm={p.grad.norm().item():.3e}")

            if True:
                with torch.no_grad():
                    grad_norm = torch.nn.utils.clip_grad_norm_(value_head.parameters(), 9999)
                    logger.info(
                        f"step {global_step}: L_pi={L_pi.item():.3f}, L_v={L_v.item():.3f}, "
                        f"value_mean={values.mean().item():.2f}, value_std={values.std().item():.2f}, "
                        f"grad_norm={grad_norm:.2f}"
                    )

                        
            
        
            # DEBUG    
            if False and global_step == 0:
                grads_exist = [p.grad is not None for n, p in model.named_parameters() if "lora" in n]
                logger.info(f"[DEBUG] Any LoRA grads exist after backward: {any(grads_exist)}")

            
            # DEBUG
            if True and global_step % 50 == 0:
                grad_norms = []
                for name, param in model.named_parameters():
                    if "lora" in name and param.grad is not None:
                        grad_norms.append(param.grad.norm().item())
                if grad_norms:
                    mean_grad = sum(grad_norms) / len(grad_norms)
                    logger.info(f"[GRAD] LoRA grad mean={mean_grad:.6e}, min={min(grad_norms):.6e}, max={max(grad_norms):.6e}")
                else:
                    logger.warning("[GRAD] No LoRA gradients found (all None)")

            
            
            
            
            
            # Safety clamp for gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(value_head.parameters(), 5)
            
            
            policy_optimizer.step()
            value_optimizer.step()
            policy_optimizer.zero_grad()
            value_optimizer.zero_grad()
            
            
            
            if False:
                if global_step % 50 == 0:  # check every 50 steps
                    with torch.no_grad():
                        # Basic stats
                        ret_mean = returns.mean().item()
                        ret_std = returns.std().item()
                        val_mean = values.mean().item()
                        val_std = values.std().item()
                        adv = returns - values
                        adv_mean = adv.mean().item()
                        adv_std = adv.std().item()
                        w = torch.exp(adv / temperature)
                        w_mean = w.mean().item()
                        w_std = w.std().item()
            
                        logger.info(
                            f"[DEBUG] Step {global_step}: "
                            f"returns μ={ret_mean:.4f}, σ={ret_std:.4f} | "
                            f"values μ={val_mean:.4f}, σ={val_std:.4f} | "
                            f"advantages μ={adv_mean:.4f}, σ={adv_std:.4f} | "
                            f"weights μ={w_mean:.4f}, σ={w_std:.4f}"
                        )
            
                        # Optional: inspect a few samples
                        logger.info(f"[DEBUG] Sample returns: {returns[:5].view(-1).tolist()}")
                        logger.info(f"[DEBUG] Sample advantages: {adv[:5].view(-1).tolist()}")
                        logger.info(f"[DEBUG] Sample weights: {w[:5].view(-1).tolist()}")
                        
            
        

            # Save checkpoint'
            if accelerator.is_main_process and global_step % 2000 == 0:
                step_ckpt_path = os.path.join(adapter_model_save_path, f"checkpoint-step-{global_step}")
                os.makedirs(step_ckpt_path, exist_ok=True)

                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(step_ckpt_path, save_function=accelerator.save)
                accelerator.save(value_head.state_dict(), os.path.join(step_ckpt_path, "value_head.pt"))

                logger.info(f"Saved checkpoint at {step_ckpt_path}")
                        
            
            
            
            global_step += 1
            if global_step % logging_steps == 0:
                logger.info(f"Step {global_step}: total_loss={loss.item():.4f}, L_pi={L_pi.item():.4f}, L_v={L_v.item():.4f}")

    # Save models


    
    unwrapped_model = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        unwrapped_model.save_pretrained(
            adapter_model_save_path,
            save_function=accelerator.save
        )
        tokenizer.save_pretrained(adapter_model_save_path)
        accelerator.save(value_head.state_dict(),
                        os.path.join(adapter_model_save_path, "value_head.pt")) # saved in case continued training needed
        logger.info(f"Offline RL model saved at: {adapter_model_save_path}")

        
    

    # accelerator cleanup
    if "accelerator" in locals() and accelerator.distributed_type != "NO":
        from torch.distributed import destroy_process_group
        destroy_process_group()



    end_time = datetime.now()
    total_time = end_time - start_time
    logger.info(f"Run ended at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total training time: {str(total_time)}")



# ---------------- Execution ----------------
if __name__ == "__main__":
    accelerator = Accelerator()

    # Init the log
    logger = logging.getLogger(__name__)
    
    if accelerator.is_main_process: # only log once
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            #handlers=[logging.FileHandler(log_filename), logging.StreamHandler(sys.stdout)],
            handlers=[logging.FileHandler(log_filename)],
        )
    else:
        logging.basicConfig(level=logging.ERROR)  # silence logs from other ranks

    
    # call
    training_run(debug=True)

