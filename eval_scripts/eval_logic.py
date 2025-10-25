from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import logging
from itertools import islice
import spacy
from tqdm import tqdm
import torch
import pandas as pd
import ast
# device = "cpu" # can be "cpu" or "cuda
# inference on cuda takes too much memory
import pathlib
import sys




sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from utils import Config

config = Config("../configs/config_eval.json")
DEVICE = config.get("compute_device")

base = AutoModelForCausalLM.from_pretrained(config["model_dir"], device_map=DEVICE)

model = PeftModel.from_pretrained(base, config["finetuned_path"]).to(DEVICE)

tokenizer = AutoTokenizer.from_pretrained(config["finetuned_path"], use_fast=True)

# Configure tokenizer to match training setup
tokenizer.model_max_length = 4096  # Match training config
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'left'  # For generation

# System message for logic evaluation tasks
SYSTEM_MESSAGE_LOGIC = "You are a scientist, proficient in logic. Read the premises and evaluate whether the following statement is true, false, or uncertain, based solely on the premises. Your response can be only true, false or uncertain."

def format_logic_prompt_with_chat_template(premises: str, statement: str) -> str:
    """
    Format a logic evaluation prompt using Qwen chat template.

    Args:
        premises (str): The premises to base the evaluation on
        statement (str): The statement to evaluate

    Returns:
        str: Formatted prompt ready for generation
    """
    # Combine premises and statement into user message
    user_content = f"### Premises:\n{premises}\n\n### Statement:\n{statement}\n\nStatement above is:"

    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE_LOGIC},
        {"role": "user", "content": user_content},
    ]

    # Apply chat template with generation prompt (for inference)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

logging.info("Loading dataset:")


inputs = pd.read_csv(config['logic_data'])

inputs = inputs[inputs["depth"]<config["eval_depth_logic"]]
inputs['depth'] = inputs['depth'].astype(int)

inputs['premises'] = inputs['premises'].apply(lambda x:ast.literal_eval(x))



inputs = inputs.groupby('depth', group_keys=False).sample(config.get("eval_size_logic", 32), replace=False, random_state=0)

inputs["formatted_input"] = inputs.apply(lambda x: format_logic_prompt_with_chat_template("\n".join(x["premises"]), x['question']), axis = 1)
inputs["formatted_output"] = inputs["formatted_input"] + inputs["label"].str.lower()

inputs_in = inputs["formatted_input"].tolist()
inputs_out = inputs["label"].str.lower().tolist()
inputs_depth = inputs["depth"].tolist()

batch_size = config.get("batch_size_eval", 4)

def batched(iterable, n):
    it = iter(iterable)
    while (chunk := list(islice(it, n))):
        yield chunk

all_outputs_orig, all_outputs_ft = [], []

base.eval()
model.eval()

with torch.no_grad():                          # no grads for inference
    for chunk in tqdm(batched(inputs_in, batch_size)):
        encoded = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length
        ).to("cuda")

        outs_base = base.generate(
            **encoded,
            max_new_tokens=1200,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
        all_outputs_orig.extend(
            tokenizer.batch_decode(outs_base, skip_special_tokens=True)
        )

        # === finetuned (PEFT) model ===
        outs_ft = model.generate(
            **encoded,
            max_new_tokens=1200,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
        all_outputs_ft.extend(
            tokenizer.batch_decode(outs_ft, skip_special_tokens=True)
        )

responses_orig = all_outputs_orig
responses       = all_outputs_ft

nlp = spacy.load("en_core_web_md")

nlp_input_out = [nlp(x) for x in inputs_out]

results = dict()
results["inputs"] = inputs_in
results["inputs_depth"] = inputs_depth
results["ground_truth"] = inputs_out
results["responses_orig"] = responses_orig
results["responses_ft"] = responses

with open(f"out/{config.get('model_name')}_logic.json", "w") as f:
    json.dump(results, f)