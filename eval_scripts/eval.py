from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import logging
import gzip
import simphile
from itertools import islice
import spacy
from tqdm import tqdm
import torch
from faststylometry import Corpus, tokenise_remove_pronouns_en, calculate_burrows_delta
from datasets import load_dataset
import re
import pathlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent)) # otherwise ModuleNotFoundError: No module named 'utils' -- not sure why?
from utils import Config

def tokenise_en(text: str):
    """
    Tokenize English text into words, converting to lowercase.
    
    Args:
        text (str): Input text to tokenize
        
    Returns:
        list: List of lowercase word tokens
        
    Examples:
        >>> tokenise_en("Hello World!")
        ['hello', 'world']
    """
    return re.findall(r"[A-Za-z']+", text.lower())

def safe_tokenise(txt: str):
    """
    Safely tokenize text with pronoun removal, falling back to basic tokenization if needed.
    
    Args:
        txt (str): Input text to tokenize
        
    Returns:
        list: List of tokenized words with pronouns removed, or basic tokens if stripping fails
        
    Examples:
        >>> safe_tokenise("I am going to the store")
        ['going', 'store']
    """
    toks = tokenise_remove_pronouns_en(txt)
    # If everything was stripped, fall back to plain tokenisation
    if not toks:
        toks = tokenise_en(txt)
    return toks



llm_corpora = load_dataset("browndw/human-ai-parallel-corpus")["train"]
llm_corpus = llm_corpora.to_pandas()["text"].tolist()
llm_titles = llm_corpora.to_pandas()["doc_id"].tolist()


corpus = Corpus()
for i, llm_doc in enumerate(llm_corpus):
    corpus.add_book("LLM-corpus", llm_titles[i], llm_doc)



sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

config = Config("../configs/config_eval.json")

DEVICE = config.get("compute_device")

base = AutoModelForCausalLM.from_pretrained(config["model_dir"], device_map="auto").eval()
# important - use "device_map" = "auto - reduces memory footprint
model = PeftModel.from_pretrained(
    base, config["finetuned_path"], device_map="auto"
).eval()

tokenizer = AutoTokenizer.from_pretrained(config["finetuned_path"], use_fast=True)

# Configure tokenizer to match training setup
tokenizer.model_max_length = 4096  # Match training config
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'left'  # For generation

# System message matching training script
SYSTEM_MESSAGE = "You are a scientist with advanced knowledge in philosophy and social sciences. Please, write the next paragraph for the following text."

def format_prompt_with_chat_template(context: str) -> str:
    """
    Format a prompt using Qwen chat template.

    Args:
        context (str): The text context to continue

    Returns:
        str: Formatted prompt ready for generation
    """
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": context},
    ]

    # Apply chat template with generation prompt (for inference)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

logging.info("Loading dataset from preprocessed eval data:")

# Load preprocessed eval data (same format as training)
inputs_context = []
inputs_response = []

# Get eval_size early for memory-efficient loading
eval_size = config.get("eval_size", 256)
# Load a bit more than eval_size to account for potential decode errors
max_load = eval_size + 100

logging.info(f"  Loading from {config['preprocessed_eval_path']}")
with gzip.open(config['preprocessed_eval_path'], 'rt', encoding='utf-8') as f:
    for line_num, line in enumerate(f):
        # Stop early if we've loaded enough examples
        if len(inputs_context) >= max_load:
            break

        try:
            example = json.loads(line)
            inputs_context.append(example.get("context", ""))
            inputs_response.append(example.get("response", ""))
        except json.JSONDecodeError as e:
            logging.warning(f"  Line {line_num}: JSON decode error: {e}")
            continue

logging.info(f"  Loaded {len(inputs_context)} examples from preprocessed eval data")

# Limit to eval_size samples BEFORE building corpus (more efficient)
# (eval_size already retrieved above during loading)
inputs_context = inputs_context[:eval_size]
inputs_response = inputs_response[:eval_size]

logging.info(f"Using {len(inputs_context)} examples for evaluation")

# Build corpus from contexts for stylometry analysis (only the examples we'll use)
for i, context in enumerate(inputs_context):
    corpus.add_book("Our corpus", str(i), context)

corpus.tokenise(safe_tokenise)

# Format inputs with chat template (contexts are already properly chunked)
inputs_in = [format_prompt_with_chat_template(context) for context in inputs_context]
inputs_out = inputs_response  # Responses are already the ground truth


# inputs_formatted = tokenizer(
#     inputs_in,
#     return_tensors="pt"
# ).to("cuda")

# max_seq_len = tokenizer.model_max_length          # e.g. 4096 for Llama-family, 131 072 for RWKV-world etc.
# inputs_formatted = tokenizer(
#     inputs_in,
#     return_tensors="pt",
#     padding=True,             # or "longest"
#     truncation=True,          # **required** so over-length samples are cut
#     max_length=max_seq_len    # be explicit; you can also set a smaller value
# ).to("cuda")
#
#
# outputs_orig = base.generate(
#     input_ids=inputs_formatted.input_ids,
#     attention_mask=inputs_formatted.attention_mask,
#     max_new_tokens=1200,
#     eos_token_id=tokenizer.eos_token_id,
#     use_cache=True)
#
# outputs = model.generate(
#     input_ids=inputs_formatted.input_ids,
#     attention_mask=inputs_formatted.attention_mask,
#     max_new_tokens=1200,
#     eos_token_id=tokenizer.eos_token_id,
#     use_cache=True
# )

test_corpus_orig = Corpus()
test_corpus_finetuned = Corpus()

batch_size = config.get("batch_size", 1)       # try smaller if you OOM
# ----------------------------------

def batched(iterable, n):
    """
    Split an iterable into batches of specified size.
    
    Args:
        iterable: Any iterable object
        n (int): Size of each batch
        
    Yields:
        list: Batches of size n from the iterable
        
    Examples:
        >>> list(batched([1, 2, 3, 4, 5], 2))
        [[1, 2], [3, 4], [5]]
    """
    it = iter(iterable)
    while (chunk := list(islice(it, n))):
        yield chunk

all_outputs_orig, all_outputs_ft = [], []


# with torch.no_grad():                          # no grads for inference
#     for chunk in tqdm(batched(inputs_in, batch_size)):
#         encoded = tokenizer(
#             chunk,
#             return_tensors="pt",
#             padding=True,
#             truncation=True,
#             max_length=tokenizer.model_max_length, padding_side = 'left'
#         ).to("cuda")
#
#         outs_base = base.generate(
#             **encoded,
#             max_new_tokens=1200,
#             eos_token_id=tokenizer.eos_token_id,
#             use_cache=True
#         )
#         all_outputs_orig.extend(
#             tokenizer.batch_decode(outs_base, skip_special_tokens=True)
#         )
#
#         # === finetuned (PEFT) model ===
#         outs_ft = model.generate(
#             **encoded,
#             max_new_tokens=1200,
#             eos_token_id=tokenizer.eos_token_id,
#             use_cache=True
#         )
#         all_outputs_ft.extend(
#             tokenizer.batch_decode(outs_ft, skip_special_tokens=True)
#         )

# Generate responses from base model first
with torch.no_grad():                          # no grads for inference
    for chunk in tqdm(batched(inputs_in, batch_size)):
        encoded = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length
        ).to(DEVICE) 

        outs_base = base.generate(
            **encoded,
            max_new_tokens=1200,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
        all_outputs_orig.extend(
            tokenizer.batch_decode(outs_base, skip_special_tokens=True)
        )
    # base.to("cpu");
    # del base;
    # gc.collect();
    # torch.cuda.empty_cache();
    # torch.cuda.ipc_collect()
    
    # Generate responses from finetuned model
    for chunk in tqdm(batched(inputs_in, batch_size)):
        encoded = tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=tokenizer.model_max_length
            ).to(DEVICE)
        # finetuned (PEFT) model
        outs_ft = model.generate(
            **encoded,
            max_new_tokens=1200,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
        all_outputs_ft.extend(
            tokenizer.batch_decode(outs_ft, skip_special_tokens=True)
        )


# replace the old variables so the rest of the script stays unchanged
responses_orig = all_outputs_orig
responses       = all_outputs_ft

# print(len(responses_orig))
# print(responses_orig[0])


# responses_orig = tokenizer.batch_decode(outputs_orig, skip_special_tokens=True)
#
# responses = token_response

# Create corpora for stylometric analysis of generated responses
for i, resp in enumerate(responses_orig):
    test_corpus_orig.add_book("Test corpus", str(i), resp)

test_corpus_orig.tokenise(safe_tokenise)

for i, resp in enumerate(responses):
    test_corpus_finetuned.add_book("Test corpus, finetuned", str(i), resp)

test_corpus_finetuned.tokenise(safe_tokenise)

nlp = spacy.load("en_core_web_md")

nlp_input_out = [nlp(x) for x in inputs_out]

# Calculate various similarity metrics between generated responses and ground truth
results = dict()
results["inputs"] = inputs_in
results["orig"] = dict()
results["orig"]["jaccard"] = [simphile.jaccard_similarity(x, inputs_out[i]) for i,x in enumerate(responses_orig)]
results["orig"]["compression"] = [simphile.compression_similarity(x, inputs_out[i]) for i,x in enumerate(responses_orig)]
results["orig"]["spacy_sim"] = [nlp(x).similarity(nlp_input_out[i]) for i,x in enumerate(responses_orig)]

results["orig"]["responses"] = responses_orig


# print("books in test_corpus_orig :", len(test_corpus_orig.books))
# print("tokens in first book     :",
#       test_corpus_orig.books[0].tokens if test_corpus_orig.books else [])
# print("difference_matrices size :", len(getattr(test_corpus_orig, "difference_matrices", [])))



results["orig"]["burrows"] = calculate_burrows_delta(corpus, test_corpus_orig, vocab_size = 100).to_dict()

results["finetuned"] = dict()
results["finetuned"]["jaccard"] = [simphile.jaccard_similarity(x, inputs_out[i]) for i,x in enumerate(responses)]
results["finetuned"]["compression"] = [simphile.compression_similarity(x, inputs_out[i]) for i,x in enumerate(responses)]
results["finetuned"]["spacy_sim"] = [nlp(x).similarity(nlp_input_out[i]) for i,x in enumerate(responses)]

results["finetuned"]["responses"] = responses

results["finetuned"]["burrows"] = calculate_burrows_delta(corpus, test_corpus_finetuned, vocab_size = 100).to_dict()

# Save evaluation results to output file
with open(f"out/{config.get('model_name')}_completion.json", "w") as f:
    json.dump(results, f)