"""
Qwen LLM Finetuning with LoRA - Multi-GPU Support

This script finetunes Qwen2.5 or Qwen3 models using:
  - LoRA (Low-Rank Adaptation) for efficient finetuning
  - Preprocessed markdown chunks (prepared by preprocess_markdown_chunks.py)
  - Qwen chat template for instruction tuning
  - Distributed Data Parallel (DDP) for multi-GPU training

=== RUNNING ON MULTI-GPU (e.g., 2x A100) ===

Option 1: Using accelerate (recommended for HTCondor)
  $ accelerate launch finetuning_skip.py

Option 2: Using torchrun (manual process spawning)
  $ torchrun --nproc_per_node=2 finetuning_skip.py

Option 3: Direct execution (auto-detects GPUs if only 1 GPU)
  $ python finetuning_skip.py

The script will:
  - Auto-detect available GPUs on startup (logged at beginning)
  - Automatically use DDP if multiple GPUs are detected
  - Calculate effective batch size = per_device_batch_size * num_gpus * grad_accum_steps
  - Distribute data across GPUs transparently

=== HTCondor SUBMISSION EXAMPLE ===

Create submit file with:
  request_gpus = 2
  request_memory = 80 GB
  executable = /path/to/accelerate_wrapper.sh

Where accelerate_wrapper.sh contains:
  #!/bin/bash
  cd /path/to/llm_project
  source /path/to/venv/bin/activate
  accelerate launch llm_scripts/finetuning_skip.py

=== PERFORMANCE TIPS ===

For 2x A100 (141GB total):
  - Current: batch_size=4 per GPU, effective=16 (with grad_accum=2)
  - Faster: batch_size=8 per GPU, effective=32 (uses ~60GB total)
    Edit: per_device_train_batch_size=8 in config or script

Monitor GPU utilization:
  - Run on compute node: nvidia-smi -l 1
  - Check in WandB: gpu and gpu_memory charts should show both GPUs
"""

import os

#import subprocess
#subprocess.run("yes | pip install bitsandbytes")
import itertools
import transformers.utils
transformers.utils.is_rich_available = lambda: False

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from tqdm import trange, tqdm
import os
from typing import List, Dict, Iterator
import datasets
from transformers import DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments, TrainerCallback
from transformers.integrations import WandbCallback
from transformers import Seq2SeqTrainingArguments
from sentence_transformers import SentenceTransformer, util
from transformers import GenerationConfig
import json
import gzip

# from dotenv import load_dotenv
# from b2_uploader import B2Uploader
import time
from pathlib import Path

import sys
import pathlib
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gc, torch
import os, tempfile, wandb, json
from utils import Config

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.pipeline import Pipeline, PipelineParameters, ParallelismManager
from lighteval.models.transformers.transformers_model import TransformersModelConfig
import argparse

import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


config = Config(str(Path(__file__).resolve().parent.parent / "configs/config_finetuning.json"))
os.environ["WANDB_PROJECT"] = config["WANDB_PROJECT"]   # must come before Trainer is built
os.environ["WANDB_LOG_MODEL"] = config["WANDB_LOG_MODEL"]

# NLTK no longer needed for preprocessing; using section-aware chunking instead

path_to_out = Path(config["output_dir"])
path_to_out.mkdir(exist_ok=True)

# GPU Detection and Logging
num_gpus = torch.cuda.device_count()
logging.info(f"GPU Detection: {num_gpus} GPU(s) available")
for i in range(num_gpus):
    gpu_name = torch.cuda.get_device_name(i)
    gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
    logging.info(f"  GPU {i}: {gpu_name} ({gpu_mem:.1f} GB)")

if num_gpus == 0:
    logging.warning("⚠️  No GPUs detected! Training will use CPU (very slow)")
elif num_gpus > 1:
    logging.info(f"✓ Multi-GPU training enabled: {num_gpus} GPUs will be used with DDP")

def run_lighteval(checkpoint_path, tasks):
    """
    Run LightEval evaluation on a model checkpoint.
    
    Evaluates the model on specified tasks using the LightEval framework.
    The evaluation is performed using CPU to avoid GPU memory conflicts.
    
    Args:
        checkpoint_path (str): Path to the model checkpoint to evaluate
        tasks (list): List of task specifications to evaluate on
        
    Returns:
        dict: Nested dictionary containing evaluation results for all tasks
        
    Examples:
        >>> tasks = ["leaderboard|gsm8k|0|true"]
        >>> results = run_lighteval("./checkpoint", tasks)
        >>> print(results["results"]["leaderboard"]["gsm8k"])
    """
    tracker = EvaluationTracker(output_dir="./le_results", save_details=False)

    pipe_params = PipelineParameters(
        launcher_type=ParallelismManager.ACCELERATE,
        env_config=None,
    )

    model_cfg = TransformersModelConfig(
        model_name=checkpoint_path,
        dtype="float16",
        use_chat_template=True,
        device_map="cpu",
    )

    pipeline = Pipeline(
        tasks=",".join(tasks),           # e.g. "leaderboard|gsm8k|0|true"
        pipeline_parameters=pipe_params,
        evaluation_tracker=tracker,
        model_config=model_cfg,
    )

    # returns a nested dict with all scores
    return pipeline.evaluate()



class LightEvalCallback(TrainerCallback):
    """
    Custom callback for running LightEval evaluations during training.
    
    This callback periodically evaluates the model on specified tasks using
    the LightEval framework and logs the results to Weights & Biases.
    
    Attributes:
        tasks (list): List of task specifications for evaluation
        freq (int): Frequency of evaluation (every N evaluations)
        count (int): Counter for tracking evaluation calls
    """
    
    def __init__(self, tasks, freq=2):
        """
        Initialize the LightEval callback.
        
        Args:
            tasks (list): List of task specifications for evaluation
            freq (int, optional): Frequency of evaluation. Defaults to 2.
        """
        self.tasks, self.freq = tasks, freq
        self.count = 0

    def on_evaluate(self, args, state, control, **kw):
        """
        Called after each evaluation step during training.
        
        Runs LightEval evaluation every N evaluations and logs results to W&B.
        
        Args:
            args: Training arguments
            state: Training state
            control: Training control object
            **kw: Additional keyword arguments including 'model' and 'tokenizer'
        """
        self.count += 1
        if self.count % self.freq:
            return                      # only every Nth HF eval

        model, tok = kw["model"], kw["tokenizer"]

        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            tok.save_pretrained(tmp)

            res = run_lighteval(tmp, self.tasks)

        flat = {
            f"lighteval/{t}/{m}": v
            for t, metrics in res["results"].items()
            for m, v in metrics.items()
        }
        wandb.log(flat, step=state.global_step)

# disable quantization for now
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=False,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
# )

light_tasks = [
    "leaderboard|truthfulqa:mc|0|0",
    "leaderboard|gsm8k|0|true",
]
# callbacks = [LightEvalCallback(light_tasks, freq=1000000)]
callbacks = []



model_dir = config["model_dir"]

tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
tokenizer.model_max_length = config["max_sequence_length"]  # 4096 currently. Could try 2048 for safer memory usage, original was 50000; but problematic for memory usage

# Set pad_token if not already set (Qwen models use eos_token as pad_token)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Detect if running in distributed mode (accelerate/DDP)
# When using accelerate launch with multiple GPUs, WORLD_SIZE > 1
is_distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1

logging.info(f"Distributed training mode: {is_distributed}")

# device_map='auto' conflicts with DDP - only use it for single GPU
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    # quantization_config=bnb_config, disable quantization for now
    device_map="auto" if not is_distributed else None,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

model.config.use_cache = False
model.config.pretraining_tp = 1

# System message for Qwen chat template
SYSTEM_MESSAGE = "You are a scientist with advanced knowledge in philosophy and social sciences. Please, write the next paragraph for the following text."

logging.info("Loading preprocessed datasets:")

def load_preprocessed_dataset(filepath: str) -> datasets.Dataset:
    """
    Load preprocessed examples from gzipped JSONL file.

    Expected format: {"context": "...", "response": "...", ...}

    Args:
        filepath: Path to gzipped JSONL file

    Returns:
        datasets.Dataset with "context" and "response" columns
    """
    examples = []
    line_count = 0

    logging.info(f"  Opening {filepath}...")
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            try:
                example = json.loads(line)
                examples.append({
                    "context": example.get("context", ""),
                    "response": example.get("response", ""),
                })
                line_count += 1

                # Log progress every 1000 lines
                if line_count % 1000 == 0:
                    logging.info(f"  Loaded {line_count} examples from {filepath}")

            except json.JSONDecodeError as e:
                logging.warning(f"  Line {line_num}: JSON decode error: {e}")
                continue

    logging.info(f"  Loaded {line_count} total examples from {filepath}")

    return datasets.Dataset.from_dict({
        "context": [ex["context"] for ex in examples],
        "response": [ex["response"] for ex in examples],
    })


def get_formatted_cache_path(preprocessed_path: str) -> str:
    """
    Derive the formatted cache directory path from preprocessed file path.

    Converts path like:
      /path/to/v0.61-preprocessed/train.jsonl.gz
    To:
      /path/to/v0.61-formatted/train

    Args:
        preprocessed_path: Path to the preprocessed .jsonl.gz file

    Returns:
        Path to the formatted cache directory
    """
    path = Path(preprocessed_path)

    # Get the parent directory (e.g., /path/to/v0.61-preprocessed)
    parent = path.parent

    # Get the base filename without extensions (e.g., train.jsonl.gz -> train)
    base_name = path.name.replace('.jsonl.gz', '').replace('.jsonl', '')

    # Replace 'preprocessed' with 'formatted' in the parent directory name
    formatted_parent = str(parent).replace('-preprocessed', '-formatted')

    # Construct the cache path
    cache_path = Path(formatted_parent) / base_name

    return str(cache_path)


def load_or_create_formatted_dataset(preprocessed_path: str, is_train: bool = True) -> datasets.Dataset:
    """
    Load formatted dataset from cache, or create and cache it if not exists.

    This function implements a two-tier caching strategy:
    1. Checks for pre-formatted dataset on disk (instant loading)
    2. If not found, loads JSONL, formats with HF caching, and saves to disk

    Args:
        preprocessed_path: Path to the preprocessed .jsonl.gz file
        is_train: Whether this is training data (for logging purposes)

    Returns:
        Formatted dataset ready for training
    """
    cache_path = get_formatted_cache_path(preprocessed_path)
    force_recreate = config.get("force_recreate_cache", False)
    dataset_type = "train" if is_train else "eval"

    # Check if formatted cache exists and we're not forcing recreation
    if Path(cache_path).exists() and not force_recreate:
        logging.info(f"✓ Found formatted {dataset_type} cache at {cache_path}")
        logging.info(f"  Loading formatted {dataset_type} dataset from cache (instant)...")
        ds = datasets.load_from_disk(cache_path)
        logging.info(f"  Loaded {len(ds)} formatted {dataset_type} examples from cache")
        return ds

    # Cache doesn't exist or forced recreation - need to format
    if force_recreate:
        logging.info(f"Force recreate enabled - regenerating {dataset_type} cache")
    else:
        logging.info(f"No formatted {dataset_type} cache found at {cache_path}")

    logging.info(f"Loading and formatting {dataset_type} dataset from {preprocessed_path}")

    # Load the preprocessed JSONL
    ds = load_preprocessed_dataset(preprocessed_path)

    # Create cache directory for HuggingFace's internal .map() cache
    hf_cache_dir = Path(cache_path).parent / ".hf_cache"
    hf_cache_dir.mkdir(parents=True, exist_ok=True)

    # Apply formatting with explicit HF caching
    logging.info(f"  Applying chat template formatting to {dataset_type} dataset...")
    ds = ds.map(
        formatting_prompts_func,
        batched=True,
        desc=f"Formatting {dataset_type}",
        cache_file_name=str(hf_cache_dir / f"{dataset_type}_formatted.arrow"),
    )

    # Apply truncation with explicit HF caching
    logging.info(f"  Truncating long prompts in {dataset_type} dataset...")
    ds = ds.map(
        truncate_long_prompts,
        batched=True,
        desc=f"Truncating {dataset_type}",
        cache_file_name=str(hf_cache_dir / f"{dataset_type}_truncated.arrow"),
    )

    # Save the formatted dataset to disk for future runs
    logging.info(f"  Saving formatted {dataset_type} dataset to {cache_path}")
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(cache_path)
    logging.info(f"✓ Formatted {dataset_type} dataset cached successfully ({len(ds)} examples)")

    return ds


# Load train and eval datasets from preprocessed paths
train_path = config.get("preprocessed_train_path")
eval_path = config.get("preprocessed_eval_path")

if not train_path or not eval_path:
    raise ValueError("Config must specify preprocessed_train_path and preprocessed_eval_path")

logging.info("=" * 80)
logging.info("DATASET LOADING AND CACHING")
logging.info("=" * 80)

# Load or create formatted datasets (with automatic caching)
logging.info(f"Loading/formatting train data from {train_path}")
dataset = load_or_create_formatted_dataset(train_path, is_train=True)

logging.info(f"Loading/formatting eval data from {eval_path}")
eval_dataset_mapped = load_or_create_formatted_dataset(eval_path, is_train=False)

logging.info(f"Datasets ready: train={len(dataset)}, eval={len(eval_dataset_mapped)}")
logging.info("=" * 80)

EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN

style_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

logging.info("Encoding eval dataset for style similarity (this may take several minutes):")

sample_train_texts = [ex["response"]
                      for ex, _ in zip(eval_dataset_mapped, range(256))]   # take first 256

logging.info(f"  Encoding {len(sample_train_texts)} sample responses...")
style_bank = style_encoder.encode(
    sample_train_texts,
    batch_size=128,
    normalize_embeddings=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

logging.info(f"  Encoded {len(sample_train_texts)} samples. Style bank shape: {style_bank.shape}")



class LLMSampleCB(WandbCallback):
    """
    Custom callback for logging sample predictions during training.
    
    This callback generates sample predictions during evaluation and logs
    them to Weights & Biases along with style similarity metrics.
    
    Attributes:
        sample_dataset: Dataset to sample from for generation
        model: The language model
        tokenizer: The tokenizer
        chunk_size (int): Batch size for generation
        gen_config: Generation configuration
    """
    
    def __init__(self, trainer, test_dataset, chunk_size=4,    num_samples = 32, max_new_tokens=256, log_model="checkpoint"):
        """
        Initialize the LLM sample callback.
        
        Args:
            trainer: The trainer object
            test_dataset: Dataset to sample from
            chunk_size (int, optional): Batch size for generation. Defaults to 4.
            num_samples (int, optional): Number of samples to generate. Defaults to 32.
            max_new_tokens (int, optional): Maximum new tokens to generate. Defaults to 256.
            log_model (str, optional): Model logging strategy. Defaults to "checkpoint".
        """
        super().__init__()
        # self._log_model = log_model
        self.sample_dataset = test_dataset.take(num_samples)
        self.model, self.tokenizer = trainer.model, trainer.tokenizer
        self.chunk_size = chunk_size
        self.gen_config = GenerationConfig.from_pretrained(trainer.model.name_or_path,
                                                           max_new_tokens=max_new_tokens)

    def generate(self, prompt):
        """
        Generate text from a single prompt.
        
        Args:
            prompt (str): Input prompt for generation
            
        Returns:
            str: Generated text
        """
        tokenized_prompt = self.tokenizer(prompt, return_tensors='pt')['input_ids'].cuda()
        with torch.inference_mode():
            output = self.model.generate(tokenized_prompt, generation_config=self.gen_config)
        return self.tokenizer.decode(output[0][len(tokenized_prompt[0]):], skip_special_tokens=True)

    @torch.inference_mode()
    def _generate_chunk(self, prompts):
        """
        Generate text for a batch of prompts.
        
        Args:
            prompts (List[str]): List of input prompts
            
        Returns:
            List[str]: List of generated texts
        """
        tok = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        outs = self.model.generate(**tok, generation_config=self.gen_config)
        return self.tokenizer.batch_decode(
            outs[:, tok["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

    def samples_table(self):
        """
        Create a W&B table with sample predictions and style similarity metrics.

        Generates predictions for sample prompts, computes style similarity
        against a style bank, and creates a table for logging.

        Returns:
            wandb.Table: Table containing prompts, generations, and metrics
        """
        prompts = [ex["context"] for ex in self.sample_dataset]
        gens = []  # will collect all generations

        for start in range(0, len(prompts), self.chunk_size):
            sub = prompts[start:start + self.chunk_size]
            gens.extend(self._generate_chunk(sub))

        gen_emb = []
        for start in trange(0, len(gens), self.chunk_size):
            sub = gens[start:start + self.chunk_size]
            gen_emb.extend(
                style_encoder.encode(
                    sub,
                    batch_size=self.chunk_size,
                    normalize_embeddings=True,
                    device="cuda",
                )
            )
        gen_emb = torch.tensor(gen_emb, device="cuda")  # (N, 384)

        sims = util.cos_sim(gen_emb, torch.tensor(style_bank, device="cuda"))
        max_sims = sims.max(dim=1).values.cpu().numpy()

        cols = ["prompt", "generation"] \
               + list(self.gen_config.to_dict().keys()) \
               + ["style_sim_mean", "style_sim_std"]
        table = wandb.Table(columns=cols)
        cfg_vals = list(self.gen_config.to_dict().values())

        for p, g, s in zip(prompts, gens, max_sims):
            table.add_data(p, g, *cfg_vals, float(s), 0.0)

        return table

    def on_evaluate(self, args, state, control, **kwargs):
        """
        Called after each evaluation step to log sample predictions.
        
        Args:
            args: Training arguments
            state: Training state
            control: Training control object
            **kwargs: Additional keyword arguments
        """
        super().on_evaluate(args, state, control, **kwargs)
        self._wandb.log(
            {"sample_predictions": self.samples_table()},
            step=state.global_step,
        )
    # def on_evaluate(self, args, state, control, **kwargs):
    #     "Log the wandb.Table after calling trainer.evaluate"
    #     super().on_evaluate(args, state, control, **kwargs)
    #     records_table = self.samples_table(self.sample_dataset)
    #     self._wandb.log({"sample_predictions": records_table})




def formatting_prompts_func(examples):
    """
    Format context-response pairs into Qwen chat template prompts.

    Converts preprocessed chunks into formatted prompts using Qwen's
    chat template with system message.

    Args:
        examples (dict): Dictionary containing "context" and "response" keys

    Returns:
        dict: Dictionary with formatted "text" key (chat template output)

    Examples:
        >>> examples = {"context": ["Text..."], "response": ["Continuation..."]}
        >>> result = formatting_prompts_func(examples)
        >>> print(result["text"][0])
    """
    contexts = examples["context"]
    responses = examples["response"]
    texts = []

    for context, response in zip(contexts, responses):
        # Build messages for Qwen chat template
        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": context},
            {"role": "assistant", "content": response},
        ]

        # Apply Qwen's chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Truncate if necessary (with proper truncation to avoid warnings)
        tokens = tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=config["max_eval_tok"]
        )
        # Decode back to text to ensure consistent formatting
        text = tokenizer.decode(tokens, skip_special_tokens=True)

        texts.append(text)

    return {"text": texts}


def truncate_long_prompts(batch):
    """
    Truncate prompts that exceed the maximum token limit.

    NOTE: This function is now redundant since truncation already happens
    in formatting_prompts_func. Kept for backwards compatibility.

    Args:
        batch (dict): Batch containing "text" key with prompts

    Returns:
        dict: Batch with truncated prompts

    Examples:
        >>> batch = {"text": ["very long prompt..."]}
        >>> result = truncate_long_prompts(batch)
    """

    trimmed = []
    for txt in batch["text"]:                 # txt is a string
        tokens = tokenizer.encode(
            txt,
            add_special_tokens=False,
            truncation=True,
            max_length=config["max_eval_tok"]
        )
        txt = tokenizer.decode(tokens, skip_special_tokens=True)
        trimmed.append(txt)
    return {"text": trimmed}


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)


logging.info("Loading model:")
# LoRA config
peft_config = LoraConfig(
    lora_alpha=16,                           # Scaling factor for LoRA
    lora_dropout=0.05,                       # Add slight dropout for regularization
    r=64,                                    # Rank of the LoRA update matrices
    bias="none",                             # No bias reparameterization
    task_type="CAUSAL_LM",                   # Task type: Causal Language Modeling
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],  # Target modules for LoRA
)

model = get_peft_model(model, peft_config)


# Batch size configuration
# For 2x A100 (141GB total), these are conservative settings
# With DDP, effective batch size = per_device_batch_size * num_gpus * gradient_accumulation_steps
batch_size = 4
num_gpus = torch.cuda.device_count()
effective_batch_size = batch_size * num_gpus * 2  # 2 = gradient_accumulation_steps
logging.info(f"Batch configuration: per_device={batch_size}, num_gpus={num_gpus}, grad_accum=2")
logging.info(f"  → Effective batch size: {effective_batch_size}")
logging.info(f"  → To increase throughput on 2x A100, try per_device_batch_size=8 (requires ~60GB total)")

steps = int(500000/effective_batch_size)

eval_dataset = eval_dataset_mapped.take(128)

logging.info("Model loaded. Building training arguments.")
eval_every = int(0.1 * steps)

output_dir = config["output_dir"]

# Training Arguments
training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
    optim="adamw_torch",  # Changed from paged_adamw_32bit to avoid bitsandbytes dependency
    num_train_epochs=1,
    logging_steps=0.01,
    warmup_steps=10,
    max_steps = steps,
    logging_strategy="steps",
    learning_rate=2e-4,
    fp16=False,
    bf16=False,
    report_to=["wandb"],
    eval_strategy="steps",
    eval_steps= eval_every,
    save_steps= 1000,
    disable_tqdm=False,
    # Data loading optimization - prevents GPU stalls
    dataloader_num_workers=4,           # Use 4 CPU processes for data loading
    dataloader_pin_memory=True,         # Pin memory for faster GPU transfer
    # DDP optimization - disable unused parameter detection for performance
    ddp_find_unused_parameters=False,   # All LoRA params are used; avoid extra overhead
    # predict_with_generate=True,
    # generation_max_length=128
)


logging.info("Building Trainer...")

# Initialize the Trainer
trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=dataset,
    peft_config=peft_config,
    data_collator=data_collator,
    callbacks=callbacks,
    eval_dataset=eval_dataset,
    # compute_metrics=compute_metrics
)
logging.info("Trainer built successfully")

# wandb_callback = LLMSampleCB(trainer, eval_dataset, chunk_size = 2, num_samples=16, max_new_tokens=256)
# trainer.add_callback(wandb_callback)

logging.info("Running initial evaluation...")
trainer.evaluate()
logging.info("Initial evaluation complete")

logging.info("Clearing cache and preparing for training...")
gc.collect()
torch.cuda.empty_cache()
model.config.use_cache = False

logging.info("=" * 80)
logging.info("STARTING TRAINING")
logging.info(f"Total steps: {steps}, Eval every {eval_every} steps, Save every 1000 steps")
logging.info(f"Batch size: per_device={batch_size}, effective={effective_batch_size} (with {num_gpus} GPU(s))")
logging.info(f"Learning rate: {training_arguments.learning_rate}")
logging.info(f"Data loading: {training_arguments.dataloader_num_workers} workers, pin_memory={training_arguments.dataloader_pin_memory}")
if num_gpus > 1:
    logging.info(f"✓ Using Distributed Data Parallel (DDP) with {num_gpus} GPUs")
logging.info("=" * 80)

trainer.train()

final_model_path = "{}/final_model/".format(output_dir)
trainer.save_model(final_model_path)

output_dir_last = "{}/".format(output_dir)

# Disable B2 uploading for now
# load_dotenv()
# uploader = B2Uploader()
# uploader.upload_file(final_model_path, "{}_{}".format(config["model_dir"], int(time.time())))

wandb.finish()