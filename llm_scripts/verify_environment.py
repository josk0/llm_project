#!/usr/bin/env python
"""
Verify that the conda environment is correctly set up for multi-GPU finetuning.

Usage:
    python verify_environment.py

This script checks:
    - Python version
    - PyTorch installation and CUDA support
    - Number of GPUs and their memory
    - Required HuggingFace packages
    - Accelerate setup
    - Model availability
"""

import sys
from pathlib import Path

def print_header(title):
    """Print a formatted section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")

def check_python():
    """Check Python version."""
    print_header("Python Version")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")

    if version.major >= 3 and version.minor >= 10:
        print("✓ Python version is compatible (3.10+)")
        return True
    else:
        print("✗ Python version too old (need 3.10+)")
        return False

def check_torch():
    """Check PyTorch and CUDA availability."""
    print_header("PyTorch & CUDA")

    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")

        # Check CUDA
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")
        print(f"CUDA version: {torch.version.cuda}")

        if not cuda_available:
            print("✗ CUDA not available! Check nvidia-smi and conda CUDA installation")
            return False

        # Check GPU count
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs: {num_gpus}")

        if num_gpus == 0:
            print("✗ No GPUs detected!")
            return False

        # List GPUs
        print("\nGPU Details:")
        for i in range(num_gpus):
            name = torch.cuda.get_device_name(i)
            memory_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"  GPU {i}: {name} ({memory_gb:.1f} GB)")

        if num_gpus >= 2:
            print(f"\n✓ Multi-GPU setup detected ({num_gpus} GPUs)")
            return True
        else:
            print(f"\n⚠ Only {num_gpus} GPU detected (expect 2 for A100 setup)")
            return True  # Not critical, script works with 1 GPU

    except ImportError as e:
        print(f"✗ PyTorch not installed: {e}")
        return False
    except Exception as e:
        print(f"✗ Error checking PyTorch: {e}")
        return False

def check_transformers():
    """Check HuggingFace transformers."""
    print_header("HuggingFace Transformers")

    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")

        # Try to load a small model
        from transformers import AutoTokenizer
        print("✓ AutoTokenizer available")

        from transformers import AutoModelForCausalLM
        print("✓ AutoModelForCausalLM available")

        return True
    except ImportError as e:
        print(f"✗ Transformers not installed: {e}")
        return False
    except Exception as e:
        print(f"✗ Error checking transformers: {e}")
        return False

def check_datasets():
    """Check HuggingFace datasets."""
    print_header("HuggingFace Datasets")

    try:
        import datasets
        print(f"Datasets version: {datasets.__version__}")
        print("✓ Datasets library installed")
        return True
    except ImportError as e:
        print(f"✗ Datasets not installed: {e}")
        return False

def check_accelerate():
    """Check accelerate for distributed training."""
    print_header("Accelerate (Multi-GPU Support)")

    try:
        import accelerate
        print(f"Accelerate version: {accelerate.__version__}")

        # Check if accelerate can detect GPUs
        from accelerate import Accelerator
        accelerator = Accelerator()
        print(f"Accelerator device: {accelerator.device}")

        print("✓ Accelerate library installed and configured")
        return True
    except ImportError as e:
        print(f"✗ Accelerate not installed: {e}")
        print("   This is CRITICAL for multi-GPU training!")
        return False
    except Exception as e:
        print(f"✗ Error checking accelerate: {e}")
        return False

def check_peft():
    """Check PEFT (LoRA)."""
    print_header("PEFT (LoRA)")

    try:
        import peft
        print(f"PEFT version: {peft.__version__}")

        from peft import LoraConfig, get_peft_model
        print("✓ LoRA support available")
        return True
    except ImportError as e:
        print(f"✗ PEFT not installed: {e}")
        return False

def check_trl():
    """Check TRL (SFTTrainer)."""
    print_header("TRL (Supervised Finetuning Trainer)")

    try:
        import trl
        print(f"TRL version: {trl.__version__}")

        from trl import SFTTrainer
        print("✓ SFTTrainer available")
        return True
    except ImportError as e:
        print(f"✗ TRL not installed: {e}")
        return False

def check_wandb():
    """Check Weights & Biases."""
    print_header("Weights & Biases (Logging)")

    try:
        import wandb
        print(f"WandB version: {wandb.__version__}")
        print("✓ WandB installed")
        return True
    except ImportError as e:
        print(f"⚠ WandB not installed: {e}")
        print("  (Optional but recommended for monitoring)")
        return True  # Not critical

def check_model_access():
    """Check if we can access Qwen model."""
    print_header("Model Access (Qwen2.5-3B-Instruct)")

    try:
        import torch
        from transformers import AutoTokenizer

        model_name = "Qwen/Qwen2.5-3B-Instruct"
        print(f"Attempting to load: {model_name}")

        # Don't actually download, just check we can call the API
        # (requires internet and HF token, so we skip for now)
        print("⚠ Model access check skipped (requires internet + HF token)")
        print("  Run this to download the model:")
        print(f"  $ python -c \"from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('{model_name}')\"")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def check_config_files():
    """Check if config files exist."""
    print_header("Configuration Files")

    configs = {
        "environment.yml": Path("environment.yml"),
        "config_finetuning.json": Path("configs/config_finetuning.json"),
        "preprocess script": Path("llm_scripts/preprocess_markdown_chunks.py"),
        "finetuning script": Path("llm_scripts/finetuning_skip.py"),
    }

    all_exist = True
    for name, path in configs.items():
        if path.exists():
            print(f"✓ {name}: {path}")
        else:
            print(f"✗ {name}: {path} NOT FOUND")
            all_exist = False

    return all_exist

def main():
    """Run all checks."""
    print("\n" + "=" * 70)
    print("  ENVIRONMENT VERIFICATION FOR QWEN FINETUNING")
    print("=" * 70)

    checks = [
        ("Python", check_python),
        ("PyTorch & CUDA", check_torch),
        ("Transformers", check_transformers),
        ("Datasets", check_datasets),
        ("Accelerate", check_accelerate),
        ("PEFT", check_peft),
        ("TRL", check_trl),
        ("WandB", check_wandb),
        ("Config Files", check_config_files),
    ]

    results = {}
    for name, check_fn in checks:
        try:
            results[name] = check_fn()
        except Exception as e:
            print(f"✗ Unexpected error in {name}: {e}")
            results[name] = False

    # Summary
    print_header("Summary")

    critical = ["PyTorch & CUDA", "Transformers", "Accelerate", "PEFT", "TRL"]
    critical_ok = all(results.get(name, False) for name in critical)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    print(f"Checks passed: {passed}/{total}")
    print()

    if critical_ok:
        print("✓ ENVIRONMENT IS READY FOR FINETUNING")
        print("\nNext steps:")
        print("  1. Verify preprocessed data path in config_finetuning.json")
        print("  2. Run: accelerate launch llm_scripts/finetuning_skip.py")
        return 0
    else:
        print("✗ ENVIRONMENT HAS ISSUES")
        print("\nFailing checks:")
        for name, result in results.items():
            if not result and name in critical:
                print(f"  - {name}")
        print("\nTroubleshooting:")
        print("  1. Check CONDA_SETUP.md for detailed instructions")
        print("  2. Recreate environment: conda env create -f environment.yml")
        print("  3. Verify CUDA: nvidia-smi")
        return 1

if __name__ == "__main__":
    sys.exit(main())
