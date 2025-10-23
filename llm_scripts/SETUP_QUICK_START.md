# Quick Start: Conda Environment Setup (5 minutes)

## The 4 Commands You Need

### 1. Create Conda Environment
```bash
conda env create -f environment.yml
```
This downloads and installs PyTorch with CUDA 12.1, HuggingFace packages, and everything else.

**Time**: 5-10 minutes (first time only)

### 2. Activate Environment
```bash
conda activate llm_finetuning_new
```

### 3. Verify GPU Setup
```bash
python verify_environment.py
```

Should output:
```
✓ Number of GPUs: 2
✓ GPU 0: NVIDIA A100 PCIe (80.0 GB)
✓ GPU 1: NVIDIA A100 PCIe (80.0 GB)
✓ ENVIRONMENT IS READY FOR FINETUNING
```

### 4. Run Finetuning
```bash
condor_submit finetuning_multi_gpu.sub
```

---

## What If Something's Wrong?

### Problem: "CUDA not available"
```bash
# Recreate from scratch
conda env remove -n llm_finetuning_new
conda env create -f environment.yml
```

### Problem: "Only 1 GPU detected"
- Make sure you're using `accelerate launch`, NOT `python`
- Check HTCondor asks for 2 GPUs: `request_gpus = 2` in submit file

### Problem: "Module not found" (transformers, datasets, etc.)
```bash
# Update the environment
conda env update -f environment.yml --prune
```

### For detailed help
See [CONDA_SETUP.md](CONDA_SETUP.md)

---

## What's Different from requirements.txt?

| File | Use Case | When to Use |
|------|----------|-----------|
| `environment.yml` | **Conda users** (recommended for GPU) | 👈 **USE THIS** |
| `requirements.txt` | Pip-only users (no conda) | Only if you don't have conda |

**Key difference**:
- `environment.yml`: Handles CUDA automatically
- `requirements.txt`: Requires manual nvidia-cuda-* packages (error-prone)

---

## Files Reference

| File | Purpose |
|------|---------|
| `environment.yml` | Conda environment spec (use this!) |
| `requirements.txt` | Alternative for pip-only users |
| `verify_environment.py` | Check if setup is correct |
| `llm_scripts/run_finetuning_multi_gpu.sh` | Bash wrapper for HTCondor |
| `llm_scripts/finetuning_multi_gpu.subm` | HTCondor job submission file |

---

## Next Steps

1. ✅ Run: `conda env create -f environment.yml`
2. ✅ Run: `conda activate llm_finetuning`
3. ✅ Run: `python verify_environment.py`
4. ✅ Run: `condor_submit finetuning_multi_gpu.sub`

**That's it!** 🚀
