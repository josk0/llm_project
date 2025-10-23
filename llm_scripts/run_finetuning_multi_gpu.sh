#!/bin/bash

# get conda and shell
eval "$(/home/jrhimmel/anaconda3/bin/conda shell.bash hook)"
source "/home/$(whoami)/.bashrc"

# Configuration - EDIT THESE
PROJECT_PATH="/home/jrhimmel/workspace/llm_project/llm_scripts"
LOG_DIR="${PROJECT_PATH}/logs_finetuning"

# WandB setup
export WANDB_PROJECT=finetune-llm

# Create log directory
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/finetuning.out"
#LOG_FILE="$LOG_DIR/finetuning_$(date +%Y%m%d_%H%M%S).log"

echo "================================================" | tee "$LOG_FILE"
echo "Starting Multi-GPU Finetuning" | tee -a "$LOG_FILE"
echo "Project: $PROJECT_PATH" | tee -a "$LOG_FILE"
echo "Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "================================================" | tee -a "$LOG_FILE"

# Activate virtual environment
conda activate llm_finetuning_new

# Change to project directory
cd "$PROJECT_PATH" || exit 1

# Detect available GPUs
echo "Detecting GPUs..." | tee -a "$LOG_FILE"
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
" 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "Starting training with accelerate launch..." | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Run with accelerate (automatically detects and uses all available GPUs)
accelerate launch llm_scripts/finetuning_skip.py 2>&1 | tee -a "$LOG_FILE"

RETURN_CODE=$?

echo "" | tee -a "$LOG_FILE"
echo "================================================" | tee -a "$LOG_FILE"
if [ $RETURN_CODE -eq 0 ]; then
    echo "✓ Training completed successfully" | tee -a "$LOG_FILE"
else
    echo "✗ Training failed with return code $RETURN_CODE" | tee -a "$LOG_FILE"
fi
echo "================================================" | tee -a "$LOG_FILE"

exit $RETURN_CODE
