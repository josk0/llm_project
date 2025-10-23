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

echo "================================================" 
echo "Starting Multi-GPU Finetuning" 
echo "Project: $PROJECT_PATH" 
echo "Log: $LOG_FILE" 
echo "================================================" 

# Activate virtual environment
conda activate llm_finetuning_new

# Change to project directory
cd "$PROJECT_PATH" || exit 1

# Detect available GPUs
echo "Detecting GPUs..." 
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
" 2>&1  

echo ""  
echo "Starting training with accelerate launch..."  
echo ""  

# Run with accelerate (automatically detects and uses all available GPUs)
accelerate launch llm_scripts/finetuning_skip.py 2>&1  

RETURN_CODE=$?

echo ""  
echo "================================================"  
if [ $RETURN_CODE -eq 0 ]; then
    echo "✓ Training completed successfully"  
else
    echo "✗ Training failed with return code $RETURN_CODE"  
fi
echo "================================================"  

exit $RETURN_CODE
