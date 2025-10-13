#!/bin/bash
cd "$(dirname "$0")"
# eval "$(/home/$(whoami)/anaconda3/bin/conda shell.bash hook)"
eval "$(/home/dhonchar/anaconda3/bin/conda shell.bash hook)" # use Dan's enviornment to avoid replicating our own

source "/home/$(whoami)/.bashrc"
export WANDB_PROJECT=eval_model

conda activate eval_model

$@
