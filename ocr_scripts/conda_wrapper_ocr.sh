#!/bin/bash
cd "$(dirname "$0")"
eval "$(/home/$(whoami)/anaconda3/bin/conda shell.bash hook)"

source "/home/$(whoami)/.bashrc"

conda activate olmocr

python -m olmocr.pipeline /home/jrhimmel/tmpocrout --markdown --pdfs /home/jrhimmel/workspace/testpdfs/*.pdf
# $@ 
