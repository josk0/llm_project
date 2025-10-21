#!/bin/bash
set -euo pipefail
mkdir -p logs


# Need to convert GPU device names to integers to avoid a ValueError in Olmocr
normalize_cvd_to_indices() {
  local cvd="${CUDA_VISIBLE_DEVICES:-}"
  [[ -n "$cvd" ]] || return 0
  if [[ "$cvd" =~ ^[0-9]+([,][0-9]+)*$ ]]; then return 0; fi

  mapfile -t GPU_ROWS < <(nvidia-smi --query-gpu=index,uuid,pci.bus_id --format=csv,noheader)
  trim() { echo "$1" | awk '{$1=$1;print}'; }

  IFS=',' read -ra TOKENS <<< "$cvd"
  local resolved=()
  for raw in "${TOKENS[@]}"; do
    token="$(trim "$raw")"; token_upper="${token^^}"
    if [[ "$token" =~ ^[0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]$ ]]; then
      found=""
      for row in "${GPU_ROWS[@]}"; do
        idx="$(trim "$(echo "$row" | cut -d',' -f1)")"
        bus="$(trim "$(echo "$row" | cut -d',' -f3)")"
        if [[ "${bus^^}" == "${token_upper}" ]]; then found="$idx"; break; fi
      done
      [[ -n "$found" ]] || { echo "Could not map PCI bus $token to an index"; exit 1; }
      resolved+=("$found"); continue
    fi
    if [[ "$token_upper" == GPU-* || "$token_upper" == MIG-GPU-* ]]; then
      found=""
      for row in "${GPU_ROWS[@]}"; do
        idx="$(trim "$(echo "$row" | cut -d',' -f1)")"
        uuid="$(trim "$(echo "$row" | cut -d',' -f2)")"; uuid_upper="${uuid^^}"
        if [[ "$uuid_upper" == "$token_upper" || "$uuid_upper" == "$token_upper"* || "$token_upper" == "$uuid_upper"* ]]; then
          found="$idx"; break
        fi
      done
      [[ -n "$found" ]] || { echo "Could not map GPU UUID $token to an index"; printf '  %s\n' "${GPU_ROWS[@]}"; exit 1; }
      resolved+=("$found"); continue
    fi
    echo "Unrecognized CUDA_VISIBLE_DEVICES token: $token"; exit 1
  done
  export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${resolved[*]}")"
}

normalize_cvd_to_indices
# This was only for debugging the A100s
# export VLLM_DEFAULT_DTYPE="${VLLM_DEFAULT_DTYPE:-bf16}"

# Container
# The default model is an FP8 model. In OrganeGrid, this works on the L40S nodes 
#   I could not get Olmocr to work on the A100 nodes, even when I passed the FP16 model, Flash Attention was a problem
#   To manually pass the FP16 model use: --model allenai/olmOCR-7B-0825 \
# Note: first argument is the workspace, then the pdfs
#  todo: probably should use a "neutral" folder for the workspace, not the output folder?
apptainer exec --nv /home/jrhimmel/containers/olmocr_latest.sif \
python -m olmocr.pipeline /home/jrhimmel/datasets/zotero-library/v0/documents \
  --pdfs /home/jrhimmel/datasets/zotero-library/v0/documents/*/*.pdf \
  --markdown