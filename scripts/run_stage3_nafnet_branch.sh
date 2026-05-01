#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

BRANCH="${1:?Usage: scripts/run_stage3_nafnet_branch.sh <fullweighted|maskonly>}"

case "$BRANCH" in
  fullweighted)
    PRETRAIN_CONFIG="configs/train_stage3_nafnet_coco_fullweighted.yaml"
    FT_CONFIG="configs/train_stage3_nafnet_reloblur_fullweighted.yaml"
    EVAL_CONFIG="configs/evaluation_stage3_nafnet_fullweighted_reloblur.yaml"
    PRETRAIN_DIR="output/training/stage3_nafnet_coco_fullweighted_pretrain"
    FT_DIR="output/training/stage3_nafnet_reloblur_fullweighted_ft"
    EVAL_MODEL="SD15-TileControlNet-Stage3-NAFNet-FullWeighted"
    EVAL_ROUND="reloblur_stage3_nafnet_fullweighted_infra"
    ;;
  maskonly)
    PRETRAIN_CONFIG="configs/train_stage3_nafnet_coco_maskonly.yaml"
    FT_CONFIG="configs/train_stage3_nafnet_reloblur_maskonly.yaml"
    EVAL_CONFIG="configs/evaluation_stage3_nafnet_maskonly_reloblur.yaml"
    PRETRAIN_DIR="output/training/stage3_nafnet_coco_maskonly_pretrain"
    FT_DIR="output/training/stage3_nafnet_reloblur_maskonly_ft"
    EVAL_MODEL="SD15-TileControlNet-Stage3-NAFNet-MaskOnly"
    EVAL_ROUND="reloblur_stage3_nafnet_maskonly_infra"
    ;;
  *)
    echo "Unknown branch: $BRANCH"
    exit 2
    ;;
esac

LOG_DIR="output/training/stage3_nafnet_${BRANCH}_fullflow_logs"
mkdir -p "$LOG_DIR"

CURRENT_STAGE="startup"

snapshot() {
  local name="$1"
  python - "$name" "$LOG_DIR" <<'PY'
import json
import os
import subprocess
import sys
from pathlib import Path

name = sys.argv[1]
log_dir = Path(sys.argv[2])
smi = subprocess.run(
    ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
    capture_output=True,
    text=True,
    timeout=10,
)
disk = subprocess.run(["df", "-h", "/root/autodl-tmp"], capture_output=True, text=True, timeout=10)
payload = {
    "stage": name,
    "cpu_count": os.cpu_count(),
    "nvidia_smi": smi.stdout.strip(),
    "disk_autodl_tmp": disk.stdout.strip(),
}
(log_dir / f"resource_{name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
PY
}

on_error() {
  local status="$?"
  local stamp
  stamp="$(date +%Y%m%d_%H%M%S)"
  local dest="incomplete/${stamp}_stage3_nafnet_${BRANCH}_${CURRENT_STAGE}_failed"
  mkdir -p "$dest"
  if [ "$CURRENT_STAGE" = "pretrain" ] && [ -e "$PRETRAIN_DIR" ]; then
    mv "$PRETRAIN_DIR" "$dest/"
  elif [ "$CURRENT_STAGE" = "finetune" ] && [ -e "$FT_DIR" ]; then
    mv "$FT_DIR" "$dest/"
  fi
  cp -r "$LOG_DIR" "$dest/logs"
  echo "Moved incomplete stage3 ${BRANCH} artifacts to $dest due to ${CURRENT_STAGE} failure (exit $status)"
  exit "$status"
}
trap on_error ERR

echo "Stage3 NAFNet ${BRANCH} fullflow started at $(date -Is)"
snapshot "before"

CURRENT_STAGE="pretrain"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=online \
python scripts/train_sd_controlnet_coco.py \
  --config "$PRETRAIN_CONFIG" \
  --local-files-only
snapshot "after_pretrain"

CURRENT_STAGE="finetune"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=online \
python scripts/train_sd_controlnet_coco.py \
  --config "$FT_CONFIG" \
  --local-files-only
snapshot "after_finetune"

CURRENT_STAGE="infra_eval"
WANDB_MODE=disabled \
python -m evaluation.eval_pipeline \
  --model "$EVAL_MODEL" \
  --round "$EVAL_ROUND" \
  --dataset reloblur-test \
  --count 5 \
  --mode standard \
  --detailed true \
  --config "$EVAL_CONFIG" \
  --model-type sd_controlnet \
  --visual-limit 5
snapshot "after_eval"

echo "Stage3 NAFNet ${BRANCH} fullflow completed at $(date -Is)"
