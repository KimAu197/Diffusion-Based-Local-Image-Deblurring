#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

LOG_DIR="output/training/stage3_nafnet_urgent_fullweighted_b12_logs"
FT_DIR="output/training/stage3_nafnet_reloblur_fullweighted_urgent_b12_ft"
FT_CONFIG="configs/train_stage3_nafnet_reloblur_fullweighted_urgent_b12.yaml"
EVAL_CONFIG="configs/evaluation_stage3_nafnet_fullweighted_urgent_b12_reloblur.yaml"
mkdir -p "$LOG_DIR"

screen -dmS stage3_urgent_ft_b12 bash -lc '
set -euo pipefail
CURRENT_STAGE="startup"
snapshot() {
  local name="$1"
  python - "$name" "'"$LOG_DIR"'" <<'"'"'PY'"'"'
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
  local dest="incomplete/${stamp}_stage3_urgent_fullweighted_b12_${CURRENT_STAGE}_failed"
  mkdir -p "$dest"
  if [ -e "'"$FT_DIR"'" ]; then
    mv "'"$FT_DIR"'" "$dest/"
  fi
  cp -r "'"$LOG_DIR"'" "$dest/logs"
  echo "Moved incomplete urgent Stage3 b12 artifacts to $dest due to ${CURRENT_STAGE} failure (exit $status)"
  exit "$status"
}
trap on_error ERR

echo "Urgent Stage3 fullweighted b12 FT started at $(date -Is)"
snapshot "before"
CURRENT_STAGE="finetune"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=online \
python scripts/train_sd_controlnet_coco.py \
  --config "'"$FT_CONFIG"'" \
  --local-files-only
snapshot "after_finetune"

CURRENT_STAGE="infra_eval"
WANDB_MODE=disabled \
python -m evaluation.eval_pipeline \
  --model "SD15-TileControlNet-Stage3-Urgent-FullWeighted-B12" \
  --round "reloblur_stage3_urgent_fullweighted_b12_infra" \
  --dataset reloblur-test \
  --count 5 \
  --mode standard \
  --detailed true \
  --config "'"$EVAL_CONFIG"'" \
  --model-type sd_controlnet \
  --visual-limit 5
snapshot "after_eval"
echo "Urgent Stage3 fullweighted b12 FT completed at $(date -Is)"
' 2>&1 | tee "$LOG_DIR/screen.log"

echo "Launched screen session: stage3_urgent_ft_b12"
echo "Log: $LOG_DIR/screen.log"
