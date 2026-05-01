#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

FT_CONFIG="configs/train_sd_controlnet_reloblur_lr5e-7_2epoch_b160.yaml"
EVAL_CONFIG="configs/evaluation_sd_controlnet_predmask_reloblur.yaml"
FT_DIR="output/training/reloblur_gtmask_weightedloss_lr5e-7_10epoch_b4_acc4"
LOG_DIR="output/training/fullflow_resume_ft_eval_logs"
MANIFEST_DIR="output/datasets/reloblur_real"
mkdir -p "$LOG_DIR"

CURRENT_STAGE="wait_manifest"

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
  local dest="incomplete/${stamp}_resume_fullflow_${CURRENT_STAGE}_failed"
  mkdir -p "$dest"
  if [ "$CURRENT_STAGE" = "finetune" ] && [ -e "$FT_DIR" ]; then
    mv "$FT_DIR" "$dest/"
  fi
  cp -r "$LOG_DIR" "$dest/logs"
  echo "Moved incomplete resume-fullflow artifacts to $dest due to ${CURRENT_STAGE} failure (exit $status)"
  exit "$status"
}
trap on_error ERR

echo "Resume fullflow FT+eval started at $(date -Is)"
snapshot "before"

python - <<'PY'
import json
import time
from pathlib import Path

manifest_dir = Path("output/datasets/reloblur_real")
train_manifest = manifest_dir / "train_manifest.json"
test_manifest = manifest_dir / "test_manifest.json"
deadline = time.time() + 3600
while time.time() < deadline:
    if train_manifest.is_file() and test_manifest.is_file():
        try:
            train = json.loads(train_manifest.read_text(encoding="utf-8")).get("samples", [])
            test = json.loads(test_manifest.read_text(encoding="utf-8")).get("samples", [])
        except Exception:
            time.sleep(10)
            continue
        if len(train) > 0 and len(test) > 0:
            print(f"manifests ready: train={len(train)} test={len(test)}")
            break
    print("waiting for ReLoBlur manifests...")
    time.sleep(10)
else:
    raise TimeoutError("ReLoBlur manifests were not ready within 1 hour")
PY

CURRENT_STAGE="finetune"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=online \
python scripts/train_sd_controlnet_coco.py \
  --config "$FT_CONFIG" \
  --local-files-only
snapshot "after_finetune"

CURRENT_STAGE="infra_eval"
WANDB_MODE=disabled \
python -m evaluation.eval_pipeline \
  --model SD15-TileControlNet-ReLoBlur-PredMask-FullFlow \
  --round reloblur_predmask_fullflow_infra_eval \
  --dataset reloblur-test \
  --count 5 \
  --mode standard \
  --detailed true \
  --config "$EVAL_CONFIG" \
  --model-type sd_controlnet \
  --visual-limit 5
snapshot "after_eval"

echo "Resume fullflow FT+eval completed at $(date -Is)"
