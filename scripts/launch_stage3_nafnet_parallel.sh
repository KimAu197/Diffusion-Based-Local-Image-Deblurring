#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p \
  output/training/stage3_nafnet_fullweighted_fullflow_logs \
  output/training/stage3_nafnet_maskonly_fullflow_logs

screen -dmS stage3_nafnet_fullweighted bash -lc 'scripts/run_stage3_nafnet_branch.sh fullweighted 2>&1 | tee output/training/stage3_nafnet_fullweighted_fullflow_logs/screen.log'
screen -dmS stage3_nafnet_maskonly bash -lc 'scripts/run_stage3_nafnet_branch.sh maskonly 2>&1 | tee output/training/stage3_nafnet_maskonly_fullflow_logs/screen.log'

echo "Launched screen sessions: stage3_nafnet_fullweighted, stage3_nafnet_maskonly"
echo "Logs:"
echo "  output/training/stage3_nafnet_fullweighted_fullflow_logs/screen.log"
echo "  output/training/stage3_nafnet_maskonly_fullflow_logs/screen.log"
