#!/bin/bash

################################################################################
# Evaluation Settings

# Model name/path
MODEL="ConditionalLocalDeblurNet-task010"

# Test round identifier
# Results will be saved to: results/<ROUND_NAME>_<MODEL>_<DATASET>_<COUNT>_<MMDD>[_HHMM]/
ROUND_NAME="task011_validation"

# Dataset name
DATASET="synthetic5k-val"

# Test cases (0 = full validation split)
COUNT=100

# Evaluation mode
# Options:
# - standard: Direct local deblurring evaluation
# - thinking: Reserved for detailed reasoning reports in compatible workflows
MODE="standard"

# Detailed output verbosity (true/false)
DETAILED="true"

# Trained PyTorch checkpoint from task-010
CHECKPOINT="output/training/final_baseline_task010/best.pt"

# Synthetic grouped local-blur manifest
MANIFEST="output/datasets/coco2017_train_grouped_localblur_5k/manifest.json"


################################################################################
# Run evaluation

python -m evaluation.eval_pipeline \
 --model "$MODEL" \
 --round "$ROUND_NAME" \
 --dataset "$DATASET" \
 --count "$COUNT" \
 --mode "$MODE" \
 --detailed "$DETAILED" \
 --checkpoint "$CHECKPOINT" \
 --manifest "$MANIFEST" \
 --split val \
 --split-seed 2026 \
 --val-fraction 0.1 \
 --image-size 256
