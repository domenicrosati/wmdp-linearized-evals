#!/bin/bash
#SBATCH --job-name=wmdp-syndi
#SBATCH --account=aip-frudzicz
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

module load arrow/21.0.0

export PYTHONPATH="$HOME/wmdp-linearized-evals/src"
export HF_HOME="$HOME/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

PYTHON=/scratch/d/domenic/dii-paper-smoke/venv/bin/python
CHECKPOINT=/scratch/d/domenic/synthetic-di-full-contraction-v7-20261111/checkpoint
OUTPUT="$HOME/wmdp-linearized-evals/runs/synthetic-di-v7"

mkdir -p "$OUTPUT"
"$PYTHON" -m wmdp_linearized.cli all \
  --model "$CHECKPOINT" \
  --model-label synthetic-di-v7 \
  --output-dir "$OUTPUT" \
  --trust-remote-code \
  --dtype float32 \
  --batch-size-questions 4 \
  --rank 512
