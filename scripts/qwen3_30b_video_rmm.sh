#!/bin/bash -l
#SBATCH -J qwen3_30b_video_rmm
#SBATCH -p mit_preemptable
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH -t 24:00:00
#SBATCH --requeue
#SBATCH --signal=TERM@120
#SBATCH -o logs/qwen3_30b_video_rmm_%j.out
#SBATCH -e logs/qwen3_30b_video_rmm_%j.err
#
# NOTE: #SBATCH directives are NOT shell-expanded by sbatch, so paths/partition
# above must be literals. Override at submit time on the command line, e.g.:
#   sbatch -p gpu -o ~/logs/qwen3_30b_video_%j.out scripts/qwen3_30b_video_rmm.sh
# Output paths are relative to the submission directory: submit from the repo root.

echo "Job started on $(hostname) at $(date)"

set -eo pipefail

# REPO_DIR must point at your sails-vlm checkout (no default: personal paths
# are banned by the SAILS repo schema).
if [ -z "${REPO_DIR:-}" ]; then
  echo "ERROR: export REPO_DIR=/path/to/your/sails-vlm clone before sbatch." >&2
  exit 1
fi
cd "${REPO_DIR}"
mkdir -p logs

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CONFIG=${CONFIG:-configs/qwen3/rmm_30b_video.yaml}

uv run sails-vlm-predict "${CONFIG}" "$@"
