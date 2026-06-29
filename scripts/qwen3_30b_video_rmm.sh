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

source ~/.bashrc || true
conda activate "${CONDA_ENV:-qwen}"

set -eo pipefail

REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "${REPO_DIR}"
mkdir -p logs
export PYTHONPATH="${PWD}:${PYTHONPATH}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CONFIG=${CONFIG:-configs/qwen3/rmm_30b_video.yaml}

python -m runners.run_prediction "${CONFIG}" "$@"
