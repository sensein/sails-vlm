#!/bin/bash -l
#SBATCH -J qwen3_30b_video_rmm
#SBATCH -p ${SLURM_PARTITION:-mit_preemptable}
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH -t 24:00:00
#SBATCH --requeue
#SBATCH --signal=TERM@120
#SBATCH -o ${HOME}/logs/qwen3_30b_video_rmm_%j.out
#SBATCH -e ${HOME}/logs/qwen3_30b_video_rmm_%j.err

echo "Job started on $(hostname) at $(date)"

source ~/.bashrc || true
conda activate "${CONDA_ENV:-qwen}"

set -eo pipefail

REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
LOGS_DIR="${LOGS_DIR:-${HOME}/logs}"
mkdir -p "${LOGS_DIR}"

cd "${REPO_DIR}"
export PYTHONPATH="${PWD}:${PYTHONPATH}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CONFIG=${CONFIG:-configs/qwen3/rmm_30b_video.yaml}

python -m runners.run_prediction "${CONFIG}" "$@"
