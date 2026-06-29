#!/bin/bash -l
#SBATCH -J qwen3_30b_rmm
#SBATCH -p mit_preemptable
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH -t 4:30:00
#SBATCH --requeue
#SBATCH --signal=TERM@120
#SBATCH -o /orcd/data/satra/001/users/brukew/logs/qwen3_30b_rmm_%j.out
#SBATCH -e /orcd/data/satra/001/users/brukew/logs/qwen3_30b_rmm_%j.err

echo "Job started on $(hostname) at $(date)"

source ~/.bashrc || true
conda activate qwen

set -eo pipefail

cd /orcd/data/satra/001/users/brukew/sails-vlm
export PYTHONPATH="${PWD}:${PYTHONPATH}"

mkdir -p /orcd/data/satra/001/users/brukew/logs

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CONFIG=${CONFIG:-configs/qwen3/rmm_30b.yaml}

python -m runners.run_prediction "${CONFIG}" "$@"
