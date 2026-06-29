#!/bin/bash -l
#SBATCH -J qwen3_rmm
#SBATCH -p pi_satra
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH --requeue
#SBATCH --signal=TERM@120
#SBATCH -o /orcd/data/satra/001/users/brukew/logs/qwen3_rmm_%j.out
#SBATCH -e /orcd/data/satra/001/users/brukew/logs/qwen3_rmm_%j.err

echo "Job started on $(hostname) at $(date)"

source ~/.bashrc || true
conda activate qwen

set -eo pipefail

cd /orcd/data/satra/001/users/brukew/sails-vlm
export PYTHONPATH="${PWD}:${PYTHONPATH}"

mkdir -p /orcd/data/satra/001/users/brukew/logs

# Stay offline; assumes model weights already cached.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CONFIG=${CONFIG:-configs/qwen3/rmm.yaml}

python -m runners.run_prediction "${CONFIG}" "$@"
