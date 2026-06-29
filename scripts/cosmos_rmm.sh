#!/bin/bash -l
#SBATCH -J cosmos_rmm
#SBATCH -p pi_satra
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH --requeue
#SBATCH --signal=TERM@120
#SBATCH -o logs/cosmos_rmm_%j.out
#SBATCH -e logs/cosmos_rmm_%j.err
#
# NOTE: #SBATCH directives are NOT shell-expanded by sbatch, so paths/partition
# above must be literals. Override at submit time on the command line, e.g.:
#   sbatch -p gpu -o ~/logs/cosmos_%j.out scripts/cosmos_rmm.sh
# Output paths are relative to the submission directory: submit from the repo root.

echo "Job started on $(hostname) at $(date)"

source ~/.bashrc || true
conda activate "${CONDA_ENV:-qwen}"

set -eo pipefail

# Path to your sails-vlm checkout. Edit this (or export REPO_DIR) before running.
REPO_DIR="${REPO_DIR:-/orcd/data/satra/001/users/brukew/sails-vlm}"
cd "${REPO_DIR}"
mkdir -p logs
export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Stay offline; assumes model weights already cached.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CONFIG=${CONFIG:-configs/cosmos/rmm.yaml}

python -m runners.run_prediction "${CONFIG}" "$@"
