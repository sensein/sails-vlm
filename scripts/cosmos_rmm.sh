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

set -eo pipefail

# REPO_DIR must point at your sails-vlm checkout (no default: personal paths
# are banned by the SAILS repo schema).
if [ -z "${REPO_DIR:-}" ]; then
  echo "ERROR: export REPO_DIR=/path/to/your/sails-vlm clone before sbatch." >&2
  exit 1
fi
cd "${REPO_DIR}"
mkdir -p logs

# Stay offline; assumes model weights already cached.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CONFIG=${CONFIG:-configs/cosmos/rmm.yaml}

uv run sails-vlm-predict "${CONFIG}" "$@"
