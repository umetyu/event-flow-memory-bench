#!/usr/bin/env bash
#SBATCH --job-name=ef-memory
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%A_%a.log
set -euo pipefail
# Set the account/partition/time for your cluster before submitting.
# Example (15 pilot jobs): sbatch --array=0-14 scripts/slurm_train.sh configs/generated/pilot/jobs.json
python scripts/run_grid.py "$1" --index "${SLURM_ARRAY_TASK_ID:?submit with --array}"
