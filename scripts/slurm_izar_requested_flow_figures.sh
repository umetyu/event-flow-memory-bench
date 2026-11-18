#!/usr/bin/env bash
#SBATCH --job-name=efbench-requested-figures
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=efbench-requested-figures-%j.log
set -euo pipefail
module load gcc/11.3.0 python/3.10.4 cuda/11.8.0
cd "${EFBENCH_ROOT:-${SLURM_SUBMIT_DIR:?}}"
: "${EFBENCH_PYTHON:?}"
: "${EFBENCH_MVSEC_MANIFEST:?}"
: "${EFBENCH_FLOW_REFERENCE:?Verified ConvGRU figure directory}"
: "${EFBENCH_ZHU_CHECKPOINT:?Completed Zhu 2019 reimplementation best.pt}"
export PYTHONPATH="$PWD/src:$PWD/external/runtime-deps${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
srun "$EFBENCH_PYTHON" -u scripts/plot_requested_flow_comparison.py \
  --reference "$EFBENCH_FLOW_REFERENCE" --checkpoint "$EFBENCH_ZHU_CHECKPOINT" \
  --manifest "$EFBENCH_MVSEC_MANIFEST" --output "runs/requested-flow-figures-${SLURM_JOB_ID:?}" \
  --device cuda --threads "$OMP_NUM_THREADS"
