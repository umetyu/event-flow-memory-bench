#!/usr/bin/env bash
#SBATCH --job-name=efbench-clear-flow
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:25:00
#SBATCH --output=efbench-clear-flow-%j.log
set -euo pipefail
module load gcc/11.3.0 python/3.10.4 cuda/11.8.0
cd "${EFBENCH_ROOT:-${SLURM_SUBMIT_DIR:?}}"
: "${EFBENCH_PYTHON:?}"
: "${EFBENCH_MVSEC_MANIFEST:?}"
: "${EFBENCH_AUTHOR_RUN:?}"
: "${EFBENCH_ZHU_CHECKPOINT:?}"
: "${EFBENCH_SCENE_CANDIDATES:?}"
export PYTHONPATH="$PWD/src:$PWD/external/runtime-deps${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
srun "$EFBENCH_PYTHON" -u scripts/clear_flow_figure.py infer \
  --manifest "$EFBENCH_MVSEC_MANIFEST" --candidates "$EFBENCH_SCENE_CANDIDATES" \
  --selection "$EFBENCH_SCENE_CANDIDATES/selection.json" --author-run "$EFBENCH_AUTHOR_RUN" \
  --checkpoint "$EFBENCH_ZHU_CHECKPOINT" --trust-author-pickle \
  --output "runs/clear-flow-${SLURM_JOB_ID:?}" --device cuda --threads "$OMP_NUM_THREADS"
