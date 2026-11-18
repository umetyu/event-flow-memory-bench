#!/usr/bin/env bash
#SBATCH --job-name=efbench-flow-figures
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=efbench-flow-figures-%j.log
set -euo pipefail
module load gcc/11.3.0 python/3.10.4 cuda/11.8.0
cd "${EFBENCH_ROOT:-${SLURM_SUBMIT_DIR:?}}"
: "${EFBENCH_PYTHON:?Python environment with torch CUDA and matplotlib}"
: "${EFBENCH_MVSEC_MANIFEST:?Existing MVSEC manifest}"
: "${EFBENCH_AUTHOR_RUN:?Verified author ConvGRU MLflow run directory}"
: "${EFBENCH_VOXEL_CHECKPOINT:?Our trained voxel5 best.pt}"
export PYTHONPATH="$PWD/src:$PWD/external/runtime-deps${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLBACKEND=Agg
srun "$EFBENCH_PYTHON" -m unittest discover -s tests -p test_flow_visualization.py -v
srun "$EFBENCH_PYTHON" -u scripts/plot_flow_comparison.py \
  --manifest "$EFBENCH_MVSEC_MANIFEST" --author-run "$EFBENCH_AUTHOR_RUN" \
  --voxel-checkpoint "$EFBENCH_VOXEL_CHECKPOINT" --trust-author-pickle \
  --output "runs/flow-figures-${SLURM_JOB_ID:?}" --anchor-events 500000 --threads "$OMP_NUM_THREADS"
