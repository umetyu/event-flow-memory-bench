#!/usr/bin/env bash
#SBATCH --job-name=efbench-timing
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=efbench-timing-%j.log
set -euo pipefail
module load gcc/11.3.0 python/3.10.4 cuda/11.8.0
cd "${EFBENCH_ROOT:-${SLURM_SUBMIT_DIR:?}}"
: "${EFBENCH_PYTHON:?Set an existing Python environment with PyTorch CUDA}"
: "${EFBENCH_UZH:?Set the author_training directory}"
export PYTHONPATH="$PWD/src:$PWD/external/runtime-deps${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
srun "$EFBENCH_PYTHON" -u scripts/benchmark_upstream.py \
    --data "$EFBENCH_UZH" --models FireNet RecEVFlowNet \
    --warmup 20 --updates 100 --threads "$OMP_NUM_THREADS" \
    --output "runs/timing-${SLURM_JOB_ID:?}"
