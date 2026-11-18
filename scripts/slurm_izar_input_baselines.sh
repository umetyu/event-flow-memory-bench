#!/usr/bin/env bash
#SBATCH --job-name=efbench-input
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --array=0-1%2
#SBATCH --output=efbench-input-%A_%a.log
set -euo pipefail
module load gcc/11.3.0 python/3.10.4 cuda/11.8.0
cd "${EFBENCH_ROOT:-${SLURM_SUBMIT_DIR:?}}"
: "${EFBENCH_PYTHON:?Set the existing Python environment with PyTorch CUDA}"
export PYTHONPATH="$PWD/src:$PWD/external/runtime-deps${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
i="${SLURM_ARRAY_TASK_ID:?Submit as an array}"
[[ "$i" == 0 || "$i" == 1 ]] || exit 2
names=(count voxel5)
args=(--index "$i" --threads "$OMP_NUM_THREADS")
prefix=input-baselines
if [[ "${EFBENCH_INPUT_SMOKE:-0}" == 1 ]]; then
    prefix=input-smoke
    args+=(--smoke)
    if [[ "$i" == 0 ]]; then
        srun "$EFBENCH_PYTHON" -m unittest discover -s tests -v
    fi
elif [[ "${EFBENCH_INPUT_RESUME:-0}" == 1 ]]; then
    : "${EFBENCH_INPUT_RUN_ROOT:?Set the previous array run root for resuming}"
    args+=(--resume)
fi
run_root="${EFBENCH_INPUT_RUN_ROOT:-runs/${prefix}-${SLURM_ARRAY_JOB_ID:?}}"
srun "$EFBENCH_PYTHON" -u scripts/run_input_baseline.py "${args[@]}" --output "$run_root/${names[$i]}"
