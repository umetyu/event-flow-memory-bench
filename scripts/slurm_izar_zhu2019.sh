#!/usr/bin/env bash
#SBATCH --job-name=zhu2019-uzh
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=efbench-zhu2019-%j.log
set -euo pipefail
module load gcc/11.3.0 python/3.10.4 cuda/11.8.0
cd "${EFBENCH_ROOT:-${SLURM_SUBMIT_DIR:?}}"
: "${EFBENCH_PYTHON:?Set an existing Python environment with PyTorch CUDA}"
export PYTHONPATH="$PWD/src:$PWD/external/runtime-deps${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
run_dir="${EFBENCH_RUN:-runs/zhu2019-uzh-seed0-${SLURM_JOB_ID:?}}"
args=(--threads "$OMP_NUM_THREADS" train --config configs/uzh_zhu2019.yml --output "$run_dir")
if [[ -n "${EFBENCH_SMOKE_UPDATES:-}" ]]; then
    srun "$EFBENCH_PYTHON" -m unittest discover -s tests -v
    args+=(--max-updates "$EFBENCH_SMOKE_UPDATES")
elif [[ -n "${EFBENCH_RESUME:-}" ]]; then
    args+=(--resume "$EFBENCH_RESUME")
fi
srun "$EFBENCH_PYTHON" -u -m event_flow_bench.zhu2019 "${args[@]}"
if [[ -z "${EFBENCH_SMOKE_UPDATES:-}" ]]; then
    for dt in 1 4; do
        srun "$EFBENCH_PYTHON" -u -m event_flow_bench.zhu2019 --threads "$OMP_NUM_THREADS" evaluate \
            --checkpoint "$run_dir/best.pt" --manifest runs/setup/mvsec.json \
            --dt "$dt" --output "$run_dir/mvsec-dt${dt}.json"
    done
fi
