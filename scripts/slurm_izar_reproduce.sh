#!/usr/bin/env bash
#SBATCH --job-name=efbench-reproduce
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --array=0-3%2
#SBATCH --output=efbench-reproduce-%A_%a.log
set -euo pipefail
module load gcc/11.3.0 python/3.10.4 cuda/11.8.0
cd "${EFBENCH_ROOT:-${SLURM_SUBMIT_DIR:?}}"
: "${EFBENCH_PYTHON:?Set the Python environment with PyTorch CUDA}"
: "${EFBENCH_MVSEC:?Set the directory containing the four author HDF5 recordings}"
: "${EFBENCH_MLRUNS:?Set the verified author archive experiment directory (e.g. mlruns 2/42)}"
export PYTHONPATH="$PWD/src:$PWD/external/runtime-deps${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
names=(FireNet FireNet EVFlowNet EVFlowNet)
dts=(1 4 1 4)
i="${SLURM_ARRAY_TASK_ID:?Submit as an array}"
(( i >= 0 && i < 4 )) || exit 2
out="runs/reproduce-${SLURM_ARRAY_JOB_ID:?}/${names[$i]}-dt${dts[$i]}"
# Archive folder EVFlowNet contains the RecEVFlowNet architecture.
# Only use the following legacy-pickle flag with a verified author distribution.
srun "$EFBENCH_PYTHON" -u -m event_flow_bench.cli --threads "$OMP_NUM_THREADS" reproduce \
    --data "$EFBENCH_MVSEC" --run-dir "$EFBENCH_MLRUNS/${names[$i]}" \
    --dt "${dts[$i]}" --device cuda --trust-legacy-pickle --output "${out}.json"
"$EFBENCH_PYTHON" -m event_flow_bench.cli compare-paper "${out}.json" \
    --output "${out}-comparison.json"
