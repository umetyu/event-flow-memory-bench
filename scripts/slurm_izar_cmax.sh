#!/usr/bin/env bash
#SBATCH --job-name=efbench-cmax
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2-23:00:00
#SBATCH --array=0-31%4
#SBATCH --output=efbench-cmax-%A_%a.log
set -euo pipefail
module load gcc/11.3.0 python/3.10.4 cuda/11.8.0
cd "${EFBENCH_ROOT:-${SLURM_SUBMIT_DIR:?}}"
: "${EFBENCH_PYTHON:?Set CMax-specific Python environment}"
: "${EFBENCH_MVSEC_MANIFEST:?Set existing MVSEC manifest}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLBACKEND=Agg
i="${SLURM_ARRAY_TASK_ID:?}"
((i >= 0 && i < 32)) || exit 2
methods=(patch2018 secrets2022)
profiles=(matched1k long)
sequences=(outdoor_day1 indoor_flying1 indoor_flying2 indoor_flying3)
method="${methods[$((i/16))]}"
profile="${profiles[$(((i/8)%2))]}"
sequence="${sequences[$(((i/2)%4))]}"
dt=$((1+3*(i%2)))
args=(--method "$method" --profile "$profile" --sequence "$sequence" --dt "$dt"
      --manifest "$EFBENCH_MVSEC_MANIFEST" --threads "$OMP_NUM_THREADS")
prefix=cmax
if [[ "${EFBENCH_CMAX_SMOKE:-0}" == 1 ]]; then
  prefix=cmax-smoke
  args+=(--limit 2)
  if [[ "$i" == 0 ]]; then
    export EFBENCH_SECRETS_CHECKOUT="$PWD/external/secrets-event-flow"
    srun "$EFBENCH_PYTHON" -m unittest discover -s tests -v
  fi
elif [[ "${EFBENCH_CMAX_RESUME:-0}" == 1 ]]; then
  : "${EFBENCH_CMAX_RUN_ROOT:?Set previous run root}"
  args+=(--resume)
fi
run_root="${EFBENCH_CMAX_RUN_ROOT:-runs/${prefix}-${SLURM_ARRAY_JOB_ID:?}}"
srun "$EFBENCH_PYTHON" -u scripts/run_cmax.py "${args[@]}" --output "$run_root/$method-$profile-$sequence-dt$dt"
