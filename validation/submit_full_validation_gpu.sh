#!/bin/bash
#SBATCH --job-name=amica-full-gpu
#SBATCH --account=def-kjerbi_gpu
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=1-25
#SBATCH --output=logs/full-gpu-%A_%a.out
#SBATCH --error=logs/full-gpu-%A_%a.err
#SBATCH --chdir=/home/sesma/amica-python

# Job array runs one subject per task in parallel.
# Default: 1 subject × 1 HP filter × 3 methods (no FastICA), MAX_ITER=2000.
# The prior 2 h / 2 HP / 4 methods config TIMED OUT on all 25 tasks (job
# 58939673). 8 h + HP=1.0 + {amica,picard,infomax} is the sustainable floor.
# Override any of these via `sbatch --export=...` or shell env before sbatch.

mkdir -p logs

module load cuda/12.6
source /home/sesma/envs/amica/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export DS_PATH="${DS_PATH:-/home/sesma/scratch/ds004505}"
export MAX_ITER="${MAX_ITER:-2000}"
export HP_FILTERS="${HP_FILTERS:-1.0}"
export METHODS="${METHODS:-amica,picard,infomax}"
export ICLEAN="${ICLEAN:-none}"
export RESULTS_SUBDIR="${RESULTS_SUBDIR:-}"

SUBJECT=$(printf "sub-%02d" $SLURM_ARRAY_TASK_ID)
export SUBJECTS=$SUBJECT

echo "=== Task $SLURM_ARRAY_TASK_ID: $SUBJECT ==="
echo "MAX_ITER=$MAX_ITER  HP_FILTERS=$HP_FILTERS  METHODS=$METHODS"
echo "ICLEAN=$ICLEAN  RESULTS_SUBDIR=${RESULTS_SUBDIR:-<default>}"
python -u validation/run_highdens_validation.py
