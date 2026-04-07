#!/bin/bash
#SBATCH --job-name=amica-full-gpu
#SBATCH --account=def-kjerbi_gpu
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=1-25
#SBATCH --output=logs/full-gpu-%A_%a.out
#SBATCH --error=logs/full-gpu-%A_%a.err
#SBATCH --chdir=/home/sesma/amica-python

# Job array runs one subject per task in parallel.
# Each task: 1 subject × 2 HP filters × 4 methods, MAX_ITER=2000, ~25 min on GPU.

mkdir -p logs

module load cuda/12.6
source /home/sesma/envs/amica/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export DS_PATH="/home/sesma/scratch/ds004505"
export MAX_ITER=2000
export HP_FILTERS="1.0,2.0"

SUBJECT=$(printf "sub-%02d" $SLURM_ARRAY_TASK_ID)
export SUBJECTS=$SUBJECT

echo "=== Task $SLURM_ARRAY_TASK_ID: $SUBJECT ==="
echo "MAX_ITER=$MAX_ITER  HP_FILTERS=$HP_FILTERS"
python -u validation/run_highdens_validation.py
