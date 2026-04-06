#!/bin/bash
#SBATCH --job-name=amica-full-v3
#SBATCH --account=def-kjerbi_cpu
#SBATCH --time=08:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

module load python/3.11
source /home/sesma/envs/amica/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export DS_PATH="/home/sesma/scratch/ds004505"

cd /home/sesma/amica-python
python -u validation/run_highdens_validation.py
