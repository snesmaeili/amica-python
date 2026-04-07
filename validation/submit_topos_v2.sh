#!/bin/bash
#SBATCH --job-name=amica-topos-v2
#SBATCH --account=def-kjerbi_cpu
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

module load python/3.11
source /home/sesma/envs/amica/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export DS_PATH="/home/sesma/scratch/ds004505"

cd /home/sesma/amica-python
python -u validation/plot_topos_v2.py
