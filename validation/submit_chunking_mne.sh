#!/bin/bash
#SBATCH --job-name=chunking-mne
#SBATCH --account=def-kjerbi_gpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/chunking-mne-%j.out
#SBATCH --error=logs/chunking-mne-%j.err
#SBATCH --chdir=/home/sesma/amica-python

source /home/sesma/envs/amica/bin/activate
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=4

python -u validation/validate_chunking_mne.py
