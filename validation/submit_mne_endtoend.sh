#!/bin/bash
#SBATCH --job-name=mne-endtoend
#SBATCH --account=def-kjerbi_gpu
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/mne-endtoend-%j.out
#SBATCH --error=logs/mne-endtoend-%j.err
#SBATCH --chdir=/home/sesma/amica-python

module load cuda/12.6
source /home/sesma/envs/amica/bin/activate
export OMP_NUM_THREADS=4

python -u validation/test_mne_endtoend.py
