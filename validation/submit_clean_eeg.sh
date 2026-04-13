#!/bin/bash
#SBATCH --job-name=clean-eeg-test
#SBATCH --account=def-kjerbi_gpu
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/clean-eeg-test-%j.out
#SBATCH --error=logs/clean-eeg-test-%j.err
#SBATCH --chdir=/home/sesma/amica-python
module load cuda/12.6
source /home/sesma/envs/amica/bin/activate
python -u validation/test_clean_eeg.py
