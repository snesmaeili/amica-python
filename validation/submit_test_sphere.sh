#!/bin/bash
#SBATCH --job-name=test-sphere
#SBATCH --account=def-kjerbi_gpu
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/test-sphere-%j.out
#SBATCH --error=logs/test-sphere-%j.err
#SBATCH --chdir=/home/sesma/amica-python
module load cuda/12.6
source /home/sesma/envs/amica/bin/activate
python -u validation/test_fortran_sphere.py
