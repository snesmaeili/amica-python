#!/bin/bash
#SBATCH --job-name=compare-20iter-gpu
#SBATCH --account=def-kjerbi_gpu
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/compare-20iter-gpu-%j.out
#SBATCH --error=logs/compare-20iter-gpu-%j.err
#SBATCH --chdir=/home/sesma/amica-python

module load cuda/12.6
source /home/sesma/envs/amica/bin/activate
python -u validation/compare_20iter.py
