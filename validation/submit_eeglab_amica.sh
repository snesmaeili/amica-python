#!/bin/bash
#SBATCH --job-name=eeglab-amica
#SBATCH --account=def-kjerbi_gpu
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/eeglab-amica-%j.out
#SBATCH --error=logs/eeglab-amica-%j.err
#SBATCH --chdir=/home/sesma/amica-python

module load StdEnv/2020 matlab/2023a.3
ulimit -s unlimited

echo "=== EEGLAB AMICA on sub-01 ==="
echo "MATLAB: $(which matlab)"
echo "EEGLAB: /home/sesma/refs/eeglab2026.0.0"
echo

matlab -nodisplay -nosplash -r "try; run('validation/run_eeglab_amica.m'); catch e; disp(e.message); end; exit"
