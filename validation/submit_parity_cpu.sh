#!/bin/bash
#SBATCH --job-name=parity-test
#SBATCH --account=def-kjerbi_gpu
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/parity-test-%j.out
#SBATCH --error=logs/parity-test-%j.err
#SBATCH --chdir=/home/sesma/amica-python
module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 flexiblas/3.3.1 2>/dev/null || module load gcc openmpi flexiblas 2>/dev/null || true
source /home/sesma/envs/amica/bin/activate
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=4
python -u validation/validate_parity.py
