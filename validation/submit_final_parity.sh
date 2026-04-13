#!/bin/bash
#SBATCH --job-name=final-parity
#SBATCH --account=def-kjerbi_gpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/final-parity-%j.out
#SBATCH --error=logs/final-parity-%j.err
#SBATCH --chdir=/home/sesma/amica-python

# Explicit library paths for compiled amica17_narval
export LD_LIBRARY_PATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcc12/openmpi/4.1.5/lib:/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/hwloc/2.9.1/lib:/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/flexiblascore/3.3.1/lib64:/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/lib64:/cvmfs/soft.computecanada.ca/gentoo/2023/x86-64-v3/usr/lib64:${LD_LIBRARY_PATH}

source /home/sesma/envs/amica/bin/activate
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=4

python -u validation/validate_parity.py
