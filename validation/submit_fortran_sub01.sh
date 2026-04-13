#!/bin/bash
#SBATCH --job-name=fortran-sub01
#SBATCH --account=def-kjerbi_gpu
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/fortran-sub01-%j.out
#SBATCH --error=logs/fortran-sub01-%j.err
#SBATCH --chdir=/home/sesma/amica-python

# Run the original Fortran amica15ub on the SAME preprocessed sub-01
# data that Python uses. Direct comparison of algorithm behavior.
# No GPU needed — amica15ub is CPU-only (statically linked).
#
# Previous run (59273066) failed with buffer overflow at 16G mem.
# Increased to 32G and unlimited stack.

echo "=== Fortran amica15ub on sub-01 preprocessed data ==="
echo "Binary: /home/sesma/refs/sccn-amica/amica15ub"
echo "Param:  validation/results/post_f1_audit/sub01_fortran.param"
echo

ulimit -s unlimited
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_STACKSIZE=512M

/home/sesma/refs/sccn-amica/amica15ub validation/results/post_f1_audit/sub01_fortran.param
