#!/bin/bash
#SBATCH --job-name=fortran17-sub01
#SBATCH --account=def-kjerbi_gpu
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/fortran17-sub01-%j.out
#SBATCH --error=logs/fortran17-sub01-%j.err
#SBATCH --chdir=/home/sesma/amica-python

# Run amica17 (compiled from source on Narval) on the SAME preprocessed
# sub-01 data Python uses. Definitive Fortran vs Python comparison.

module load gcc/12.3.0 openmpi/4.1.5 flexiblas/3.3.1

ulimit -s unlimited
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_STACKSIZE=512M

echo "=== Fortran amica17 (compiled) on sub-01 ==="
echo "Binary: /home/sesma/refs/sccn-amica/amica17_narval"
echo "Param:  validation/results/post_f1_audit/sub01_fortran.param"
echo

/home/sesma/refs/sccn-amica/amica17_narval validation/results/post_f1_audit/sub01_fortran.param
