#!/bin/bash
#SBATCH --job-name=ica-benchmark
#SBATCH --account=def-kjerbi_cpu
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

module load python/3.11
source /home/sesma/envs/amica/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export DS_PATH="/home/sesma/scratch/ds004505"

# Use CPU JAX (avoids CUDA issues on Narval CPU nodes)
export JAX_PLATFORMS=cpu

# Override MAX_ITER via: MAX_ITER=2000 sbatch submit_benchmark.sh
export MAX_ITER="${MAX_ITER:-500}"

cd /home/sesma/amica-python
echo "Starting benchmark: $(date)"
echo "MAX_ITER=$MAX_ITER, SUBJECTS=${SUBJECTS:-sub-01}"
python -u validation/run_highdens_validation.py 2>&1
echo "Done: $(date)"
