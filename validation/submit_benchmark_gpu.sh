#!/bin/bash
#SBATCH --job-name=ica-bench-gpu
#SBATCH --account=def-kjerbi_gpu
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

module load python/3.11
source /home/sesma/envs/amica/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export DS_PATH="/home/sesma/scratch/ds004505"

# Let JAX auto-detect the A100 GPU
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MAX_ITER="${MAX_ITER:-200}"

# Only run 1 Hz HP (skip the 2 Hz sweep for speed)
export HP_FILTERS="1.0"

cd /home/sesma/amica-python
echo "Starting GPU benchmark: $(date)"
echo "MAX_ITER=$MAX_ITER, GPU=$CUDA_VISIBLE_DEVICES"
nvidia-smi 2>/dev/null || echo "No nvidia-smi"
python -u validation/run_highdens_validation.py 2>&1
echo "Done: $(date)"
