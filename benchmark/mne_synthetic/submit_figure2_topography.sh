#!/bin/bash
#SBATCH --job-name=amica_fig2_topography
#SBATCH --account=def-kjerbi_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/home/sesma/scratch/amica_fig2_topography_%j.out
#SBATCH --error=/home/sesma/scratch/amica_fig2_topography_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source fir_env_synthetic.sh

export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export AMICA_FIGURE2_RESULTS_DIR="${AMICA_FIGURE2_RESULTS_DIR:-/scratch/$USER/amica_figure2_topography}"
mkdir -p "$AMICA_FIGURE2_RESULTS_DIR"

python run_figure2_topography.py \
    --config configs/benchmark_v1.json \
    --output-dir "$AMICA_FIGURE2_RESULTS_DIR"

