#!/bin/bash
#SBATCH --job-name=amica-audit-post-f1
#SBATCH --account=def-kjerbi_gpu
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/audit-post-f1-%j.out
#SBATCH --error=logs/audit-post-f1-%j.err
#SBATCH --chdir=/home/sesma/amica-python

# Validate F1 (lrate state machine) + F2 (invsigmin=1e-8) on the
# production failure mode: sub-01, 2000 iters, num_mix=3.
#
# This calls Amica.fit() DIRECTLY on preprocessed data, bypassing
# the MNE ICA wrapper and the validation pipeline's metric machinery.
# Pure solver-vs-solver comparison with the saved pre-F1 baseline.
#
# Pre-F1 baseline (validation/results/amica_result.pkl):
#   67/90 rho at floor, MIR=-6.13, converged=False, 2-step LL-delta cycle
#
# Time: pre-F1 took 72s for 2000 iters on A100. 1h is plenty.
mkdir -p logs

module load cuda/12.6
source /home/sesma/envs/amica/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export DS_PATH="/home/sesma/scratch/ds004505"

echo "=== F1+F2 direct-fit audit on sub-01 ==="
echo "  git HEAD:  $(git log --oneline -1)"
echo "  solver changes:"
git diff --stat amica_python/
echo

python -u validation/audit_post_f1_direct.py
