#!/bin/bash
#SBATCH --job-name=stack                      # Job name
#SBATCH --output=logs/stack_%j.out            # STDOUT log
#SBATCH --error=logs/stack_%j.err             # STDERR log
#SBATCH --ntasks=1                            # Number of tasks
#SBATCH --cpus-per-task=8                     # CPU cores
#SBATCH --mem=32G                             # Memory
#SBATCH --time=24:00:00                       # Time limit (hh:mm:ss)
#SBATCH --account=none                        
#SBATCH --mail-type=END,FAIL                  
#SBATCH --mail-user=3053994w@gla.ac.uk

# --- Env setup ---
module load apps/miniforge/24.7.1/bin         
conda activate xgb_env                        


export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

# --- Work dir ---
cd /users/3053994w/stack

# Ensure log/result folders exist
mkdir -p logs results

# --- Run ---
echo "Job started at: $(date)"
start=$(date +%s)

python -u stack.py

status=$?
end=$(date +%s)
echo "Job finished at: $(date) with status $status"
echo "Elapsed time: $((end - start)) seconds"
exit $status

