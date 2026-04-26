#!/bin/bash
#SBATCH --job-name=xgb_optuna
#SBATCH --output=logs/xgb_optuna_%j.out
#SBATCH --error=logs/xgb_optuna_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --account=none
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=3053994w@glasgow.ac.uk


cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Load Miniforge and activate env
module load apps/miniforge/24.7.1/bin
# (more robust activation for non-interactive shells)
eval "$(conda shell.bash hook)"
conda activate xgb_env

# Optuna controls + better logging
export OPTUNA_N_TRIALS=100        
export PYTHONUNBUFFERED=1         
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONHASHSEED=42


echo "Host: $(hostname)"
echo "PWD:  $(pwd)"
python --version
which python
python - <<'PY'
import sys, optuna, xgboost, sklearn
print("Python exe:", sys.executable)
print("Optuna:", optuna.__version__)
print("xgboost:", xgboost.__version__)
print("scikit-learn:", sklearn.__version__)
PY

echo "Job started at: $(date)"
start=$(date +%s)


python -u xgb_optuna.py

end=$(date +%s)
echo "Job finished at: $(date)"
echo "Elapsed time: $((end-start)) seconds"

