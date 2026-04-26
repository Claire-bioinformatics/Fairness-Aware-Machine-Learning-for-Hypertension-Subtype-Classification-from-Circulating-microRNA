#!/bin/bash
#SBATCH --job-name=rf_optuna_2
#SBATCH --output=logs/rf_optuna_2_%j.out
#SBATCH --error=logs/rf_optuna_2_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --account=none
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=3053994w@glasgow.ac.uk


cd "$SLURM_SUBMIT_DIR"

# Ensure logs dir exists
mkdir -p logs


module load apps/miniforge/24.7.1/bin
eval "$(conda shell.bash hook)"
conda activate xgb_env


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
import sys, optuna, sklearn
print("Python exe:", sys.executable)
print("Optuna:", optuna.__version__)
print("scikit-learn:", sklearn.__version__)
PY

echo "Job started at: $(date)"
start=$(date +%s)


python -u rf_optuna_2.py

end=$(date +%s)
echo "Job finished at: $(date)"
echo "Elapsed time: $((end-start)) seconds"

