#!/bin/bash
#SBATCH --job-name=cat_optuna_1               # Job name
#SBATCH --output=logs/cat_optuna_1_%j.out     # Standard output log
#SBATCH --error=logs/cat_optuna_1_%j.err      # Standard error log
#SBATCH --ntasks=1                            # Number of tasks
#SBATCH --cpus-per-task=8                     # CPU cores for CatBoost/BLAS
#SBATCH --mem=32G                             # Memory
#SBATCH --time=20:00:00                       # Time limit (hh:mm:ss)
#SBATCH --account=none                        
#SBATCH --mail-type=BEGIN,END,FAIL            
#SBATCH --mail-user=3053994w@gla.ac.uk


module load apps/miniforge/24.7.1/bin
conda activate xgb_env




cd ~/cat_gridsearch_project


mkdir -p logs



export OPTUNA_N_TRIALS=200


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK


export PYTHONHASHSEED=42


echo "Job started at: $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "CPUs: $SLURM_CPUS_PER_TASK | MEM: $SLURM_MEM_PER_NODE"

start=$(date +%s)


python cat_optuna_1.py

end=$(date +%s)
echo "Job finished at: $(date)"
echo "Elapsed time: $((end - start)) seconds"



