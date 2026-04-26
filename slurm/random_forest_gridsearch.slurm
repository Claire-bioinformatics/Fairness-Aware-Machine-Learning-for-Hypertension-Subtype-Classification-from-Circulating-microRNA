#!/bin/bash
#SBATCH --job-name=rf_grid_1
#SBATCH --output=logs/rf_grid_1_%j.out
#SBATCH --error=logs/rf_grid_1_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --account=none    
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=3053994w@glasgow.ac.uk


cd "$SLURM_SUBMIT_DIR"


module load apps/miniforge/24.7.1/bin


eval "$(conda shell.bash hook)"
conda activate xgb_env

echo "Host: $(hostname)"
echo "PWD:  $(pwd)"
python --version
which python

echo "Job started at: $(date)"
start=$(date +%s)

python rf_grid_1.py

end=$(date +%s)
echo "Job finished at: $(date)"
echo "Elapsed time: $((end-start)) seconds"
