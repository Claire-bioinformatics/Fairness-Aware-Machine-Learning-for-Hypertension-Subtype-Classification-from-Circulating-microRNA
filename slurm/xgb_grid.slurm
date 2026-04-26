#!/bin/bash
#SBATCH --job-name=xgb_grid
#SBATCH --output=logs/xgb_grid_%j.out
#SBATCH --error=logs/xgb_grid_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --account=none
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=3053994w@glasgow.ac.uk

# Load Miniforge
module load apps/miniforge/24.7.1/bin


conda activate xgb_env

# Log start
echo "Job started at: $(date)"
start=$(date +%s)

# Run the Python script
python xgb_grid.py

# Log end
end=$(date +%s)
echo "Job finished at: $(date)"
echo "Elapsed time: $((end-start)) seconds"


