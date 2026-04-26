#!/bin/bash
#SBATCH --job-name=cat_grid_1                # Job name
#SBATCH --output=logs/cat_grid_1_%j.out      # Standard output log
#SBATCH --error=logs/cat_grid_1_%j.err       # Standard error log
#SBATCH --ntasks=1                           # Number of tasks
#SBATCH --cpus-per-task=8                    # Number of CPU cores
#SBATCH --mem=32G                            # Memory
#SBATCH --time=20:00:00                      # Time limit (hh:mm:ss)
#SBATCH --account=none                       
#SBATCH --mail-type=BEGIN,END,FAIL           
#SBATCH --mail-user=3053994w@gla.ac.uk          

# Load conda (miniforge) module
module load apps/miniforge/24.7.1/bin


conda activate xgb_env

# Go to the project directory
cd ~/cat_gridsearch_project

# Log start time
echo "Job started at: $(date)"
start=$(date +%s)

# Run your Python script
python cat_grid_1.py

# Log end time
end=$(date +%s)
echo "Job finished at: $(date)"
echo "Elapsed time: $((end - start)) seconds"

