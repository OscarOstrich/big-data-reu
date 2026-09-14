#!/bin/bash
#SBATCH --job-name=GPU_Jupyter       # Job name 
#SBATCH --output=testgpu.out           # Output file name (%J expands to jobId)
#SBATCH --error=testgpu.err            # Error file name (%J expands to jobId)
#SBATCH --account=cybertrn                      # Account name
#SBATCH --cluster=chip-gpu                                  # specify gpu or cpu
#SBATCH --constraint=RTX_6000                   # Hardware constraint: [L40S, RTX_8000, RTX_6000, H100, RTX_2080Ti]
#SBATCH --mem=28GB                              # Job memory request
#SBATCH --gres=gpu:1                            # Number of requested GPU(s)
#SBATCH --time=1-12:00:00                      # Time limit days-hrs:min:sec


#This should take around 24-28 hours to run if CAMS data is not downloaded already. If you predict it will take more, please adjust the time ASAP.

#------------------------------------------
#   Activate a python environment
#-----------------------------------------
module purge

echo "Loading python environment..."
module load Anaconda3/2024.02-1
source /umbc/rs/cybertrn/reu2026/team1/research/testing/Oscar_testing/setup_env.sh

python -u savingauroraday1.py

deactivate



