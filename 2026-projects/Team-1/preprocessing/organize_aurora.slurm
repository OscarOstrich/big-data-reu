#!/bin/bash
#SBATCH --job-name=organize_aurora
#SBATCH --output=organize_aurora_%j.out
#SBATCH --error=organize_aurora_%j.err
#SBATCH --account=cybertrn
#SBATCH --cluster=chip-cpu
#SBATCH --partition=2024
#SBATCH --qos=shared
#SBATCH --time=1:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --tasks-per-node=1
#SBATCH --mem=24G
 
echo "Job started: $(date)"
echo "Running on node: $(hostname)"
 
# Activate the shared conda environment
source activate /umbc/rs/cybertrn/reu2026/team1/research/software/aurora_conda
 
# Run the script
python organize_aurora.py
 
echo "Job finished: $(date)"