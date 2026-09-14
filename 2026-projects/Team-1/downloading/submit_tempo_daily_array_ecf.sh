#!/bin/bash

#SBATCH --job-name=tempo_daily_ecf
#SBATCH --cluster=chip-cpu
#SBATCH --account=cybertrn
#SBATCH --partition=2024
#SBATCH --qos=shared
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --array=0-776%5
#SBATCH --output=/umbc/rs/cybertrn/reu2026/team1/research/data/logs/tempo_download/tempo_ecf_%A_%a.out
#SBATCH --error=/umbc/rs/cybertrn/reu2026/team1/research/data/logs/tempo_download/tempo_ecf_%A_%a.err

set -e

module load python/3.12.4
module load Anaconda3/2024.02-1

eval "$(conda shell.bash hook)"

conda activate /umbc/rs/cybertrn/users/rchen6/conda_envs/harmony_env

BASE_DIR="/umbc/rs/cybertrn/reu2026/team1/research/data"
SCRIPT_DIR="${BASE_DIR}/scripts/tempo_download"

START_DATE="2023-08-05"

cd "${SCRIPT_DIR}"

TARGET_DATE=$(python - <<EOF
import datetime
start = datetime.date.fromisoformat("${START_DATE}")
target = start + datetime.timedelta(days=${SLURM_ARRAY_TASK_ID})
print(target.isoformat())
EOF
)

echo "============================================================"
echo "SLURM job ID: ${SLURM_JOB_ID}"
echo "SLURM array task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Target date: ${TARGET_DATE}"
echo "Running on node: $(hostname)"
echo "Start time: $(date)"
echo "============================================================"

python download_tempo_by_day_ecf.py "${TARGET_DATE}"

echo "============================================================"
echo "Finished date: ${TARGET_DATE}"
echo "End time: $(date)"
echo "============================================================"
