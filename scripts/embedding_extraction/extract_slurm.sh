#!/bin/bash
#SBATCH --job-name=jepa-l
#SBATCH --partition=mit_preemptable
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-9
#SBATCH --output=/home/sebasmos/jepa-l_%A_%a.log

# ============================================================
# Extract frozen V-JEPA2 embeddings (10-folder array job)
# ============================================================
# Each task processes one pXX folder on a single L40S GPU.
# Supports resume: if preempted, resubmit and it picks up.
#
# Usage:
#   sbatch scripts/embedding_extraction/extract_slurm.sh              # vitl
#   sbatch scripts/embedding_extraction/extract_slurm.sh vith
#   sbatch --array=0 scripts/embedding_extraction/extract_slurm.sh    # p10 only
#
# Monitor:
#   tail -f ~/jepa-l_<jobid>_*.log
# ============================================================

module load miniforge/24.3.0-0
module load cuda/12.4.0
conda activate vjepa2-312

cd "$(dirname "$0")/../.."

MODEL="${1:-vitl}"

# Map array index to folder name (0→p10, 1→p11, ..., 9→p19)
FOLDERS=(p10 p11 p12 p13 p14 p15 p16 p17 p18 p19)
FOLDER="${FOLDERS[$SLURM_ARRAY_TASK_ID]}"

INPUT_DIR="/orcd/pool/006/lceli_shared/mimic-iv-echo-mp4"

echo "$(date) | Job ${SLURM_ARRAY_JOB_ID} task ${SLURM_ARRAY_TASK_ID}"
echo "Model:  ${MODEL}"
echo "Folder: ${FOLDER}"
echo "Input:  ${INPUT_DIR}/${FOLDER}"
echo "============================================================"

PYTHONUNBUFFERED=1 python scripts/embedding_extraction/extract_embeddings.py \
    --model "$MODEL" \
    --input_dir "$INPUT_DIR" \
    --folder "$FOLDER" \
    --num_workers 8 \
    --save_every 10000

EXIT_CODE=$?

echo "============================================================"
echo "$(date) | ${FOLDER} finished with exit code: ${EXIT_CODE}"
