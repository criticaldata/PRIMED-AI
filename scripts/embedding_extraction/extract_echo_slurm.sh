#!/bin/bash
#SBATCH --job-name=echo-jepa
#SBATCH --partition=mit_preemptable
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-9
#SBATCH --output=logs/echo-jepa_%A_%a.log

# ============================================================
# Extract frozen EchoJEPA embeddings (10-folder array job)
# ============================================================
# Usage:
#   sbatch scripts/embedding_extraction/extract_echo_slurm.sh echo-vitl-scratch
#   sbatch scripts/embedding_extraction/extract_echo_slurm.sh echo-vitl-mimic117
#   sbatch scripts/embedding_extraction/extract_echo_slurm.sh echo-vitb-mimic169
#
# Outputs go to:
#   /path/to/shared/jepa-embeddings-mimiciv-echo/<model>_embeddings_p{10..19}.pt
#
# Resume: safe to resubmit — script skips already-extracted videos.
# Merge after all 10 tasks complete:
#   python scripts/embedding_extraction/merge_embeddings.py --model echo-vitl-mimic117
#
# Monitor:
#   tail -f logs/echo-jepa_<jobid>_*.log
# ============================================================

module load miniforge/24.3.0-0
module load cuda/12.4.0
conda activate vjepa2-312

cd "$(dirname "$0")/../.."

MODEL="${1:-echo-vitl-mimic117}"

FOLDERS=(p10 p11 p12 p13 p14 p15 p16 p17 p18 p19)
FOLDER="${FOLDERS[$SLURM_ARRAY_TASK_ID]}"

INPUT_DIR="/path/to/shared/mimic-iv-echo-mp4"

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

exit $EXIT_CODE
