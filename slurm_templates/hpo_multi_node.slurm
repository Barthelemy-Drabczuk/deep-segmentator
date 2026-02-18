#!/bin/bash
# SLURM template: Distributed HPO across multiple nodes (array job)
# Each array task runs independently against the same SQLite database.
# Usage: sbatch slurm_templates/hpo_multi_node.slurm
#
# Edit variables below before submitting.

#SBATCH --job-name=sulcal_seg_hpo_array
#SBATCH --partition=gpu
#SBATCH --array=0-3          # 4 nodes = array tasks 0,1,2,3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/hpo_node%a_%j.out
#SBATCH --error=slurm_logs/hpo_node%a_%j.err

set -euo pipefail

# --- Configuration ---
STUDY_NAME="sulcal_seg_full_hpo"
STORAGE_PATH="outputs/hpo/${STUDY_NAME}.db"  # must be on shared filesystem
CONFIG="configs/hpo_full_model.yaml"
TRIALS_PER_NODE=25    # total = TRIALS_PER_NODE × n_array_tasks
N_JOBS=4

# --- Environment ---
module load cuda/11.8
source activate sulcal_seg

mkdir -p outputs/hpo slurm_logs

echo "[$(date)] Array task ${SLURM_ARRAY_TASK_ID} starting HPO worker"

# --- Run ---
python -m sulcal_seg.experiments.run_ablations \
    --config "${CONFIG}" \
    --study-name "${STUDY_NAME}" \
    --storage "${STORAGE_PATH}" \
    --n-trials "${TRIALS_PER_NODE}" \
    --n-jobs "${N_JOBS}" \
    --worker-id "${SLURM_ARRAY_TASK_ID}" \
    --output-dir "outputs/hpo"

echo "[$(date)] Array task ${SLURM_ARRAY_TASK_ID} completed."
