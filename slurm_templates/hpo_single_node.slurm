#!/bin/bash
# SLURM template: HPO on a single node with multiple GPUs
# Usage: sbatch slurm_templates/hpo_single_node.slurm
#
# Edit variables below before submitting:
#   STUDY_NAME   : Unique name for the Optuna study
#   STORAGE_PATH : Path to SQLite database (shared across array tasks)
#   CONFIG       : Path to HPO YAML config
#   N_TRIALS     : Total number of Optuna trials
#   N_JOBS       : Number of parallel workers (= n_gpus)
#   PARTITION    : SLURM partition name

#SBATCH --job-name=sulcal_seg_hpo
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/hpo_single_%j.out
#SBATCH --error=slurm_logs/hpo_single_%j.err

set -euo pipefail

# --- Configuration ---
STUDY_NAME="sulcal_seg_full_hpo"
STORAGE_PATH="outputs/hpo/${STUDY_NAME}.db"
CONFIG="configs/hpo_full_model.yaml"
N_TRIALS=100
N_JOBS=4

# --- Environment ---
module load cuda/11.8
source activate sulcal_seg

mkdir -p outputs/hpo slurm_logs

echo "[$(date)] Starting HPO: ${STUDY_NAME} — ${N_TRIALS} trials, ${N_JOBS} parallel workers"

# --- Run ---
python -m sulcal_seg.experiments.run_ablations \
    --config "${CONFIG}" \
    --study-name "${STUDY_NAME}" \
    --storage "${STORAGE_PATH}" \
    --n-trials "${N_TRIALS}" \
    --n-jobs "${N_JOBS}" \
    --output-dir "outputs/hpo"

echo "[$(date)] HPO completed."
