#!/bin/bash
#SBATCH -J ff-sweep
#SBATCH -p gtml
#SBATCH --account=gtml
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --array=0-29%11
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

set -euo pipefail
export PYTHONUNBUFFERED=1

module load miniconda
conda activate jupyter_stack

mkdir -p logs outputs

# ----------------------------
# Sweep definition
# ----------------------------
# Datasets (NO cs)
datasets=(sphere so3 protein disk cs)

# Model types:
# - regular/projected on all datasets
# - exponential on sphere/so3/protein (NOT disk)
# (We enumerate valid pairs explicitly.)
pairs=(
  "sphere probabilistic"
  "so3 probabilistic"
  "protein probabilistic"
  "disk probabilistic"
  "cs probabilistic"    
)
NUM_PAIRS=${#pairs[@]}   # 11

# Depth sweep
depths=(4 6 8)

# 6 configs: (3 lrs) x (2 wds)
wds=(0 1e-4)

LR=1e-3
SEED=0

# Training constants (per your spec)
BATCH_SIZE=500
NUM_EPOCHS=10000

# Scheduler settings (scale-appropriate for 100k epochs)
EVAL_EVERY=100
SCHED_PATIENCE_EPOCHS=1000
SCHED_FACTOR=0.5

idx=${SLURM_ARRAY_TASK_ID}

P=${NUM_PAIRS}
D=${#depths[@]}
W=${#wds[@]}

TOTAL_JOBS=$((P * D * W))

if [[ "${idx}" -ge "${TOTAL_JOBS}" ]]; then
  echo "idx=${idx} out of range (TOTAL_JOBS=${TOTAL_JOBS})"
  exit 2
fi

# Indexing: fastest-changing = wd, then depth, then pair
wd_i=$(( idx % W ))                 # 0..W-1
depth_i=$(( (idx / W) % D ))        # 0..D-1
pair_i=$(( idx / (W * D) ))         # 0..P-1

WD=${wds[$wd_i]}
DEPTH=${depths[$depth_i]}

read -r DATASET MODEL_TYPE <<< "${pairs[$pair_i]}"

echo "[$(date)] job_id=${SLURM_JOB_ID} array_id=${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Dataset: ${DATASET}"
echo "Model: ${MODEL_TYPE}"
echo "Depth: ${DEPTH}"
echo "Config: lr=${LR} wd=${WD} batch=${BATCH_SIZE} epochs=${NUM_EPOCHS} eval_every=${EVAL_EVERY} sched_patience_epochs=${SCHED_PATIENCE_EPOCHS} sched_factor=${SCHED_FACTOR} seed=${SEED}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ----------------------------
# Train
# ----------------------------
srun --cpu-bind=cores \
  python -u train_ff.py \
    --model_type "${MODEL_TYPE}" \
    --dataset "${DATASET}" \
    --depth "${DEPTH}" \
    --lr "${LR}" \
    --weight_decay "${WD}" \
    --batch_size "${BATCH_SIZE}" \
    --num_epochs "${NUM_EPOCHS}" \
    --eval_every "${EVAL_EVERY}" \
    --scheduler_patience_epochs "${SCHED_PATIENCE_EPOCHS}" \
    --scheduler_factor "${SCHED_FACTOR}" \
    --seed "${SEED}" \
    --device cuda \
    --residual

echo "[$(date)] done."
