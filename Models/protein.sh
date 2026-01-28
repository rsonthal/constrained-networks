#!/bin/bash
#SBATCH -J fm-sweep-protein
#SBATCH -p gtml
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --array=0-17%11
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err

set -euo pipefail
export PYTHONUNBUFFERED=1

module load miniconda
conda activate jupyter_stack

mkdir -p logs outputs

# dataset (basename without ".pt")
DATASET="protein_dataset"

# 6 configs: (3 lrs) x (2 wds)
lrs=(3e-4 1e-3 3e-3)
wds=(0 1e-4)

# alpha sweep (used when --T auto)
alphas=(0.25 0.5 1.0)

# fixed architecture + knobs
HIDDEN_DIM=256
NUM_LAYERS=8
BATCH_SIZE=256
SCHED_FACTOR=0.5
SEED=0

# epochs + early stop + scheduler patience (protein uses defaults)
NUM_EPOCHS=2000
EARLY_STOP=1000
SCHED_PATIENCE=100

idx=${SLURM_ARRAY_TASK_ID}

# total jobs = 6 configs * 3 alphas = 18
config_i=$(( idx % 6 ))              # 0..5
alpha_i=$(( (idx / 6) % 3 ))         # 0..2

lr_i=$(( config_i % 3 ))             # 0..2
wd_i=$(( config_i / 3 ))             # 0..1

LR=${lrs[$lr_i]}
WD=${wds[$wd_i]}
ALPHA=${alphas[$alpha_i]}

OUTDIR="outputs/${DATASET}/alpha${ALPHA}/lr${LR}_wd${WD}"
mkdir -p "${OUTDIR}"

echo "[$(date)] job_id=${SLURM_JOB_ID} array_id=${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Dataset: ${DATASET}"
echo "Config: T=auto alpha=${ALPHA} lr=${LR} wd=${WD} hidden=${HIDDEN_DIM} layers=${NUM_LAYERS} batch=${BATCH_SIZE} epochs=${NUM_EPOCHS} early_stop=${EARLY_STOP} sched_patience=${SCHED_PATIENCE} seed=${SEED}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# --------------------------------------------------------------------
# Node-local dataset caching + jitter + serialized staging + retries
# --------------------------------------------------------------------
SRC="../Data/${DATASET}.pt"
if [[ ! -f "${SRC}" ]]; then
  echo "ERROR: dataset not found at ${SRC}" >&2
  exit 2
fi

# small jitter to avoid stampede on the shared FS
sleep $(( (RANDOM % 10) + 1 ))

CACHE_ROOT="${SLURM_TMPDIR:-/tmp}/${USER}/fm_cache"
mkdir -p "${CACHE_ROOT}"
LOCKFILE="${CACHE_ROOT}/.stage_${DATASET}.lock"
LOCAL_DATA="${CACHE_ROOT}/${DATASET}.pt"   # shared across array tasks on the same node

stage_dataset() {
  (
    flock -x 200

    if [[ -f "${LOCAL_DATA}" ]]; then
      echo "Using cached dataset: ${LOCAL_DATA}"
      return 0
    fi

    echo "Staging dataset to node-local cache: ${LOCAL_DATA}"

    for attempt in 1 2 3 4 5; do
      if rsync -a --inplace --partial "${SRC}" "${LOCAL_DATA}.tmp"; then
        mv -f "${LOCAL_DATA}.tmp" "${LOCAL_DATA}"
        echo "Staged OK: ${LOCAL_DATA}"
        ls -lh "${LOCAL_DATA}" || true
        return 0
      fi

      echo "stage failed (attempt ${attempt}/5). Retrying after backoff..." >&2
      rm -f "${LOCAL_DATA}.tmp" || true
      sleep $((attempt * 10))
    done

    echo "ERROR: failed to stage ${SRC} after 5 attempts" >&2
    return 1
  ) 200>"${LOCKFILE}"
}

stage_dataset

# ----------------------------
# Train
# ----------------------------
srun --cpu-bind=cores \
  python -u train_flowmatch.py \
    --dataset "${LOCAL_DATA}" \
    --outdir "${OUTDIR}" \
    --hidden-dim "${HIDDEN_DIM}" \
    --num-layers "${NUM_LAYERS}" \
    --lr "${LR}" \
    --weight-decay "${WD}" \
    --T auto \
    --alpha "${ALPHA}" \
    --batch-size "${BATCH_SIZE}" \
    --num-epochs "${NUM_EPOCHS}" \
    --scheduler-patience "${SCHED_PATIENCE}" \
    --scheduler-factor "${SCHED_FACTOR}" \
    --early-stop "${EARLY_STOP}" \
    --seed "${SEED}" \
    --device cuda

echo "[$(date)] done."

