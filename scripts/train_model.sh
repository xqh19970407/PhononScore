#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
STAGE="${1:?Usage: train_model.sh pretrain|dft_posttrain DATA_ROOT OUTPUT_DIR}"
DATA_ROOT="${2:?Provide the unpacked phononscore_data_20261104 directory}"
OUT_DIR="${3:?Provide an output directory}"
DEVICE="${DEVICE:-cuda}"
case "${STAGE}" in
  pretrain|dft_posttrain) ;;
  *) echo "Unknown stage: ${STAGE}" >&2; exit 2 ;;
esac

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/alignn-main:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/phononscore-mplconfig}"
LMDB_PREFIX="${OUT_DIR}/lmdb/graphs_"
IDS_JSON="${OUT_DIR}/lmdb/ids_train_val_test.json"
CONFIG="${REPO_ROOT}/configs/training_${STAGE}.json"
mkdir -p "${OUT_DIR}/lmdb"

if [[ ! -d "${LMDB_PREFIX}train_data" ]]; then
  "${PYTHON_BIN}" -m phononscore.training.build_lmdb \
    --alignn_src "${REPO_ROOT}/third_party/alignn-main" \
    --dataset_dir "${DATA_ROOT}/${STAGE}" --config "${CONFIG}" \
    --lmdb_prefix "${LMDB_PREFIX}" --ids_json "${IDS_JSON}" \
    --split_ids "${DATA_ROOT}/splits/${STAGE}_internal_ids.json"
fi

RESTART_ARGS=()
if [[ "${STAGE}" == "dft_posttrain" ]]; then
  RESTART_ARGS=(--restart_encoder "${PRETRAIN_CKPT:-${REPO_ROOT}/checkpoints/phononscore_pretrain/best_model.pt}")
fi
"${PYTHON_BIN}" -m phononscore.training.train \
  --alignn_src "${REPO_ROOT}/third_party/alignn-main" --config "${CONFIG}" \
  --lmdb_prefix "${LMDB_PREFIX}" --ids_json "${IDS_JSON}" --out_dir "${OUT_DIR}" \
  --epochs "${EPOCHS:-100}" --batch_size "${BATCH_SIZE:-96}" --num_workers 0 \
  --learning_rate 0.0003 --weight_decay 0.00001 \
  --thresholds="-0.001,-0.01,-0.1,-1.0" --reg_target_mode clip5 \
  --beta 0.6 --alpha 0.1 \
  --lambda_cls 0.3 --lambda_rank 0.3 --lambda_ordinal 0.02 \
  --lambda_mdn 0.05 --lambda_geom_rank 0.05 \
  --cls_loss balanced_bce --rank_loss threshold_aware --max_rank_pairs 4096 \
  --mdn_hidden_dim 128 --mdn_gaussians 10 --mdn_distance_max 8.0 \
  --stable_threshold -0.1 --stable_tau 0.2 --seed 20260605 --save_every 100 \
  --device "${DEVICE}" "${RESTART_ARGS[@]}"
