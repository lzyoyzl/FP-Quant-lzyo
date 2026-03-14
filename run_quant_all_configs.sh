#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# 0) Optional env vars (HF cache / OUT_CALIB / proxies / etc.)
###############################################################################
if [[ -f "env_var.sh" ]]; then
  # shellcheck disable=SC1091
  source "env_var.sh"
fi

###############################################################################
# 1) Core settings (override by env)
###############################################################################
MODEL_DIR="${MODEL_DIR:-/cephfs/shared/model/llama-3-8b-instruct}"
DTYPE="${DTYPE:-auto}"
SEQ_LEN="${SEQ_LEN:-2048}"
N_SEQS="${N_SEQS:-128}"
SEED="${SEED:-0}"

# pseudoquant is recommended for broad compatibility.
EXPORT_MODE="${EXPORT_MODE:-pseudoquant}"

OUT_ROOT="${OUT_ROOT:-outputs/quant_all_configs}"
LOG_DIR="${LOG_DIR:-logs/quant_all_configs}"
mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

# 0: no offload (faster if memory is enough), 1: offload modules+activations.
USE_OFFLOAD="${USE_OFFLOAD:-0}"
OFFLOAD_ARGS=()
if [[ "${USE_OFFLOAD}" == "1" ]]; then
  OFFLOAD_ARGS+=(--cpu_offload_modules --cpu_offload_activations)
fi

# Continue running remaining experiments even if one fails.
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

###############################################################################
# 2) Calibration dataset selection: pt > jsonl > fineweb-edu fallback
###############################################################################
CALIB_PT="${CALIB_PT:-${OUT_CALIB:-}/fineweb_calib_1024x2048_tokens.pt}"
CALIB_JSONL="${CALIB_JSONL:-${OUT_CALIB:-}/fineweb_calib_1024x2048_text.jsonl}"

if [[ -n "${CALIB_PT}" && -f "${CALIB_PT}" ]]; then
  CALIB_DATASET="${CALIB_PT}"
elif [[ -n "${CALIB_JSONL}" && -f "${CALIB_JSONL}" ]]; then
  CALIB_DATASET="${CALIB_JSONL}"
else
  CALIB_DATASET="fineweb-edu"
  echo "[WARN] Local calib file not found, fallback to dataset: ${CALIB_DATASET}"
fi

###############################################################################
# 3) Pre-checks
###############################################################################
python -c "import lm_eval" >/dev/null 2>&1 || {
  echo "[ERROR] Cannot import lm_eval in current environment."
  echo "        Install with: python -m pip install lm-eval"
  exit 1
}

###############################################################################
# 4) Common args
###############################################################################
COMMON_ARGS=(
  --model_name_or_path "${MODEL_DIR}"
  --dataset_name_or_path "${CALIB_DATASET}"
  --num_sequences "${N_SEQS}"
  --sequence_length "${SEQ_LEN}"
  --seed "${SEED}"
  --dtype "${DTYPE}"
  --w_bits 4
  --a_bits 4
  --w_granularity group
  --a_granularity group
  --export_quantized_model "${EXPORT_MODE}"
  --fuse_global_scale
  --amp
)

DONE_LIST="${LOG_DIR}/done.txt"
FAILED_LIST="${LOG_DIR}/failed.txt"
: > "${DONE_LIST}" || true
: > "${FAILED_LIST}" || true

run_one() {
  local name="$1"
  shift

  local outdir="${OUT_ROOT}/${name}"
  local logfile="${LOG_DIR}/${name}.log"

  echo "=== RUN: ${name} ==="
  echo "  model: ${MODEL_DIR}"
  echo "  calib: ${CALIB_DATASET}"
  echo "  out  : ${outdir}"
  echo "  log  : ${logfile}"

  mkdir -p "${outdir}"

  set +e
  stdbuf -oL -eL python /cephfs/shared/zlouyang/FP-Quant/model_quant.py \
    "${COMMON_ARGS[@]}" \
    --save_path "${outdir}" \
    "${OFFLOAD_ARGS[@]}" \
    "$@" > "${logfile}" 2>&1
  rc=$?
  set -e

  if [[ $rc -ne 0 ]]; then
    echo "[FAIL] ${name} (rc=${rc})"
    echo "${name} rc=${rc} log=${logfile}" >> "${FAILED_LIST}"
    return 1
  fi

  echo "[OK] ${name}"
  echo "${name}" >> "${DONE_LIST}"
  return 0
}

run_or_continue() {
  if ! run_one "$@"; then
    if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
      echo "[WARN] Continue on error enabled; moving to next experiment."
    else
      echo "[ERROR] Stopping due to failure."
      exit 1
    fi
  fi
}

###############################################################################
# 5) Experiment suite
# NOTE: quantization_order only supports: default / activation
###############################################################################

# ---------------- NVFP4 (group=16) ----------------
run_or_continue "nvfp_rtn_identity" \
  --format nvfp --w_group_size 16 --a_group_size 16 \
  --transform_class identity --w_observer minmax --quantization_order default

run_or_continue "nvfp_rtn_hadamard_h16" \
  --format nvfp --w_group_size 16 --a_group_size 16 \
  --transform_class hadamard --hadamard_group_size 16 \
  --w_observer minmax --quantization_order default

run_or_continue "nvfp_gptq_identity" \
  --format nvfp --w_group_size 16 --a_group_size 16 \
  --transform_class identity --w_observer minmax --quantization_order default \
  --gptq

run_or_continue "nvfp_mrgptq_h128_mse_activation" \
  --format nvfp --w_group_size 16 --a_group_size 16 \
  --transform_class hadamard --hadamard_group_size 128 \
  --w_observer mse --quantization_order activation \
  --gptq

# New: selective rotation search (includes householder)
run_or_continue "nvfp_rtn_rotsearch" \
  --format nvfp --w_group_size 16 --a_group_size 16 \
  --transform_class identity --w_observer minmax --quantization_order default \
  --transform_search \
  --transform_search_candidates identity hadamard dct dst gsr householder

run_or_continue "nvfp_gptq_rotsearch" \
  --format nvfp --w_group_size 16 --a_group_size 16 \
  --transform_class identity --w_observer minmax --quantization_order default \
  --gptq \
  --transform_search \
  --transform_search_candidates identity hadamard dct dst gsr householder

# ---------------- MXFP4 (group=32) ----------------
run_or_continue "mxfp_rtn_identity" \
  --format mxfp --w_group_size 32 --a_group_size 32 \
  --transform_class identity --w_observer minmax --quantization_order default

run_or_continue "mxfp_rtn_hadamard_h32" \
  --format mxfp --w_group_size 32 --a_group_size 32 \
  --transform_class hadamard --hadamard_group_size 32 \
  --w_observer minmax --quantization_order default

run_or_continue "mxfp_gptq_identity" \
  --format mxfp --w_group_size 32 --a_group_size 32 \
  --transform_class identity --w_observer minmax --quantization_order default \
  --gptq

run_or_continue "mxfp_mrgptq_h128_mse_activation" \
  --format mxfp --w_group_size 32 --a_group_size 32 \
  --transform_class hadamard --hadamard_group_size 128 \
  --w_observer mse --quantization_order activation \
  --gptq

# New: selective rotation search (includes householder)
run_or_continue "mxfp_rtn_rotsearch" \
  --format mxfp --w_group_size 32 --a_group_size 32 \
  --transform_class identity --w_observer minmax --quantization_order default \
  --transform_search \
  --transform_search_candidates identity hadamard dct dst gsr householder

run_or_continue "mxfp_gptq_rotsearch" \
  --format mxfp --w_group_size 32 --a_group_size 32 \
  --transform_class identity --w_observer minmax --quantization_order default \
  --gptq \
  --transform_search \
  --transform_search_candidates identity hadamard dct dst gsr householder

echo
echo "[DONE] All experiments processed."
echo "Outputs: ${OUT_ROOT}/"
echo "Logs   : ${LOG_DIR}/"
echo "Done list   : ${DONE_LIST}"
echo "Failed list : ${FAILED_LIST}"
