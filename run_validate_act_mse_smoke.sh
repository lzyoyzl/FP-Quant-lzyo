#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Small smoke test for transform_search objective=act_mse
#
# Purpose:
# 1) verify the new act_mse path can run end-to-end
# 2) verify slot activation samples are collected and logged
# 3) compare it against the existing cov path under the same small setup
#
# Default behavior:
# - model: llama-3-8b-instruct
# - format: nvfp
# - runs 2 jobs: cov, act_mse
# - uses a small calibration slice to reduce runtime
###############################################################################

if [[ -f "env_var.sh" ]]; then
  # shellcheck disable=SC1091
  source "env_var.sh"
fi

MODEL_DIR="${MODEL_DIR:-/cephfs/shared/model/llama-3-8b-instruct}"
DTYPE="${DTYPE:-auto}"
SEQ_LEN="${SEQ_LEN:-2048}"
N_SEQS="${N_SEQS:-8}"
SEED="${SEED:-0}"

FORMAT="${FORMAT:-nvfp}"          # nvfp | mxfp
W_BITS="${W_BITS:-4}"
A_BITS="${A_BITS:-4}"
W_OBSERVER="${W_OBSERVER:-mse}"
EXPORT_MODE="${EXPORT_MODE:-pseudoquant}"
ACT_SAMPLE_SIZE="${ACT_SAMPLE_SIZE:-256}"
CANDIDATES_STR="${CANDIDATES_STR:-identity hadamard householder}"

OUT_ROOT="${OUT_ROOT:-outputs/act_mse_smoke}"
LOG_DIR="${LOG_DIR:-logs/act_mse_smoke}"
mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

USE_OFFLOAD="${USE_OFFLOAD:-1}"
OFFLOAD_ARGS=()
if [[ "${USE_OFFLOAD}" == "1" ]]; then
  OFFLOAD_ARGS+=(--cpu_offload_modules --cpu_offload_activations)
fi

case "${FORMAT}" in
  nvfp)
    GROUP_SIZE=16
    ;;
  mxfp)
    GROUP_SIZE=32
    ;;
  *)
    echo "[ERROR] Unsupported FORMAT=${FORMAT}. Use nvfp or mxfp."
    exit 1
    ;;
esac

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

read -r -a CANDIDATES <<< "${CANDIDATES_STR}"

COMMON_ARGS=(
  --model_name_or_path "${MODEL_DIR}"
  --dataset_name_or_path "${CALIB_DATASET}"
  --num_sequences "${N_SEQS}"
  --sequence_length "${SEQ_LEN}"
  --seed "${SEED}"
  --dtype "${DTYPE}"
  --format "${FORMAT}"
  --w_bits "${W_BITS}"
  --a_bits "${A_BITS}"
  --w_group_size "${GROUP_SIZE}"
  --a_group_size "${GROUP_SIZE}"
  --w_granularity group
  --a_granularity group
  --w_observer "${W_OBSERVER}"
  --gptq
  --quantization_order default
  --transform_search
  --transform_search_candidates "${CANDIDATES[@]}"
  --export_quantized_model "${EXPORT_MODE}"
  --fuse_global_scale
  --amp
  "${OFFLOAD_ARGS[@]}"
)

run_one() {
  local objective="$1"
  local name="${FORMAT}_gptq_${objective}_smoke"
  local outdir="${OUT_ROOT}/${name}"
  local logfile="${LOG_DIR}/${name}.log"

  echo "============================================================"
  echo "[RUN] ${name}"
  echo "  model     : ${MODEL_DIR}"
  echo "  calib     : ${CALIB_DATASET}"
  echo "  format    : ${FORMAT}"
  echo "  group     : ${GROUP_SIZE}"
  echo "  objective : ${objective}"
  echo "  candidates: ${CANDIDATES_STR}"
  echo "  n_seqs    : ${N_SEQS}"
  echo "  seq_len   : ${SEQ_LEN}"
  echo "  out       : ${outdir}"
  echo "  log       : ${logfile}"
  echo "============================================================"

  python model_quant.py \
    "${COMMON_ARGS[@]}" \
    --transform_search_objective "${objective}" \
    --transform_search_act_sample_size "${ACT_SAMPLE_SIZE}" \
    --save_path "${outdir}" \
    2>&1 | tee "${logfile}"
}

check_log_contains() {
  local logfile="$1"
  local pattern="$2"
  if ! grep -q "${pattern}" "${logfile}"; then
    echo "[ERROR] Expected pattern not found in ${logfile}: ${pattern}"
    exit 1
  fi
}

echo "[INFO] MODEL_DIR=${MODEL_DIR}"
echo "[INFO] CALIB_DATASET=${CALIB_DATASET}"
echo "[INFO] FORMAT=${FORMAT}"
echo "[INFO] GROUP_SIZE=${GROUP_SIZE}"
echo "[INFO] W_OBSERVER=${W_OBSERVER}"
echo "[INFO] ACT_SAMPLE_SIZE=${ACT_SAMPLE_SIZE}"
echo "[INFO] OUT_ROOT=${OUT_ROOT}"
echo "[INFO] LOG_DIR=${LOG_DIR}"
echo

run_one "cov"
run_one "act_mse"

COV_LOG="${LOG_DIR}/${FORMAT}_gptq_cov_smoke.log"
ACT_LOG="${LOG_DIR}/${FORMAT}_gptq_act_mse_smoke.log"

check_log_contains "${COV_LOG}" "objective=cov"
check_log_contains "${COV_LOG}" "group_covariances=enabled"

check_log_contains "${ACT_LOG}" "objective=act_mse"
check_log_contains "${ACT_LOG}" "act_samples=enabled"

echo
echo "[OK] act_mse smoke validation passed."
echo "[INFO] Compare logs:"
echo "  - ${COV_LOG}"
echo "  - ${ACT_LOG}"
echo "[INFO] Outputs:"
echo "  - ${OUT_ROOT}/${FORMAT}_gptq_cov_smoke"
echo "  - ${OUT_ROOT}/${FORMAT}_gptq_act_mse_smoke"
