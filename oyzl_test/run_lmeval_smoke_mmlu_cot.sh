#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# 0) 自动后台脱离（断网继续跑）
###############################################################################
AUTO_DETACH="${AUTO_DETACH:-1}"
LOG_DIR="${LOG_DIR:-lm_eval_logs_mx_mse_rotsearch_smoke_mmlu_cot}"
mkdir -p "${LOG_DIR}"

if [[ "${AUTO_DETACH}" == "1" && -t 1 && -z "${__DETACHED:-}" ]]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  MASTER_LOG="${LOG_DIR}/master_${TS}.log"
  PID_FILE="${LOG_DIR}/master_${TS}.pid"
  echo "[INFO] Detaching to background..."
  echo "[INFO] Master log: ${MASTER_LOG}"
  __DETACHED=1 nohup bash "$0" "$@" > "${MASTER_LOG}" 2>&1 &
  echo $! > "${PID_FILE}"
  disown || true
  echo "[INFO] PID saved to: ${PID_FILE}"
  echo "[INFO] Follow logs with: tail -f ${MASTER_LOG}"
  exit 0
fi

###############################################################################
# 1) 路径与公共配置（按需改）
###############################################################################
MODEL_DIR="${MODEL_DIR:-/cephfs/shared/zlouyang/FP-Quant/oyzl_test/outputs/mxfp_gptq_mse_rotsearch_smoke}"

OUT_DIR="${OUT_DIR:-lm_eval_results_mse_rotsearch_smoke_mmlu_cot}"
mkdir -p "${OUT_DIR}"

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-True}"

CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

MODEL_NAME="$(basename "${MODEL_DIR}")"

###############################################################################
# 2) 工具函数
###############################################################################
is_done () {
  local outpath="$1"
  if [[ -f "${outpath}" ]]; then
    [[ -s "${outpath}" ]] && return 0 || return 1
  fi
  if [[ -d "${outpath}" ]]; then
    find "${outpath}" -maxdepth 2 -type f -name "*.json" -size +0c >/dev/null 2>&1 && return 0
  fi
  return 1
}

run_cmd () {
  local tag="$1"
  local task="$2"
  local cmdline="$3"
  local outpath="$4"
  local logfile="$5"

  if is_done "${outpath}"; then
    echo "[SKIP] ${tag} :: ${task} already has output: ${outpath}"
    return 0
  fi

  echo "=== RUN ${tag} :: ${task} ==="
  echo "  out : ${outpath}"
  echo "  log : ${logfile}"
  echo "  cmd : ${cmdline}"
  echo

  mkdir -p "$(dirname "${outpath}")"

  set +e
  stdbuf -oL -eL bash -lc "${cmdline}" > "${logfile}" 2>&1
  rc=$?
  set -e

  if [[ $rc -ne 0 ]]; then
    echo "[FAIL] ${tag} :: ${task} (rc=${rc}). See: ${logfile}"
    return $rc
  fi

  echo "[OK] ${tag} :: ${task}"
  return 0
}

make_model_args () {
  local pretrained="$1"
  echo "pretrained=${pretrained},device=${DEVICE},dtype=${DTYPE},trust_remote_code=${TRUST_REMOTE_CODE}"
}

###############################################################################
# 3) 只跑 MMLU-CoT
###############################################################################
echo "[INFO] MODEL_DIR=${MODEL_DIR}"
echo "[INFO] OUT_DIR=${OUT_DIR}"
echo "[INFO] LOG_DIR=${LOG_DIR}"
echo

[[ -d "${MODEL_DIR}" ]] || { echo "[ERROR] MODEL_DIR not found: ${MODEL_DIR}"; exit 1; }

task="mmlu_cot_llama"
model_args="$(make_model_args "${MODEL_DIR}")"
out="${OUT_DIR}/${MODEL_NAME}/MMLU_COT_full"
log="${LOG_DIR}/${MODEL_NAME}__mmlu_cot_llama.log"
cmd="lm_eval --model hf --model_args \"${model_args}\" --tasks mmlu_cot_llama --batch_size 64 --apply_chat_template --fewshot_as_multiturn --output_path \"${out}\""

if ! run_cmd "${MODEL_NAME}" "${task}" "${cmd}" "${out}" "${log}"; then
  rc=$?
  if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
    echo "[WARN] ${MODEL_NAME} failed on ${task} (rc=${rc})"
    exit 0
  else
    exit $rc
  fi
fi

echo "[DONE] MMLU-CoT completed for ${MODEL_NAME}."
echo "Results: ${OUT_DIR}/${MODEL_NAME}/"
echo "Logs   : ${LOG_DIR}/"