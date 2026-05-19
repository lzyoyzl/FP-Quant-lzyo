#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# 0) 自动后台脱离（断网继续跑）
###############################################################################
AUTO_DETACH="${AUTO_DETACH:-1}"
LOG_DIR="${LOG_DIR:-lm_eval_logs_llama3_legacy_only}"
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
# export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
# export http_proxy="${http_proxy:-http://127.0.0.1:7990}"
# export https_proxy="${https_proxy:-http://127.0.0.1:7990}"

BASE_MODEL_DIR="${BASE_MODEL_DIR:-/cephfs/shared/model/llama-3-8b-instruct}"
LEGACY_ROOT="${LEGACY_ROOT:-/cephfs/shared/zlouyang/FP-Quant/oyzl_test/results_quant/llama-3-8b-instruct}"

OUT_DIR="${OUT_DIR:-lm_eval_results_llama3_legacy_only}"
mkdir -p "${OUT_DIR}"

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-True}"

CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
RUN_BASE="${RUN_BASE:-0}"  # 与 rotsearch 脚本并行时，默认不重复跑 BASE

GSM_APPLY_CHAT_TEMPLATE="${GSM_APPLY_CHAT_TEMPLATE:-0}"

declare -a TASKS=("boolq" "arc_easy" "arc_challenge" "piqa" "winogrande" "hellaswag")

BATCH_SIZE="${BATCH_SIZE:-16}"
echo "[INFO] BATCH_SIZE=${BATCH_SIZE}"

###############################################################################
# 1.1) fewshot
###############################################################################
FS_BOOLQ=0
FS_ARC_EASY=0
FS_ARC_CHALLENGE=25
FS_PIQA=0
FS_WINOGRANDE=5
FS_HELLASWAG=0

###############################################################################
# 2) legacy 模型集合（20个）
###############################################################################
MODELS_LEGACY=(
  "mxfp_gptq_w4a4_hadamard_h128_minmax_activation"
  "mxfp_gptq_w4a4_hadamard_h128_minmax_default"
  "mxfp_gptq_w4a4_hadamard_h128_mse_default"
  "mxfp_gptq_w4a4_identity"
  "mxfp_gptq_w4a4_identity_minmax_activation"
  "mxfp_gptq_w4a4_identity_mse_default"
  "mxfp_mrgptq_w4a4_hadamard_h128_mse_activation"
  "mxfp_rtn_w4a4_hadamard_h128"
  "mxfp_rtn_w4a4_hadamard_h32"
  "mxfp_rtn_w4a4_identity"
  "nvfp_gptq_w4a4_hadamard_h128_minmax_activation"
  "nvfp_gptq_w4a4_hadamard_h128_minmax_default"
  "nvfp_gptq_w4a4_hadamard_h128_mse_default"
  "nvfp_gptq_w4a4_identity_minmax_activation"
  "nvfp_gptq_w4a4_identity_mse_default"
  "nvfp_mrgptq_w4a4_hadamard_h128_mse_activation"
  "nvfp_rtn_w4a4_hadamard_h128"
  "nvfp_rtn_w4a4_hadamard_h16"
  "nvfp_rtn_w4a4_identity"
  "nvfp_gptq_w4a4_identity"
)

###############################################################################
# 3) 工具函数
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

build_cmd () {
  local model_args="$1"
  local task="$2"
  local out="$3"
  local fs=0

  case "${task}" in
    boolq) fs="${FS_BOOLQ}" ;;
    arc_easy) fs="${FS_ARC_EASY}" ;;
    arc_challenge) fs="${FS_ARC_CHALLENGE}" ;;
    piqa) fs="${FS_PIQA}" ;;
    winogrande) fs="${FS_WINOGRANDE}" ;;
    hellaswag) fs="${FS_HELLASWAG}" ;;
  esac

  echo "lm_eval --model hf --model_args \"${model_args}\" --tasks ${task} --num_fewshot ${fs} --batch_size ${BATCH_SIZE} --output_path \"${out}\""
}

###############################################################################
# 4) 调度
###############################################################################
echo "[INFO] BASE_MODEL_DIR=${BASE_MODEL_DIR}"
echo "[INFO] LEGACY_ROOT=${LEGACY_ROOT}"
echo "[INFO] OUT_DIR=${OUT_DIR}"
echo "[INFO] LOG_DIR=${LOG_DIR}"
echo "[INFO] RUN_BASE=${RUN_BASE}"
echo "[INFO] GSM_APPLY_CHAT_TEMPLATE=${GSM_APPLY_CHAT_TEMPLATE}"
echo

[[ -d "${BASE_MODEL_DIR}" ]] || { echo "[ERROR] BASE_MODEL_DIR not found: ${BASE_MODEL_DIR}"; exit 1; }
[[ -d "${LEGACY_ROOT}" ]] || { echo "[ERROR] LEGACY_ROOT not found: ${LEGACY_ROOT}"; exit 1; }

for task in "${TASKS[@]}"; do
  echo "############################"
  echo "### TASK: ${task}"
  echo "############################"

  if [[ "${RUN_BASE}" == "1" ]]; then
    base_args="$(make_model_args "${BASE_MODEL_DIR}")"
    out="${OUT_DIR}/BASE/${task}_full"
    log="${LOG_DIR}/BASE__${task}.log"
    cmd="$(build_cmd "${base_args}" "${task}" "${out}")"

    if ! run_cmd "BASE" "${task}" "${cmd}" "${out}" "${log}"; then
      rc=$?
      if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
        echo "[WARN] BASE failed on ${task} (rc=${rc}), continue..."
      else
        exit $rc
      fi
    fi
  fi

  for name in "${MODELS_LEGACY[@]}"; do
    model_path="${LEGACY_ROOT}/${name}"
    if [[ ! -d "${model_path}" ]]; then
      echo "[WARN] Missing LEGACY model dir, skip: ${model_path}"
      continue
    fi

    model_args="$(make_model_args "${model_path}")"
    out="${OUT_DIR}/LEGACY/${name}/${task}_full"
    log="${LOG_DIR}/LEGACY__${name}__${task}.log"
    cmd="$(build_cmd "${model_args}" "${task}" "${out}")"

    if ! run_cmd "LEGACY__${name}" "${task}" "${cmd}" "${out}" "${log}"; then
      rc=$?
      if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
        echo "[WARN] LEGACY ${name} failed on ${task} (rc=${rc}), continue..."
        continue
      else
        exit $rc
      fi
    fi
  done

  echo
done

echo "[DONE] legacy script completed (6 tasks)."
echo "Results: ${OUT_DIR}/"
echo "Logs   : ${LOG_DIR}/"
