#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# 0) 自动后台脱离（断网继续跑）
###############################################################################
AUTO_DETACH="${AUTO_DETACH:-1}"
LOG_DIR="${LOG_DIR:-lm_eval_logs_3tasks}"
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
QUANT_ROOT="${QUANT_ROOT:-/cephfs/shared/zlouyang/FP-Quant/oyzl_test/results_quant/llama-3-8b-instruct}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-/cephfs/shared/model/llama-3-8b-instruct}"

OUT_DIR="${OUT_DIR:-lm_eval_results_3tasks}"
mkdir -p "${OUT_DIR}"

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-True}"

CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

###############################################################################
# 2) 模型列表（20个）
###############################################################################
MODELS=(
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

###############################################################################
# 4) 只跑前三个任务：Winogrande -> Hellaswag -> GSM8K
###############################################################################
echo "[INFO] QUANT_ROOT=${QUANT_ROOT}"
echo "[INFO] BASE_MODEL_DIR=${BASE_MODEL_DIR}"
echo "[INFO] OUT_DIR=${OUT_DIR}"
echo "[INFO] LOG_DIR=${LOG_DIR}"
echo

[[ -d "${BASE_MODEL_DIR}" ]] || { echo "[ERROR] BASE_MODEL_DIR not found: ${BASE_MODEL_DIR}"; exit 1; }

declare -a TASKS=("winogrande" "hellaswag" "gsm8k_llama")

for task in "${TASKS[@]}"; do
  echo "############################"
  echo "### TASK: ${task}"
  echo "############################"

  # BASE
  base_args="$(make_model_args "${BASE_MODEL_DIR}")"
  case "${task}" in
    winogrande)
      out="${OUT_DIR}/BASE/winogrande_full"
      log="${LOG_DIR}/BASE__winogrande.log"
      cmd="lm_eval --model hf --model_args \"${base_args}\" --tasks winogrande --num_fewshot 5 --batch_size 64 --output_path \"${out}\""
      ;;
    hellaswag)
      out="${OUT_DIR}/BASE/hellaswag_full"
      log="${LOG_DIR}/BASE__hellaswag.log"
      cmd="lm_eval --model hf --model_args \"${base_args}\" --tasks hellaswag --batch_size 64 --num_fewshot 10 --output_path \"${out}\""
      ;;
    gsm8k_llama)
      out="${OUT_DIR}/BASE/gsm8k_llama_full"
      log="${LOG_DIR}/BASE__gsm8k_llama.log"
      cmd="lm_eval --model hf --model_args \"${base_args}\" --tasks gsm8k_llama --batch_size 64 --apply_chat_template --fewshot_as_multiturn --output_path \"${out}\""
      ;;
  esac

  if ! run_cmd "BASE" "${task}" "${cmd}" "${out}" "${log}"; then
    rc=$?
    if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
      echo "[WARN] BASE failed on ${task} (rc=${rc}), continue..."
    else
      exit $rc
    fi
  fi

  # 20 models
  for name in "${MODELS[@]}"; do
    model_path="${QUANT_ROOT}/${name}"
    if [[ ! -d "${model_path}" ]]; then
      echo "[WARN] Missing model dir, skip: ${model_path}"
      continue
    fi

    model_args="$(make_model_args "${model_path}")"

    case "${task}" in
      winogrande)
        out="${OUT_DIR}/${name}/winogrande_full"
        log="${LOG_DIR}/${name}__winogrande.log"
        cmd="lm_eval --model hf --model_args \"${model_args}\" --tasks winogrande --num_fewshot 5 --batch_size 64 --output_path \"${out}\""
        ;;
      hellaswag)
        out="${OUT_DIR}/${name}/hellaswag_full"
        log="${LOG_DIR}/${name}__hellaswag.log"
        cmd="lm_eval --model hf --model_args \"${model_args}\" --tasks hellaswag --batch_size 64 --num_fewshot 10 --output_path \"${out}\""
        ;;
      gsm8k_llama)
        out="${OUT_DIR}/${name}/gsm8k_llama_full"
        log="${LOG_DIR}/${name}__gsm8k_llama.log"
        cmd="lm_eval --model hf --model_args \"${model_args}\" --tasks gsm8k_llama --batch_size 64 --apply_chat_template --fewshot_as_multiturn --output_path \"${out}\""
        ;;
    esac

    if ! run_cmd "${name}" "${task}" "${cmd}" "${out}" "${log}"; then
      rc=$?
      if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
        echo "[WARN] ${name} failed on ${task} (rc=${rc}), continue..."
        continue
      else
        exit $rc
      fi
    fi
  done

  echo
done

echo "[DONE] 3 tasks completed."
echo "Results: ${OUT_DIR}/"
echo "Logs   : ${LOG_DIR}/"