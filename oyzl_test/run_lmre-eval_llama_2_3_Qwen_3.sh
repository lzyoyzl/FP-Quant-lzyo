#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# 0) 自动后台脱离（断网继续跑）
###############################################################################
AUTO_DETACH="${AUTO_DETACH:-1}"
LOG_DIR="${LOG_DIR:-lm_re-eval_logs_tasks_llama_2_3-Qwen_3}"
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
export HF_ENDPOINT="https://huggingface.co"
export http_proxy="http://127.0.0.1:7990"
export https_proxy="http://127.0.0.1:7990"

RESULTS_ROOT="${RESULTS_ROOT:-/cephfs/shared/zlouyang/FP-Quant/oyzl_test/results_quant}"
BASE_ROOT="${BASE_ROOT:-/cephfs/shared/model}"

OUT_DIR="${OUT_DIR:-lm_re-eval_results_tasks_llama_2_3-Qwen_3}"
mkdir -p "${OUT_DIR}"

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-True}"

CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

# 你的脚本里会 echo 这个变量；给个默认值避免 set -u 报错（不影响任务）
GSM_APPLY_CHAT_TEMPLATE="${GSM_APPLY_CHAT_TEMPLATE:-0}"

# 调整任务列表：只跑 arc_challenge 和 winogrande  # CHANGED
declare -a TASKS=("arc_challenge" "winogrande")  # CHANGED

# batch size（你原本设定保留；若 OOM 再往下调 8/4）
BATCH_SIZE=16
echo "[INFO] BATCH_SIZE=${BATCH_SIZE}"

###############################################################################
# 1.1) fewshot 数：arc_challenge 和 winogrande 都设为 0  # CHANGED
###############################################################################
FS_BOOLQ=0
FS_ARC_EASY=0
FS_ARC_CHALLENGE=0     # CHANGED
FS_PIQA=0
FS_WINOGRANDE=0        # CHANGED
FS_HELLASWAG=10

###############################################################################
# 2) 模型集合
###############################################################################
MODEL_IDS=(
  "llama-2-7b-hf"
  "llama-3-8b-hf"
  "Qwen3-8B"
)

MODELS=(
  "mxfp_gptq_w4a4_hadamard_h128_mse_default"
  "mxfp_mrgptq_w4a4_hadamard_h128_mse_activation"
)

###############################################################################
# 3) 工具函数（原样保留）
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
# 4) 跑任务：arc_challenge, winogrande
###############################################################################
echo "[INFO] RESULTS_ROOT=${RESULTS_ROOT}"
echo "[INFO] BASE_ROOT=${BASE_ROOT}"
echo "[INFO] OUT_DIR=${OUT_DIR}"
echo "[INFO] LOG_DIR=${LOG_DIR}"
echo "[INFO] GSM_APPLY_CHAT_TEMPLATE=${GSM_APPLY_CHAT_TEMPLATE}"
echo

for task in "${TASKS[@]}"; do
  echo "############################"
  echo "### TASK: ${task}"
  echo "############################"

  for model_id in "${MODEL_IDS[@]}"; do
    BASE_MODEL_DIR="${BASE_ROOT}/${model_id}"
    QUANT_ROOT="${RESULTS_ROOT}/${model_id}"

    [[ -d "${BASE_MODEL_DIR}" ]] || { echo "[ERROR] BASE_MODEL_DIR not found: ${BASE_MODEL_DIR}"; exit 1; }
    [[ -d "${QUANT_ROOT}" ]] || { echo "[ERROR] QUANT_ROOT not found: ${QUANT_ROOT}"; exit 1; }

    # -------- BASE --------
    base_args="$(make_model_args "${BASE_MODEL_DIR}")"
    case "${task}" in
      arc_challenge)
        out="${OUT_DIR}/${model_id}/BASE/arc_challenge_full"
        log="${LOG_DIR}/${model_id}__BASE__arc_challenge.log"
        cmd="lm_eval --model hf --model_args \"${base_args}\" --tasks arc_challenge --num_fewshot ${FS_ARC_CHALLENGE} --batch_size ${BATCH_SIZE} --output_path \"${out}\""
        ;;
      winogrande)
        out="${OUT_DIR}/${model_id}/BASE/winogrande_full"
        log="${LOG_DIR}/${model_id}__BASE__winogrande.log"
        cmd="lm_eval --model hf --model_args \"${base_args}\" --tasks winogrande --num_fewshot ${FS_WINOGRANDE} --batch_size ${BATCH_SIZE} --output_path \"${out}\""
        ;;
    esac

    if ! run_cmd "${model_id}__BASE" "${task}" "${cmd}" "${out}" "${log}"; then
      rc=$?
      if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
        echo "[WARN] ${model_id} BASE failed on ${task} (rc=${rc}), continue..."
      else
        exit $rc
      fi
    fi

    # -------- 量化模型（两个目录）--------
    for name in "${MODELS[@]}"; do
      model_path="${QUANT_ROOT}/${name}"
      if [[ ! -d "${model_path}" ]]; then
        echo "[WARN] Missing model dir, skip: ${model_path}"
        continue
      fi

      model_args="$(make_model_args "${model_path}")"

      case "${task}" in
        arc_challenge)
          out="${OUT_DIR}/${model_id}/${name}/arc_challenge_full"
          log="${LOG_DIR}/${model_id}__${name}__arc_challenge.log"
          cmd="lm_eval --model hf --model_args \"${model_args}\" --tasks arc_challenge --num_fewshot ${FS_ARC_CHALLENGE} --batch_size ${BATCH_SIZE} --output_path \"${out}\""
          ;;
        winogrande)
          out="${OUT_DIR}/${model_id}/${name}/winogrande_full"
          log="${LOG_DIR}/${model_id}__${name}__winogrande.log"
          cmd="lm_eval --model hf --model_args \"${model_args}\" --tasks winogrande --num_fewshot ${FS_WINOGRANDE} --batch_size ${BATCH_SIZE} --output_path \"${out}\""
          ;;
      esac

      if ! run_cmd "${model_id}__${name}" "${task}" "${cmd}" "${out}" "${log}"; then
        rc=$?
        if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
          echo "[WARN] ${model_id} ${name} failed on ${task} (rc=${rc}), continue..."
          continue
        else
          exit $rc
        fi
      fi
    done

    echo
  done
done

echo "[DONE] 2 tasks completed."
echo "Results: ${OUT_DIR}/"
echo "Logs   : ${LOG_DIR}/"

