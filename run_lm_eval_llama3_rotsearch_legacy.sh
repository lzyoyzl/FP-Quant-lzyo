#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# 0) 自动后台脱离（断网继续跑）
###############################################################################
AUTO_DETACH="${AUTO_DETACH:-1}"
LOG_DIR="${LOG_DIR:-lm_eval_logs_llama3_rotsearch_legacy}"
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
export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export http_proxy="${http_proxy:-http://127.0.0.1:7990}"
export https_proxy="${https_proxy:-http://127.0.0.1:7990}"

# 单一基准模型
BASE_MODEL_DIR="${BASE_MODEL_DIR:-/cephfs/shared/model/llama-3-8b-instruct}"

# 两组量化模型根目录
ROTSEARCH_ROOT="${ROTSEARCH_ROOT:-/cephfs/shared/zlouyang/FP-Quant/oyzl_test/results_quant_all_pseudoquant/llama-3-8b-instruct}"
LEGACY_ROOT="${LEGACY_ROOT:-/cephfs/shared/zlouyang/FP-Quant/oyzl_test/results_quant/llama-3-8b-instruct}"

OUT_DIR="${OUT_DIR:-lm_eval_results_llama3_rotsearch_legacy}"
mkdir -p "${OUT_DIR}"

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-True}"

CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

# 你的脚本里会 echo 这个变量；给默认值避免 set -u 报错
GSM_APPLY_CHAT_TEMPLATE="${GSM_APPLY_CHAT_TEMPLATE:-0}"

declare -a TASKS=("boolq" "arc_easy" "arc_challenge" "piqa" "winogrande" "hellaswag")

# 若 OOM 可改成 8 / 4
BATCH_SIZE="${BATCH_SIZE:-16}"
echo "[INFO] BATCH_SIZE=${BATCH_SIZE}"

###############################################################################
# 1.1) fewshot（显式写出，保证复现一致）
###############################################################################
FS_BOOLQ=0
FS_ARC_EASY=0
FS_ARC_CHALLENGE=25
FS_PIQA=0
FS_WINOGRANDE=5
FS_HELLASWAG=0

###############################################################################
# 2) 模型集合
###############################################################################
# A. 旋转搜索模型（results_quant_all_pseudoquant）
MODELS_ROTSEARCH=(
  "nvfp_rtn_search_auto_minmax_default"
  "nvfp_rtn_search_cov_minmax_default"
  "nvfp_rtn_search_auto_mse_default"
  "nvfp_rtn_search_cov_mse_default"
  "nvfp_gptq_search_auto_minmax_default"
  "nvfp_gptq_search_mse_minmax_default"
  "nvfp_gptq_search_auto_minmax_activation"
  "nvfp_gptq_search_mse_minmax_activation"
  "nvfp_gptq_search_auto_mse_default"
  "nvfp_gptq_search_mse_mse_default"
  "nvfp_gptq_search_auto_mse_activation"
  "nvfp_gptq_search_mse_mse_activation"
  "mxfp_rtn_search_auto_minmax_default"
  "mxfp_rtn_search_cov_minmax_default"
  "mxfp_rtn_search_auto_mse_default"
  "mxfp_rtn_search_cov_mse_default"
  "mxfp_gptq_search_auto_minmax_default"
  "mxfp_gptq_search_mse_minmax_default"
  "mxfp_gptq_search_auto_minmax_activation"
  "mxfp_gptq_search_mse_minmax_activation"
  "mxfp_gptq_search_auto_mse_default"
  "mxfp_gptq_search_mse_mse_default"
  "mxfp_gptq_search_auto_mse_activation"
  "mxfp_gptq_search_mse_mse_activation"
)

# B. 历史模型（results_quant）
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

###############################################################################
# 4) 跑任务：boolq, arc_easy, arc_challenge, piqa, winogrande, hellaswag
###############################################################################
echo "[INFO] BASE_MODEL_DIR=${BASE_MODEL_DIR}"
echo "[INFO] ROTSEARCH_ROOT=${ROTSEARCH_ROOT}"
echo "[INFO] LEGACY_ROOT=${LEGACY_ROOT}"
echo "[INFO] OUT_DIR=${OUT_DIR}"
echo "[INFO] LOG_DIR=${LOG_DIR}"
echo "[INFO] GSM_APPLY_CHAT_TEMPLATE=${GSM_APPLY_CHAT_TEMPLATE}"
echo

[[ -d "${BASE_MODEL_DIR}" ]] || { echo "[ERROR] BASE_MODEL_DIR not found: ${BASE_MODEL_DIR}"; exit 1; }
[[ -d "${ROTSEARCH_ROOT}" ]] || { echo "[ERROR] ROTSEARCH_ROOT not found: ${ROTSEARCH_ROOT}"; exit 1; }
[[ -d "${LEGACY_ROOT}" ]] || { echo "[ERROR] LEGACY_ROOT not found: ${LEGACY_ROOT}"; exit 1; }

for task in "${TASKS[@]}"; do
  echo "############################"
  echo "### TASK: ${task}"
  echo "############################"

  # -------- BASE --------
  base_args="$(make_model_args "${BASE_MODEL_DIR}")"
  case "${task}" in
    boolq)
      out="${OUT_DIR}/BASE/boolq_full"
      log="${LOG_DIR}/BASE__boolq.log"
      cmd="lm_eval --model hf --model_args \"${base_args}\" --tasks boolq --num_fewshot ${FS_BOOLQ} --batch_size ${BATCH_SIZE} --output_path \"${out}\""
      ;;
    arc_easy)
      out="${OUT_DIR}/BASE/arc_easy_full"
      log="${LOG_DIR}/BASE__arc_easy.log"
      cmd="lm_eval --model hf --model_args \"${base_args}\" --tasks arc_easy --num_fewshot ${FS_ARC_EASY} --batch_size ${BATCH_SIZE} --output_path \"${out}\""
      ;;
    arc_challenge)
      out="${OUT_DIR}/BASE/arc_challenge_full"
      log="${LOG_DIR}/BASE__arc_challenge.log"
      cmd="lm_eval --model hf --model_args \"${base_args}\" --tasks arc_challenge --num_fewshot ${FS_ARC_CHALLENGE} --batch_size ${BATCH_SIZE} --output_path \"${out}\""
      ;;
    piqa)
      out="${OUT_DIR}/BASE/piqa_full"
      log="${LOG_DIR}/BASE__piqa.log"
      cmd="lm_eval --model hf --model_args \"${base_args}\" --tasks piqa --num_fewshot ${FS_PIQA} --batch_size ${BATCH_SIZE} --output_path \"${out}\""
      ;;
    winogrande)
      out="${OUT_DIR}/BASE/winogrande_full"
      log="${LOG_DIR}/BASE__winogrande.log"
      cmd="lm_eval --model hf --model_args \"${base_args}\" --tasks winogrande --num_fewshot ${FS_WINOGRANDE} --batch_size ${BATCH_SIZE} --output_path \"${out}\""
      ;;
    hellaswag)
      out="${OUT_DIR}/BASE/hellaswag_full"
      log="${LOG_DIR}/BASE__hellaswag.log"
      cmd="lm_eval --model hf --model_args \"${base_args}\" --tasks hellaswag --num_fewshot ${FS_HELLASWAG} --batch_size ${BATCH_SIZE} --output_path \"${out}\""
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

  # -------- ROTSEARCH 模型（日志前缀 ROTSEARCH__）--------
  for name in "${MODELS_ROTSEARCH[@]}"; do
    model_path="${ROTSEARCH_ROOT}/${name}"
    if [[ ! -d "${model_path}" ]]; then
      echo "[WARN] Missing ROTSEARCH model dir, skip: ${model_path}"
      continue
    fi

    model_args="$(make_model_args "${model_path}")"
    case "${task}" in
      boolq)
        out="${OUT_DIR}/ROTSEARCH/${name}/boolq_full"
        log="${LOG_DIR}/ROTSEARCH__${name}__boolq.log"
        cmd="lm_eval --model hf --model_args \"${model_args}\" --tasks boolq --num_fewshot ${FS_BOOLQ} --batch_size ${BATCH_SIZE} --output_path \"${out}\""
        ;;
      arc_easy)
        out="${OUT_DIR}/ROTSEARCH/${name}/arc_easy_full"
        log="${LOG_DIR}/ROTSEARCH__${name}__arc_easy.log"
        cmd="lm_eval --model hf --model_args \"${model_args}\" --tasks arc_easy --num_fewshot ${FS_ARC_EASY} --batch_size ${BATCH_SIZE} --output_path \"${out}\""
        ;;
      arc_challenge)
        out="${OUT_DIR}/ROTSEARCH/${name}/arc_challenge_full"
        log="${LOG_DIR}/ROTSEARCH__${name}__arc_challenge.log"
        cmd="lm_eval --model hf --model_args \"${model_args}\" --tasks arc_challenge --num_fewshot ${FS_ARC_CHALLENGE} --batch_size ${BATCH_SIZE} --output_path \"${out}\""
        ;;
      piqa)
        out="${OUT_DIR}/ROTSEARCH/${name}/piqa_full"
        log="${LOG_DIR}/ROTSEARCH__${name}__piqa.log"
        cmd="lm_eval --model hf --model_args \"${model_args}\" --tasks piqa --num_fewshot ${FS_PIQA} --batch_size ${BATCH_SIZE} --output_path \"${out}\""
        ;;
      winogrande)
        out="${OUT_DIR}/ROTSEARCH/${name}/winogrande_full"
        log="${LOG_DIR}/ROTSEARCH__${name}__winogrande.log"
        cmd="lm_eval --model hf --model_args \"${model_args}\" --tasks winogrande --num_fewshot ${FS_WINOGRANDE} --batch_size ${BATCH_SIZE} --output_path \"${out}\""
        ;;
      hellaswag)
        out="${OUT_DIR}/ROTSEARCH/${name}/hellaswag_full"
        log="${LOG_DIR}/ROTSEARCH__${name}__hellaswag.log"
        cmd="lm_eval --model hf --model_args \"${model_args}\" --tasks hellaswag --num_fewshot ${FS_HELLASWAG} --batch_size ${BATCH_SIZE} --output_path \"${out}\""
        ;;
    esac

    if ! run_cmd "ROTSEARCH__${name}" "${task}" "${cmd}" "${out}" "${log}"; then
      rc=$?
      if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
        echo "[WARN] ROTSEARCH ${name} failed on ${task} (rc=${rc}), continue..."
        continue
      else
        exit $rc
      fi
    fi
  done

  # -------- LEGACY 模型（日志前缀 LEGACY__）--------
  for name in "${MODELS_LEGACY[@]}"; do
    model_path="${LEGACY_ROOT}/${name}"
    if [[ ! -d "${model_path}" ]]; then
      echo "[WARN] Missing LEGACY model dir, skip: ${model_path}"
      continue
    fi

    model_args="$(make_model_args "${model_path}")"
    case "${task}" in
      boolq)
        out="${OUT_DIR}/LEGACY/${name}/boolq_full"
        log="${LOG_DIR}/LEGACY__${name}__boolq.log"
        cmd="lm_eval --model hf --model_args \"${model_args}\" --tasks boolq --num_fewshot ${FS_BOOLQ} --batch_size ${BATCH_SIZE} --output_path \"${out}\""
        ;;
      arc_easy)
        out="${OUT_DIR}/LEGACY/${name}/arc_easy_full"
        log="${LOG_DIR}/LEGACY__${name}__arc_easy.log"
        cmd="lm_eval --model hf --model_args \"${model_args}\" --tasks arc_easy --num_fewshot ${FS_ARC_EASY} --batch_size ${BATCH_SIZE} --output_path \"${out}\""
        ;;
      arc_challenge)
        out="${OUT_DIR}/LEGACY/${name}/arc_challenge_full"
        log="${LOG_DIR}/LEGACY__${name}__arc_challenge.log"
        cmd="lm_eval --model hf --model_args \"${model_args}\" --tasks arc_challenge --num_fewshot ${FS_ARC_CHALLENGE} --batch_size ${BATCH_SIZE} --output_path \"${out}\""
        ;;
      piqa)
        out="${OUT_DIR}/LEGACY/${name}/piqa_full"
        log="${LOG_DIR}/LEGACY__${name}__piqa.log"
        cmd="lm_eval --model hf --model_args \"${model_args}\" --tasks piqa --num_fewshot ${FS_PIQA} --batch_size ${BATCH_SIZE} --output_path \"${out}\""
        ;;
      winogrande)
        out="${OUT_DIR}/LEGACY/${name}/winogrande_full"
        log="${LOG_DIR}/LEGACY__${name}__winogrande.log"
        cmd="lm_eval --model hf --model_args \"${model_args}\" --tasks winogrande --num_fewshot ${FS_WINOGRANDE} --batch_size ${BATCH_SIZE} --output_path \"${out}\""
        ;;
      hellaswag)
        out="${OUT_DIR}/LEGACY/${name}/hellaswag_full"
        log="${LOG_DIR}/LEGACY__${name}__hellaswag.log"
        cmd="lm_eval --model hf --model_args \"${model_args}\" --tasks hellaswag --num_fewshot ${FS_HELLASWAG} --batch_size ${BATCH_SIZE} --output_path \"${out}\""
        ;;
    esac

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

echo "[DONE] 6 tasks completed."
echo "Results: ${OUT_DIR}/"
echo "Logs   : ${LOG_DIR}/"


