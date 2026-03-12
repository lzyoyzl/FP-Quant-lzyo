#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# 0) 可选：加载你的环境变量（HF cache、OUT_CALIB 等）
#    如果没有 env_var.sh，这行不会报错
###############################################################################
if [[ -f "env_var.sh" ]]; then
  # shellcheck disable=SC1091
  source "env_var.sh"
fi

###############################################################################
# 1) 断线续跑：自动后台脱离（可用 AUTO_DETACH=0 关闭）
###############################################################################
AUTO_DETACH="${AUTO_DETACH:-1}"
LOG_DIR="${LOG_DIR:-logs}"
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
# 2) 基本资源设置（可按需改）
###############################################################################
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

###############################################################################
# 3) 你需要按自己路径确认/填写的关键变量
###############################################################################
# (1) 模型路径：改为多个模型（也允许你用环境变量覆盖）
#     用法示例：
#       MODEL_DIRS="/path/to/llama2-7b-hf /path/to/Qwen3-8B" bash run.sh
if [[ -n "${MODEL_DIRS:-}" ]]; then
  # shellcheck disable=SC2206
  MODEL_DIRS_ARR=(${MODEL_DIRS})
else
  MODEL_DIRS_ARR=(
    "/cephfs/shared/model/llama-2-7b-hf"
    "/cephfs/shared/model/Qwen3-8B"
    "/cephfs/shared/model/llama-3-8b-hf"
  )
fi

# (2) 输出根目录：导出的量化模型会放这里
OUT_ROOT="${OUT_ROOT:-/cephfs/shared/zlouyang/FP-Quant/oyzl_test/results_quant}"

###############################################################################
# 4) 其他参数（一般不用动）
###############################################################################
SEQ_LEN="${SEQ_LEN:-2048}"
N_SEQS="${N_SEQS:-1024}"
SEED="${SEED:-0}"
DTYPE="${DTYPE:-auto}"

EXPORT_MODE="${EXPORT_MODE:-pseudoquant}"

HAD_LARGE="${HAD_LARGE:-128}"

CPU_OFFLOAD_MODULES="${CPU_OFFLOAD_MODULES:-0}"
CPU_OFFLOAD_ACTIVATIONS="${CPU_OFFLOAD_ACTIVATIONS:-0}"
OFFLOAD_ARGS=()
[[ "${CPU_OFFLOAD_MODULES}" == "1" ]] && OFFLOAD_ARGS+=(--cpu_offload_modules)
[[ "${CPU_OFFLOAD_ACTIVATIONS}" == "1" ]] && OFFLOAD_ARGS+=(--cpu_offload_activations)

###############################################################################
# 4.5) 校准数据：按“模型”自动匹配到 datasets/calib 下的对应目录
#      你当前已有：
#        fineweb_1024x2048_Qwen3_8B
#        fineweb_1024x2048_llama3_8b_inst
#        fineweb_1024x2048_llama_2_7b_hf
#        fineweb_1024x2048_llama_3_8b
#
#      规则：
#        - 默认使用 tokens.pt；不存在则用 text.jsonl；都不存在按 ALLOW_HF_FALLBACK 决定是否回退 HF
#        - 如你显式设置 CALIB_PT / CALIB_JSONL，则会覆盖自动选择（对所有模型生效）
###############################################################################
CALIB_ROOT="${CALIB_ROOT:-/cephfs/shared/zlouyang/FP-Quant/oyzl_test/datasets/calib}"
ALLOW_HF_FALLBACK="${ALLOW_HF_FALLBACK:-0}"
CALIB_HF="${CALIB_HF:-HuggingFaceFW/fineweb-edu}"

calib_subdir_for_model_id () {
  local mid="$1"
  case "${mid}" in
    "Qwen3-8B")
      echo "fineweb_1024x2048_Qwen3_8B"
      ;;
    "llama-2-7b-hf")
      echo "fineweb_1024x2048_llama_2_7b_hf"
      ;;
    "llama-3-8b-hf")
      echo "fineweb_1024x2048_llama_3_8b"
      ;;
    # 如果你未来把 instruct 也加入 MODEL_DIRS_ARR，可直接匹配它
    "llama-3-8b-instruct")
      echo "fineweb_1024x2048_llama3_8b_inst"
      ;;
    *)
      return 1
      ;;
  esac
}

###############################################################################
# 5) 前置检查：lm_eval 必须能 import（因为 model_quant.py 顶部硬 import）
###############################################################################
python -c "import lm_eval" >/dev/null 2>&1 || {
  echo "[ERROR] Cannot import lm_eval in current env."
  echo "        Fix one of the following:"
  echo "          (1) Install: python -m pip install lm-eval"
  echo "          (2) Or modify model_quant.py: move lm_eval imports inside 'if args.eval_openllm:' block."
  exit 1
}

###############################################################################
# 6) 断点续跑：判断某个实验是否已成功导出
###############################################################################
is_done () {
  local outdir="$1"
  [[ -f "${outdir}/config.json" ]] || return 1
  [[ -f "${outdir}/model.safetensors.index.json" ]] || return 1
  ls "${outdir}"/model-*-of-*.safetensors >/dev/null 2>&1 || return 1
  return 0
}

###############################################################################
# 7) 运行单个实验：每个实验一个日志文件，失败会记录到 logs/failed.txt
###############################################################################
FAILED_LIST="${LOG_DIR}/failed.txt"
DONE_LIST="${LOG_DIR}/done.txt"
: > "${FAILED_LIST}" || true
: > "${DONE_LIST}" || true

run_one () {
  local model_dir="$1"; shift
  local name="$1"; shift
  local fmt="$1"; shift

  local model_id
  model_id="$(basename "${model_dir}")"

  local outdir="${OUT_ROOT}/${model_id}/${name}"
  local logfile="${LOG_DIR}/${model_id}__${name}.log"

  mkdir -p "${OUT_ROOT}/${model_id}"

  if is_done "${outdir}"; then
    echo "[SKIP] ${model_id} :: ${name} already exported: ${outdir}"
    echo "${model_id}__${name}" >> "${DONE_LIST}"
    return 0
  fi

  mkdir -p "${outdir}"
  echo "=== EXPORT: ${model_id} :: ${name} ==="
  echo "  model : ${model_dir}"
  echo "  calib : ${CALIB_DATASET}"
  echo "  out   : ${outdir}"
  echo "  log   : ${logfile}"
  echo

  {
    echo "### START $(date) ###"
    echo "MODEL_ID=${model_id}"
    echo "NAME=${name}"
    echo "FORMAT=${fmt}"
    echo "MODEL_DIR=${model_dir}"
    echo "CALIB=${CALIB_DATASET}"
    echo "SEQ_LEN=${SEQ_LEN}  N_SEQS=${N_SEQS}  SEED=${SEED}  DTYPE=${DTYPE}"
    echo "EXPORT_MODE=${EXPORT_MODE}"
    echo "OFFLOAD_ARGS=${OFFLOAD_ARGS[*]:-<none>}"
    echo
  } >> "${logfile}"

  set +e
  stdbuf -oL -eL python /cephfs/shared/zlouyang/FP-Quant/model_quant.py \
    --model_name_or_path "${model_dir}" \
    --dataset_name_or_path "${CALIB_DATASET}" \
    --sequence_length "${SEQ_LEN}" \
    --num_sequences "${N_SEQS}" \
    --seed "${SEED}" \
    --dtype "${DTYPE}" \
    --format "${fmt}" \
    --w_bits 4 \
    --a_bits 4 \
    --w_group_size 32 \
    --a_group_size 32 \
    --export_quantized_model "${EXPORT_MODE}" \
    --save_path "${outdir}" \
    --fuse_global_scale \
    --amp \
    "${OFFLOAD_ARGS[@]}" \
    "$@" >> "${logfile}" 2>&1
  rc=$?
  set -e

  if [[ $rc -ne 0 ]]; then
    echo "[FAIL] ${model_id} :: ${name} (rc=${rc}). See log: ${logfile}"
    echo "${model_id}__${name} rc=${rc} log=${logfile}" >> "${FAILED_LIST}"
    return 1
  fi

  if is_done "${outdir}"; then
    echo "[OK] ${model_id} :: ${name} exported to: ${outdir}"
    echo "${model_id}__${name}" >> "${DONE_LIST}"
    return 0
  else
    echo "[FAIL] ${model_id} :: ${name} finished but output incomplete. See log: ${logfile}"
    echo "${model_id}__${name} rc=0 but incomplete log=${logfile}" >> "${FAILED_LIST}"
    return 1
  fi
}

###############################################################################
# 8) 只跑你指定的两个配置（MXFP）
###############################################################################
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

run_or_continue () {
  if ! run_one "$@"; then
    if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
      echo "[WARN] Continue on error enabled; moving to next experiment."
      return 0
    else
      echo "[ERROR] Stopping due to failure. Set CONTINUE_ON_ERROR=1 to continue."
      exit 1
    fi
  fi
}

for MODEL_DIR in "${MODEL_DIRS_ARR[@]}"; do
  if [[ ! -d "${MODEL_DIR}" ]]; then
    echo "[ERROR] MODEL_DIR not found: ${MODEL_DIR}"
    exit 1
  fi

  MODEL_ID="$(basename "${MODEL_DIR}")"

  # --------- 自动选择该模型对应的校准数据 ---------
  if [[ -n "${CALIB_PT:-}" || -n "${CALIB_JSONL:-}" ]]; then
    # 用户显式指定则全局覆盖（保持你原来逻辑的可控性）
    calib_pt="${CALIB_PT:-}"
    calib_jsonl="${CALIB_JSONL:-}"
  else
    calib_subdir="$(calib_subdir_for_model_id "${MODEL_ID}")" || {
      echo "[ERROR] No calib mapping for model_id='${MODEL_ID}'."
      echo "        Please add a case in calib_subdir_for_model_id(), or set CALIB_PT/CALIB_JSONL explicitly."
      exit 1
    }
    calib_pt="${CALIB_ROOT}/${calib_subdir}/fineweb_calib_${N_SEQS}x${SEQ_LEN}_tokens.pt"
    calib_jsonl="${CALIB_ROOT}/${calib_subdir}/fineweb_calib_${N_SEQS}x${SEQ_LEN}_text.jsonl"
  fi

  if [[ -n "${calib_pt}" && -f "${calib_pt}" ]]; then
    CALIB_DATASET="${calib_pt}"
  elif [[ -n "${calib_jsonl}" && -f "${calib_jsonl}" ]]; then
    CALIB_DATASET="${calib_jsonl}"
  else
    if [[ "${ALLOW_HF_FALLBACK}" == "1" ]]; then
      CALIB_DATASET="${CALIB_HF}"
    else
      echo "[ERROR] Calib file not found for model '${MODEL_ID}'."
      echo "        Tried:"
      echo "          - ${calib_pt}"
      echo "          - ${calib_jsonl}"
      echo "        Either generate the calib set, or set ALLOW_HF_FALLBACK=1, or set CALIB_PT/CALIB_JSONL explicitly."
      exit 1
    fi
  fi

  echo "[INFO] Using calib for ${MODEL_ID}: ${CALIB_DATASET}"

  # 1) mxfp_gptq_w4a4_hadamard_h128_mse_default
  run_or_continue "${MODEL_DIR}" "mxfp_gptq_w4a4_hadamard_h${HAD_LARGE}_mse_default" mxfp \
    --transform_class hadamard --hadamard_group_size "${HAD_LARGE}" \
    --w_observer mse \
    --quantization_order default \
    --gptq

  # 2) mxfp_mrgptq_w4a4_hadamard_h128_mse_activation
  run_or_continue "${MODEL_DIR}" "mxfp_mrgptq_w4a4_hadamard_h${HAD_LARGE}_mse_activation" mxfp \
    --transform_class hadamard --hadamard_group_size "${HAD_LARGE}" \
    --w_observer mse \
    --quantization_order activation \
    --gptq
done

echo
echo "[DONE] All requested experiments processed."
echo "Outputs: ${OUT_ROOT}/<model_id>/"
echo "Logs   : ${LOG_DIR}/"
echo "Done list   : ${DONE_LIST}"
echo "Failed list : ${FAILED_LIST}"

