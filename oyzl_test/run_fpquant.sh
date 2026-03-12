#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# 0) 可选：加载你的环境变量（HF cache、OUT_CALIB 等）
#    如果没有 00_env.sh，这行不会报错
###############################################################################
if [[ -f "env_var.sh" ]]; then
  # shellcheck disable=SC1091
  source "env_var.sh"
fi

###############################################################################
# 1) 断线续跑：自动后台脱离（可用 AUTO_DETACH=0 关闭）
#    - 你在 SSH 里直接运行本脚本，它会自己 nohup & disown，
#      即使网络断开也会继续跑。
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
# (1) 模型路径：你本地已有的模型之一
MODEL_DIR="${MODEL_DIR:-/cephfs/shared/model/llama-3-8b-instruct}"

# (2) 校准数据：优先用本地 tokens.pt（最稳定/最快/严格一致），其次 jsonl，最后可选 HF fallback
CALIB_PT_DEFAULT=""
CALIB_JSONL_DEFAULT=""

if [[ -n "${OUT_CALIB:-}" ]]; then
  CALIB_PT_DEFAULT="${OUT_CALIB}/fineweb_calib_1024x2048_tokens.pt"
  CALIB_JSONL_DEFAULT="${OUT_CALIB}/fineweb_calib_1024x2048_text.jsonl"
fi

# 允许用户用环境变量显式指定（不指定则用默认拼出来的）
CALIB_PT="${CALIB_PT:-${CALIB_PT_DEFAULT:-}}"
CALIB_JSONL="${CALIB_JSONL:-${CALIB_JSONL_DEFAULT:-}}"

# (可选) 允许回退到 HF streaming（默认不允许，避免断网导致失败）
ALLOW_HF_FALLBACK="${ALLOW_HF_FALLBACK:-0}"
CALIB_HF="${CALIB_HF:-HuggingFaceFW/fineweb-edu}"

# 选择校准来源：pt > jsonl > hf
if [[ -n "${CALIB_PT}" && -f "${CALIB_PT}" ]]; then
  CALIB_DATASET="${CALIB_PT}"
elif [[ -n "${CALIB_JSONL}" && -f "${CALIB_JSONL}" ]]; then
  CALIB_DATASET="${CALIB_JSONL}"
else
  if [[ "${ALLOW_HF_FALLBACK}" == "1" ]]; then
    CALIB_DATASET="${CALIB_HF}"
  else
    echo "[ERROR] Local calib file not found (.pt/.jsonl)."
    echo "        Set CALIB_PT or CALIB_JSONL to your local file path, or set ALLOW_HF_FALLBACK=1 to use HF dataset."
    echo "        Current CALIB_PT='${CALIB_PT}'"
    echo "        Current CALIB_JSONL='${CALIB_JSONL}'"
    exit 1
  fi
fi








# (3) 输出根目录：导出的量化模型会放这里
OUT_ROOT="${OUT_ROOT:-/cephfs/shared/zlouyang/FP-Quant/oyzl_test/results_quant}"

###############################################################################
# 4) 其他参数（一般不用动）
###############################################################################
MODEL_ID="$(basename "${MODEL_DIR}")"
SEQ_LEN="${SEQ_LEN:-2048}"
N_SEQS="${N_SEQS:-1024}"
SEED="${SEED:-0}"
DTYPE="${DTYPE:-auto}"

# 导出模式：A100 建议 pseudoquant（realquant 主要面向 Blackwell + QuTLASS）
EXPORT_MODE="${EXPORT_MODE:-pseudoquant}"

# Hadamard 组大小：你两份脚本里出现过 16/32/128，这里都跑到
HAD_SMALL_NVFP="${HAD_SMALL_NVFP:-16}"
HAD_SMALL_MXFP="${HAD_SMALL_MXFP:-32}"
HAD_LARGE="${HAD_LARGE:-128}"

# 是否使用 CPU offload（默认关；A100 80GB 跑 8B 通常不需要，开了会慢）
CPU_OFFLOAD_MODULES="${CPU_OFFLOAD_MODULES:-0}"
CPU_OFFLOAD_ACTIVATIONS="${CPU_OFFLOAD_ACTIVATIONS:-0}"
OFFLOAD_ARGS=()
[[ "${CPU_OFFLOAD_MODULES}" == "1" ]] && OFFLOAD_ARGS+=(--cpu_offload_modules)
[[ "${CPU_OFFLOAD_ACTIVATIONS}" == "1" ]] && OFFLOAD_ARGS+=(--cpu_offload_activations)

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
#    导出成功通常至少包含：
#      - config.json
#      - model.safetensors.index.json
#      - >=1 个 model-*-of-*.safetensors
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
mkdir -p "${OUT_ROOT}/${MODEL_ID}"
: > "${FAILED_LIST}" || true
: > "${DONE_LIST}" || true

run_one () {
  local name="$1"; shift
  local fmt="$1"; shift
  local outdir="${OUT_ROOT}/${MODEL_ID}/${name}"
  local logfile="${LOG_DIR}/${MODEL_ID}__${name}.log"

  if is_done "${outdir}"; then
    echo "[SKIP] ${name} already exported: ${outdir}"
    echo "${name}" >> "${DONE_LIST}"
    return 0
  fi

  mkdir -p "${outdir}"
  echo "=== EXPORT: ${name} ==="
  echo "  model : ${MODEL_DIR}"
  echo "  calib : ${CALIB_DATASET}"
  echo "  out   : ${outdir}"
  echo "  log   : ${logfile}"
  echo

  # 记录运行参数到日志头部
  {
    echo "### START $(date) ###"
    echo "NAME=${name}"
    echo "FORMAT=${fmt}"
    echo "MODEL_DIR=${MODEL_DIR}"
    echo "CALIB=${CALIB_DATASET}"
    echo "SEQ_LEN=${SEQ_LEN}  N_SEQS=${N_SEQS}  SEED=${SEED}  DTYPE=${DTYPE}"
    echo "EXPORT_MODE=${EXPORT_MODE}"
    echo "OFFLOAD_ARGS=${OFFLOAD_ARGS[*]:-<none>}"
    echo
  } >> "${logfile}"

  set +e
  stdbuf -oL -eL python /cephfs/shared/zlouyang/FP-Quant/model_quant.py \
    --model_name_or_path "${MODEL_DIR}" \
    --dataset_name_or_path "${CALIB_DATASET}" \
    --sequence_length "${SEQ_LEN}" \
    --num_sequences "${N_SEQS}" \
    --seed "${SEED}" \
    --dtype "${DTYPE}" \
    --format "${fmt}" \
    --w_bits 4 \
    --a_bits 4 \
    --w_group_size $([[ "${fmt}" == "nvfp" ]] && echo 16 || echo 32) \
    --a_group_size $([[ "${fmt}" == "nvfp" ]] && echo 16 || echo 32) \
    --export_quantized_model "${EXPORT_MODE}" \
    --save_path "${outdir}" \
    --fuse_global_scale \
    --amp \
    "${OFFLOAD_ARGS[@]}" \
    "$@" >> "${logfile}" 2>&1
  rc=$?
  set -e

  if [[ $rc -ne 0 ]]; then
    echo "[FAIL] ${name} (rc=${rc}). See log: ${logfile}"
    echo "${name} rc=${rc} log=${logfile}" >> "${FAILED_LIST}"
    return 1
  fi

  if is_done "${outdir}"; then
    echo "[OK] ${name} exported to: ${outdir}"
    echo "${name}" >> "${DONE_LIST}"
    return 0
  else
    echo "[FAIL] ${name} finished but output incomplete. See log: ${logfile}"
    echo "${name} rc=0 but incomplete log=${logfile}" >> "${FAILED_LIST}"
    return 1
  fi
}

###############################################################################
# 8) 实验集合：= 你的 8 个 + 消融脚本的所有实验（去重后并集）
#    注意：quantization_order 在源码里只接受 default/activation（不是 actorder）
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

# ---------------- NVFP4 ----------------
# 你的 8 实验里的 NVFP：RTN(identity), RTN+Had(h16), GPTQ(identity), MR-GPTQ(h128+mse+activation)
run_or_continue "nvfp_rtn_w4a4_identity" nvfp \
  --transform_class identity \
  --w_observer minmax \
  --quantization_order default

run_or_continue "nvfp_rtn_w4a4_hadamard_h${HAD_SMALL_NVFP}" nvfp \
  --transform_class hadamard --hadamard_group_size "${HAD_SMALL_NVFP}" \
  --w_observer minmax \
  --quantization_order default

# 消融脚本补充：RTN+Had(h128)
run_or_continue "nvfp_rtn_w4a4_hadamard_h${HAD_LARGE}" nvfp \
  --transform_class hadamard --hadamard_group_size "${HAD_LARGE}" \
  --w_observer minmax \
  --quantization_order default

run_or_continue "nvfp_gptq_w4a4_identity" nvfp \
  --transform_class identity \
  --w_observer minmax \
  --quantization_order default \
  --gptq

# 消融：GPTQ + Hadamard（minmax + default）
run_or_continue "nvfp_gptq_w4a4_hadamard_h${HAD_LARGE}_minmax_default" nvfp \
  --transform_class hadamard --hadamard_group_size "${HAD_LARGE}" \
  --w_observer minmax \
  --quantization_order default \
  --gptq

# 消融：GPTQ + MSE（无旋转）
run_or_continue "nvfp_gptq_w4a4_identity_mse_default" nvfp \
  --transform_class identity \
  --w_observer mse \
  --quantization_order default \
  --gptq

# 消融：GPTQ + activation order（无旋转）
run_or_continue "nvfp_gptq_w4a4_identity_minmax_activation" nvfp \
  --transform_class identity \
  --w_observer minmax \
  --quantization_order activation \
  --gptq

# 消融：GPTQ + Hadamard + MSE（无 activation）
run_or_continue "nvfp_gptq_w4a4_hadamard_h${HAD_LARGE}_mse_default" nvfp \
  --transform_class hadamard --hadamard_group_size "${HAD_LARGE}" \
  --w_observer mse \
  --quantization_order default \
  --gptq

# 消融：GPTQ + Hadamard + activation（无 MSE）
run_or_continue "nvfp_gptq_w4a4_hadamard_h${HAD_LARGE}_minmax_activation" nvfp \
  --transform_class hadamard --hadamard_group_size "${HAD_LARGE}" \
  --w_observer minmax \
  --quantization_order activation \
  --gptq

# MR-GPTQ（完整三件套）
run_or_continue "nvfp_mrgptq_w4a4_hadamard_h${HAD_LARGE}_mse_activation" nvfp \
  --transform_class hadamard --hadamard_group_size "${HAD_LARGE}" \
  --w_observer mse \
  --quantization_order activation \
  --gptq


# ---------------- MXFP4 ----------------
run_or_continue "mxfp_rtn_w4a4_identity" mxfp \
  --transform_class identity \
  --w_observer minmax \
  --quantization_order default

run_or_continue "mxfp_rtn_w4a4_hadamard_h${HAD_SMALL_MXFP}" mxfp \
  --transform_class hadamard --hadamard_group_size "${HAD_SMALL_MXFP}" \
  --w_observer minmax \
  --quantization_order default

run_or_continue "mxfp_rtn_w4a4_hadamard_h${HAD_LARGE}" mxfp \
  --transform_class hadamard --hadamard_group_size "${HAD_LARGE}" \
  --w_observer minmax \
  --quantization_order default

run_or_continue "mxfp_gptq_w4a4_identity" mxfp \
  --transform_class identity \
  --w_observer minmax \
  --quantization_order default \
  --gptq

run_or_continue "mxfp_gptq_w4a4_hadamard_h${HAD_LARGE}_minmax_default" mxfp \
  --transform_class hadamard --hadamard_group_size "${HAD_LARGE}" \
  --w_observer minmax \
  --quantization_order default \
  --gptq

run_or_continue "mxfp_gptq_w4a4_identity_mse_default" mxfp \
  --transform_class identity \
  --w_observer mse \
  --quantization_order default \
  --gptq

run_or_continue "mxfp_gptq_w4a4_identity_minmax_activation" mxfp \
  --transform_class identity \
  --w_observer minmax \
  --quantization_order activation \
  --gptq

run_or_continue "mxfp_gptq_w4a4_hadamard_h${HAD_LARGE}_mse_default" mxfp \
  --transform_class hadamard --hadamard_group_size "${HAD_LARGE}" \
  --w_observer mse \
  --quantization_order default \
  --gptq

run_or_continue "mxfp_gptq_w4a4_hadamard_h${HAD_LARGE}_minmax_activation" mxfp \
  --transform_class hadamard --hadamard_group_size "${HAD_LARGE}" \
  --w_observer minmax \
  --quantization_order activation \
  --gptq

run_or_continue "mxfp_mrgptq_w4a4_hadamard_h${HAD_LARGE}_mse_activation" mxfp \
  --transform_class hadamard --hadamard_group_size "${HAD_LARGE}" \
  --w_observer mse \
  --quantization_order activation \
  --gptq

echo
echo "[DONE] All requested experiments processed."
echo "Outputs: ${OUT_ROOT}/${MODEL_ID}/"
echo "Logs   : ${LOG_DIR}/"
echo "Done list   : ${DONE_LIST}"
echo "Failed list : ${FAILED_LIST}"

