#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# 0) Optional env
###############################################################################
if [[ -f "env_var.sh" ]]; then
  # shellcheck disable=SC1091
  source "env_var.sh"
fi

###############################################################################
# 1) Auto detach (SSH-safe)
###############################################################################
AUTO_DETACH="${AUTO_DETACH:-1}"
LOG_DIR="${LOG_DIR:-logs_rerun_mse_fix}"
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
# 2) Runtime settings
###############################################################################
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

###############################################################################
# 3) User paths
###############################################################################
MODEL_QUANT_PY="${MODEL_QUANT_PY:-/cephfs/shared/zlouyang/FP-Quant/model_quant.py}"
MODEL_DIR="${MODEL_DIR:-/cephfs/shared/model/llama-3-8b-instruct}"
OUT_ROOT="${OUT_ROOT:-/cephfs/shared/zlouyang/FP-Quant/oyzl_test/results_quant_rerun_mse_fix}"

CALIB_PT_DEFAULT=""
CALIB_JSONL_DEFAULT=""
if [[ -n "${OUT_CALIB:-}" ]]; then
  CALIB_PT_DEFAULT="${OUT_CALIB}/fineweb_calib_1024x2048_tokens.pt"
  CALIB_JSONL_DEFAULT="${OUT_CALIB}/fineweb_calib_1024x2048_text.jsonl"
fi

CALIB_PT="${CALIB_PT:-${CALIB_PT_DEFAULT:-}}"
CALIB_JSONL="${CALIB_JSONL:-${CALIB_JSONL_DEFAULT:-}}"
ALLOW_HF_FALLBACK="${ALLOW_HF_FALLBACK:-0}"
CALIB_HF="${CALIB_HF:-HuggingFaceFW/fineweb-edu}"

if [[ -n "${CALIB_PT}" && -f "${CALIB_PT}" ]]; then
  CALIB_DATASET="${CALIB_PT}"
elif [[ -n "${CALIB_JSONL}" && -f "${CALIB_JSONL}" ]]; then
  CALIB_DATASET="${CALIB_JSONL}"
elif [[ "${ALLOW_HF_FALLBACK}" == "1" ]]; then
  CALIB_DATASET="${CALIB_HF}"
else
  echo "[ERROR] Local calib file not found (.pt/.jsonl)."
  echo "        Set CALIB_PT/CALIB_JSONL, or ALLOW_HF_FALLBACK=1."
  exit 1
fi

[[ -f "${MODEL_QUANT_PY}" ]] || { echo "[ERROR] MODEL_QUANT_PY not found: ${MODEL_QUANT_PY}"; exit 1; }
[[ -d "${MODEL_DIR}" ]] || { echo "[ERROR] MODEL_DIR not found: ${MODEL_DIR}"; exit 1; }

###############################################################################
# 4) Common quant settings
###############################################################################
MODEL_ID="$(basename "${MODEL_DIR}")"
SEQ_LEN="${SEQ_LEN:-2048}"
N_SEQS="${N_SEQS:-1024}"
SEED="${SEED:-0}"
DTYPE="${DTYPE:-auto}"
MAX_SHARD_SIZE="${MAX_SHARD_SIZE:-5368709120}"

# Pseudoquant only
EXPORT_MODE="pseudoquant"

# Offload toggles (0/1)
CPU_OFFLOAD_MODULES="${CPU_OFFLOAD_MODULES:-1}"
CPU_OFFLOAD_ACTIVATIONS="${CPU_OFFLOAD_ACTIVATIONS:-0}"
OFFLOAD_ARGS=()
[[ "${CPU_OFFLOAD_MODULES}" == "1" ]] && OFFLOAD_ARGS+=(--cpu_offload_modules)
[[ "${CPU_OFFLOAD_ACTIVATIONS}" == "1" ]] && OFFLOAD_ARGS+=(--cpu_offload_activations)

# Continue on failure
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

# 0 means unlimited
MAX_EXPERIMENTS="${MAX_EXPERIMENTS:-0}"
RUN_COUNT=0

###############################################################################
# 5) Target matrix (this script is intentionally narrow)
###############################################################################
# Re-run all RTN + mse for these formats.
if [[ -z "${FORMATS+x}" ]]; then
  FORMATS=(nvfp mxfp)
fi

# RTN objective list (keep aligned with your current policy).
if [[ -z "${SEARCH_OBJECTIVES_RTN+x}" ]]; then
  SEARCH_OBJECTIVES_RTN=(auto cov)
fi

# Candidates list
if [[ -z "${SEARCH_CANDIDATES+x}" ]]; then
  SEARCH_CANDIDATES=(identity hadamard dct dst gsr householder)
fi

# GPTQ+mse sample probes (1-2 recommended). Default: exactly 2 probes.
# Format: "fmt:objective:order"
if [[ -z "${GPTQ_SAMPLE_TARGETS+x}" ]]; then
  GPTQ_SAMPLE_TARGETS=(
    "nvfp:auto:default"
    "mxfp:auto:default"
  )
fi

INCLUDE_FAST_FOOD="${INCLUDE_FAST_FOOD:-0}"
if [[ "${INCLUDE_FAST_FOOD}" == "1" ]]; then
  SEARCH_CANDIDATES+=("fast_food")
fi

# Avoid clobbering old outputs by default.
NAME_SUFFIX="${NAME_SUFFIX:-_rerun_msefix}"

###############################################################################
# 6) Pre-check: lm_eval import is required by model_quant.py top-level imports
###############################################################################
python -c "import lm_eval" >/dev/null 2>&1 || {
  echo "[ERROR] Cannot import lm_eval in current env."
  echo "        Install: python -m pip install lm-eval"
  exit 1
}

###############################################################################
# 7) Resume helper
###############################################################################
is_done () {
  local outdir="$1"
  [[ -f "${outdir}/config.json" ]] || return 1
  [[ -f "${outdir}/model.safetensors.index.json" ]] || return 1
  ls "${outdir}"/model-*-of-*.safetensors >/dev/null 2>&1 || return 1
  return 0
}

###############################################################################
# 8) Runner
###############################################################################
FAILED_LIST="${LOG_DIR}/failed.txt"
DONE_LIST="${LOG_DIR}/done.txt"
mkdir -p "${OUT_ROOT}/${MODEL_ID}"
: > "${FAILED_LIST}" || true
: > "${DONE_LIST}" || true

run_one () {
  local name="$1"; shift
  local fmt="$1"; shift

  if [[ "${MAX_EXPERIMENTS}" != "0" && "${RUN_COUNT}" -ge "${MAX_EXPERIMENTS}" ]]; then
    echo "[INFO] MAX_EXPERIMENTS=${MAX_EXPERIMENTS} reached, stop scheduling."
    return 2
  fi

  local outdir="${OUT_ROOT}/${MODEL_ID}/${name}${NAME_SUFFIX}"
  local logfile="${LOG_DIR}/${MODEL_ID}__${name}${NAME_SUFFIX}.log"
  local gsize
  local hgs_search
  if [[ "${fmt}" == "nvfp" ]]; then
    gsize=16
    hgs_search=16
  elif [[ "${fmt}" == "mxfp" ]]; then
    gsize=32
    hgs_search=32
  else
    echo "[WARN] Unsupported fmt=${fmt}, skip."
    return 0
  fi

  if is_done "${outdir}"; then
    echo "[SKIP] ${name}${NAME_SUFFIX} already exported: ${outdir}"
    echo "${name}${NAME_SUFFIX}" >> "${DONE_LIST}"
    return 0
  fi

  mkdir -p "${outdir}"
  echo "=== EXPORT: ${name}${NAME_SUFFIX} ==="
  echo "  model : ${MODEL_DIR}"
  echo "  calib : ${CALIB_DATASET}"
  echo "  out   : ${outdir}"
  echo "  log   : ${logfile}"
  echo

  {
    echo "### START $(date) ###"
    echo "NAME=${name}${NAME_SUFFIX}"
    echo "FORMAT=${fmt}"
    echo "MODEL_DIR=${MODEL_DIR}"
    echo "CALIB=${CALIB_DATASET}"
    echo "SEQ_LEN=${SEQ_LEN}  N_SEQS=${N_SEQS}  SEED=${SEED}  DTYPE=${DTYPE}"
    echo "EXPORT_MODE=${EXPORT_MODE}"
    echo "GROUP_SIZE=${gsize}"
    echo "HADAMARD_GROUP_SIZE=${hgs_search}"
    echo "SEARCH_CANDIDATES=${SEARCH_CANDIDATES[*]}"
    echo "OFFLOAD_ARGS=${OFFLOAD_ARGS[*]:-<none>}"
    echo
  } >> "${logfile}"

  set +e
  stdbuf -oL -eL python "${MODEL_QUANT_PY}" \
    --model_name_or_path "${MODEL_DIR}" \
    --dataset_name_or_path "${CALIB_DATASET}" \
    --sequence_length "${SEQ_LEN}" \
    --num_sequences "${N_SEQS}" \
    --seed "${SEED}" \
    --dtype "${DTYPE}" \
    --format "${fmt}" \
    --w_bits 4 \
    --a_bits 4 \
    --w_granularity group \
    --a_granularity group \
    --w_group_size "${gsize}" \
    --a_group_size "${gsize}" \
    --export_quantized_model "${EXPORT_MODE}" \
    --save_path "${outdir}" \
    --max_shard_size "${MAX_SHARD_SIZE}" \
    --fuse_global_scale \
    --amp \
    "${OFFLOAD_ARGS[@]}" \
    "$@" >> "${logfile}" 2>&1
  rc=$?
  set -e

  if [[ $rc -ne 0 ]]; then
    echo "[FAIL] ${name}${NAME_SUFFIX} (rc=${rc}). See log: ${logfile}"
    echo "${name}${NAME_SUFFIX} rc=${rc} log=${logfile}" >> "${FAILED_LIST}"
    return 1
  fi

  if is_done "${outdir}"; then
    echo "[OK] ${name}${NAME_SUFFIX} exported to: ${outdir}"
    echo "${name}${NAME_SUFFIX}" >> "${DONE_LIST}"
    RUN_COUNT=$((RUN_COUNT + 1))
    return 0
  else
    echo "[FAIL] ${name}${NAME_SUFFIX} finished but output incomplete. See log: ${logfile}"
    echo "${name}${NAME_SUFFIX} rc=0 but incomplete log=${logfile}" >> "${FAILED_LIST}"
    return 1
  fi
}

run_or_continue () {
  local rc
  if run_one "$@"; then
    return 0
  else
    rc=$?
    if [[ $rc -eq 2 ]]; then
      return 2
    fi
    if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
      echo "[WARN] Continue on error enabled; moving to next experiment."
      return 0
    else
      echo "[ERROR] Stopping due to failure. Set CONTINUE_ON_ERROR=1 to continue."
      exit 1
    fi
  fi
}

###############################################################################
# 9) Scheduling
###############################################################################
echo "[INFO] MODEL_DIR=${MODEL_DIR}"
echo "[INFO] CALIB_DATASET=${CALIB_DATASET}"
echo "[INFO] OUT_ROOT=${OUT_ROOT}/${MODEL_ID}"
echo "[INFO] LOG_DIR=${LOG_DIR}"
echo "[INFO] EXPORT_MODE=${EXPORT_MODE}"
echo "[INFO] SEARCH_CANDIDATES=${SEARCH_CANDIDATES[*]}"
echo "[INFO] SEARCH_OBJECTIVES_RTN=${SEARCH_OBJECTIVES_RTN[*]}"
echo "[INFO] GPTQ_SAMPLE_TARGETS=${GPTQ_SAMPLE_TARGETS[*]}"
echo "[INFO] NAME_SUFFIX=${NAME_SUFFIX}"
echo

STOP_ALL=0

# A) Re-run all RTN + w_observer=mse
for fmt in "${FORMATS[@]}"; do
  for objective in "${SEARCH_OBJECTIVES_RTN[@]}"; do
    name="${fmt}_rtn_search_${objective}_mse_default"
    run_or_continue "${name}" "${fmt}" \
      --transform_class identity \
      --w_observer mse \
      --quantization_order default \
      --transform_search \
      --transform_search_candidates "${SEARCH_CANDIDATES[@]}" \
      --transform_search_objective "${objective}" || STOP_ALL=$?
    [[ "${STOP_ALL}" -eq 2 ]] && break 2
  done
done

# B) GPTQ + mse spot-checks (1-2 recommended; default 2)
for target in "${GPTQ_SAMPLE_TARGETS[@]}"; do
  IFS=':' read -r fmt objective order <<< "${target}"
  if [[ -z "${fmt}" || -z "${objective}" || -z "${order}" ]]; then
    echo "[WARN] Invalid GPTQ sample target: ${target}"
    continue
  fi
  name="${fmt}_gptq_search_${objective}_mse_${order}_sample"
  run_or_continue "${name}" "${fmt}" \
    --gptq \
    --transform_class identity \
    --w_observer mse \
    --quantization_order "${order}" \
    --transform_search \
    --transform_search_candidates "${SEARCH_CANDIDATES[@]}" \
    --transform_search_objective "${objective}" || STOP_ALL=$?
  [[ "${STOP_ALL}" -eq 2 ]] && break
done

echo
echo "[DONE] Re-run scheduling finished."
echo "Completed exports : ${RUN_COUNT}"
echo "Outputs           : ${OUT_ROOT}/${MODEL_ID}/"
echo "Logs              : ${LOG_DIR}/"
echo "Done list         : ${DONE_LIST}"
echo "Failed list       : ${FAILED_LIST}"

