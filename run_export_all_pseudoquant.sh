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
LOG_DIR="${LOG_DIR:-logs_export_all_pseudoquant}"
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
OUT_ROOT="${OUT_ROOT:-/cephfs/shared/zlouyang/FP-Quant/oyzl_test/results_quant_all_pseudoquant}"

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

# Pseudoquant only (requested)
EXPORT_MODE="pseudoquant"

# Offload toggles (0/1)
CPU_OFFLOAD_MODULES="${CPU_OFFLOAD_MODULES:-0}"
CPU_OFFLOAD_ACTIVATIONS="${CPU_OFFLOAD_ACTIVATIONS:-0}"
OFFLOAD_ARGS=()
[[ "${CPU_OFFLOAD_MODULES}" == "1" ]] && OFFLOAD_ARGS+=(--cpu_offload_modules)
[[ "${CPU_OFFLOAD_ACTIVATIONS}" == "1" ]] && OFFLOAD_ARGS+=(--cpu_offload_activations)

# Hadamard group choices
HAD_LARGE="${HAD_LARGE:-128}"

# Continue on failure
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

# Optional limiter (0 means unlimited)
MAX_EXPERIMENTS="${MAX_EXPERIMENTS:-0}"
RUN_COUNT=0

###############################################################################
# 5) Quant/search matrix
#
# NOTE:
# - This script intentionally covers the full practical matrix for current project:
#   fixed transforms + transform search objectives/tail settings.
# - If you want fewer runs, override arrays via env or use MAX_EXPERIMENTS.
###############################################################################
FORMATS=(${FORMATS:-"nvfp mxfp"})
METHODS=(${METHODS:-"rtn gptq"})
W_OBSERVERS=(${W_OBSERVERS:-"minmax mse"})
GPTQ_ORDERS=(${GPTQ_ORDERS:-"default activation"})

# Fixed transform matrix
FIXED_TRANSFORMS=(${FIXED_TRANSFORMS:-"identity hadamard dct dst gsr householder"})
# Search candidate list (default matches project default list)
SEARCH_CANDIDATES=(${SEARCH_CANDIDATES:-"identity hadamard dct dst gsr householder"})

# Search objective matrix
SEARCH_OBJECTIVES=(${SEARCH_OBJECTIVES:-"auto mse cov jtail"})
SEARCH_BASE_LOSSES=(${SEARCH_BASE_LOSSES:-"mse cov"})
SEARCH_TAIL_WEIGHT_MODES=(${SEARCH_TAIL_WEIGHT_MODES:-"a_low b_high mixed_uniform mixed_middle two_tail auto_abm"})
SEARCH_TAIL_LAMBDAS=(${SEARCH_TAIL_LAMBDAS:-"0.25"})
SEARCH_TAIL_BINS=(${SEARCH_TAIL_BINS:-"4"})
SEARCH_TAIL_POWERS=(${SEARCH_TAIL_POWERS:-"2.0"})

# Optional: include fast_food in fixed/search.
INCLUDE_FAST_FOOD="${INCLUDE_FAST_FOOD:-0}"
if [[ "${INCLUDE_FAST_FOOD}" == "1" ]]; then
  FIXED_TRANSFORMS+=("fast_food")
  SEARCH_CANDIDATES+=("fast_food")
fi

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

  # limiter
  if [[ "${MAX_EXPERIMENTS}" != "0" && "${RUN_COUNT}" -ge "${MAX_EXPERIMENTS}" ]]; then
    echo "[INFO] MAX_EXPERIMENTS=${MAX_EXPERIMENTS} reached, stop scheduling."
    return 2
  fi

  local outdir="${OUT_ROOT}/${MODEL_ID}/${name}"
  local logfile="${LOG_DIR}/${MODEL_ID}__${name}.log"
  local gsize
  if [[ "${fmt}" == "nvfp" ]]; then
    gsize=16
  else
    gsize=32
  fi

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

  {
    echo "### START $(date) ###"
    echo "NAME=${name}"
    echo "FORMAT=${fmt}"
    echo "MODEL_DIR=${MODEL_DIR}"
    echo "CALIB=${CALIB_DATASET}"
    echo "SEQ_LEN=${SEQ_LEN}  N_SEQS=${N_SEQS}  SEED=${SEED}  DTYPE=${DTYPE}"
    echo "EXPORT_MODE=${EXPORT_MODE}"
    echo "GROUP_SIZE=${gsize}"
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
    echo "[FAIL] ${name} (rc=${rc}). See log: ${logfile}"
    echo "${name} rc=${rc} log=${logfile}" >> "${FAILED_LIST}"
    return 1
  fi

  if is_done "${outdir}"; then
    echo "[OK] ${name} exported to: ${outdir}"
    echo "${name}" >> "${DONE_LIST}"
    RUN_COUNT=$((RUN_COUNT + 1))
    return 0
  else
    echo "[FAIL] ${name} finished but output incomplete. See log: ${logfile}"
    echo "${name} rc=0 but incomplete log=${logfile}" >> "${FAILED_LIST}"
    return 1
  fi
}

run_or_continue () {
  if ! run_one "$@"; then
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
  return 0
}

###############################################################################
# 9) Export matrix
###############################################################################
echo "[INFO] MODEL_DIR=${MODEL_DIR}"
echo "[INFO] CALIB_DATASET=${CALIB_DATASET}"
echo "[INFO] OUT_ROOT=${OUT_ROOT}/${MODEL_ID}"
echo "[INFO] LOG_DIR=${LOG_DIR}"
echo "[INFO] EXPORT_MODE=${EXPORT_MODE}"
echo

STOP_ALL=0

for fmt in "${FORMATS[@]}"; do
  if [[ "${fmt}" != "nvfp" && "${fmt}" != "mxfp" ]]; then
    echo "[WARN] Skip unsupported format: ${fmt}"
    continue
  fi
  if [[ "${fmt}" == "nvfp" ]]; then
    HAD_NATIVE=16
  else
    HAD_NATIVE=32
  fi

  # ---------------- FIXED TRANSFORMS (no search) ----------------
  for method in "${METHODS[@]}"; do
    for obs in "${W_OBSERVERS[@]}"; do
      for tf in "${FIXED_TRANSFORMS[@]}"; do
        HGROUP_SET=("${HAD_NATIVE}" "${HAD_LARGE}")
        if [[ "${tf}" == "identity" ]]; then
          HGROUP_SET=("${HAD_NATIVE}")
        fi

        for hgs in "${HGROUP_SET[@]}"; do
          if [[ "${method}" == "rtn" ]]; then
            name="${fmt}_rtn_fix_${tf}_h${hgs}_${obs}"
            run_or_continue "${name}" "${fmt}" \
              --transform_class "${tf}" \
              --hadamard_group_size "${hgs}" \
              --w_observer "${obs}" \
              --quantization_order default || STOP_ALL=$?
          elif [[ "${method}" == "gptq" ]]; then
            for order in "${GPTQ_ORDERS[@]}"; do
              name="${fmt}_gptq_fix_${tf}_h${hgs}_${obs}_${order}"
              run_or_continue "${name}" "${fmt}" \
                --transform_class "${tf}" \
                --hadamard_group_size "${hgs}" \
                --w_observer "${obs}" \
                --quantization_order "${order}" \
                --gptq || STOP_ALL=$?
              [[ "${STOP_ALL}" -eq 2 ]] && break 4
            done
          else
            echo "[WARN] Skip unsupported method: ${method}"
          fi
          [[ "${STOP_ALL}" -eq 2 ]] && break 4
        done
        [[ "${STOP_ALL}" -eq 2 ]] && break 4
      done
      [[ "${STOP_ALL}" -eq 2 ]] && break 4
    done
    [[ "${STOP_ALL}" -eq 2 ]] && break 3
  done
  [[ "${STOP_ALL}" -eq 2 ]] && break

  # ---------------- SEARCH TRANSFORMS ----------------
  for method in "${METHODS[@]}"; do
    for obs in "${W_OBSERVERS[@]}"; do
      if [[ "${method}" == "rtn" ]]; then
        ORDERS=("default")
      else
        ORDERS=("${GPTQ_ORDERS[@]}")
      fi

      for order in "${ORDERS[@]}"; do
        for objective in "${SEARCH_OBJECTIVES[@]}"; do
          if [[ "${objective}" != "jtail" ]]; then
            name="${fmt}_${method}_search_${objective}_${obs}_${order}"
            extra=()
            [[ "${method}" == "gptq" ]] && extra+=(--gptq)
            run_or_continue "${name}" "${fmt}" \
              --transform_class identity \
              --hadamard_group_size "${HAD_LARGE}" \
              --w_observer "${obs}" \
              --quantization_order "${order}" \
              --transform_search \
              --transform_search_candidates "${SEARCH_CANDIDATES[@]}" \
              --transform_search_objective "${objective}" \
              "${extra[@]}" || STOP_ALL=$?
            [[ "${STOP_ALL}" -eq 2 ]] && break 5
          else
            for base in "${SEARCH_BASE_LOSSES[@]}"; do
              for mode in "${SEARCH_TAIL_WEIGHT_MODES[@]}"; do
                for lam in "${SEARCH_TAIL_LAMBDAS[@]}"; do
                  for bins in "${SEARCH_TAIL_BINS[@]}"; do
                    for power in "${SEARCH_TAIL_POWERS[@]}"; do
                      name="${fmt}_${method}_search_jtail_b${base}_m${mode}_l${lam}_k${bins}_p${power}_${obs}_${order}"
                      extra=()
                      [[ "${method}" == "gptq" ]] && extra+=(--gptq)
                      run_or_continue "${name}" "${fmt}" \
                        --transform_class identity \
                        --hadamard_group_size "${HAD_LARGE}" \
                        --w_observer "${obs}" \
                        --quantization_order "${order}" \
                        --transform_search \
                        --transform_search_candidates "${SEARCH_CANDIDATES[@]}" \
                        --transform_search_objective jtail \
                        --transform_search_base_loss "${base}" \
                        --transform_search_tail_lambda "${lam}" \
                        --transform_search_tail_bins "${bins}" \
                        --transform_search_tail_weight_mode "${mode}" \
                        --transform_search_tail_weight_power "${power}" \
                        "${extra[@]}" || STOP_ALL=$?
                      [[ "${STOP_ALL}" -eq 2 ]] && break 10
                    done
                  done
                done
              done
            done
          fi
          [[ "${STOP_ALL}" -eq 2 ]] && break 5
        done
        [[ "${STOP_ALL}" -eq 2 ]] && break 4
      done
      [[ "${STOP_ALL}" -eq 2 ]] && break 3
    done
    [[ "${STOP_ALL}" -eq 2 ]] && break 2
  done
  [[ "${STOP_ALL}" -eq 2 ]] && break
done

echo
echo "[DONE] Export scheduling finished."
echo "Completed exports : ${RUN_COUNT}"
echo "Outputs           : ${OUT_ROOT}/${MODEL_ID}/"
echo "Logs              : ${LOG_DIR}/"
echo "Done list         : ${DONE_LIST}"
echo "Failed list       : ${FAILED_LIST}"