#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# 0) Optional env
###############################################################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/env_var.sh" ]]; then
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/env_var.sh"
fi

###############################################################################
# 1) Shard and auto detach
###############################################################################
SHARD_COUNT="${SHARD_COUNT:-1}"
SHARD_INDEX="${SHARD_INDEX:-1}"
if ! [[ "${SHARD_COUNT}" =~ ^[0-9]+$ && "${SHARD_INDEX}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] SHARD_COUNT and SHARD_INDEX must be positive integers."
  exit 1
fi
if (( SHARD_COUNT < 1 || SHARD_INDEX < 1 || SHARD_INDEX > SHARD_COUNT )); then
  echo "[ERROR] Invalid shard settings: SHARD_INDEX=${SHARD_INDEX}, SHARD_COUNT=${SHARD_COUNT}"
  exit 1
fi

AUTO_DETACH="${AUTO_DETACH:-1}"
DEFAULT_LOG_DIR="logs_export_transform_search_84"
if (( SHARD_COUNT > 1 )); then
  DEFAULT_LOG_DIR="logs_export_transform_search_84_part${SHARD_INDEX}_of_${SHARD_COUNT}"
fi
LOG_DIR="${LOG_DIR:-${DEFAULT_LOG_DIR}}"
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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"

###############################################################################
# 3) User paths
###############################################################################
MODEL_QUANT_PY="${MODEL_QUANT_PY:-/cephfs/shared/zlouyang/FP-Quant/model_quant.py}"
MODEL_DIR="${MODEL_DIR:-/cephfs/shared/model/llama-3-8b-instruct}"
OUT_ROOT="${OUT_ROOT:-/cephfs/shared/zlouyang/FP-Quant/oyzl_test/results_quant_all_pseudoquant_transform_search_84}"

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
EXPORT_MODE="pseudoquant"

CPU_OFFLOAD_MODULES="${CPU_OFFLOAD_MODULES:-0}"
CPU_OFFLOAD_ACTIVATIONS="${CPU_OFFLOAD_ACTIVATIONS:-1}"
OFFLOAD_ARGS=()
[[ "${CPU_OFFLOAD_MODULES}" == "1" ]] && OFFLOAD_ARGS+=(--cpu_offload_modules)
[[ "${CPU_OFFLOAD_ACTIVATIONS}" == "1" ]] && OFFLOAD_ARGS+=(--cpu_offload_activations)

CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
MAX_EXPERIMENTS="${MAX_EXPERIMENTS:-0}"
LIST_ONLY="${LIST_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"
RUN_COUNT=0

###############################################################################
# 5) Experiment matrix: 84 experiments by default
###############################################################################
if [[ -z "${FORMATS+x}" ]]; then
  FORMATS=(nvfp mxfp)
fi
if [[ -z "${METHODS+x}" ]]; then
  METHODS=(rtn gptq)
fi
if [[ -z "${W_OBSERVERS+x}" ]]; then
  W_OBSERVERS=(minmax mse)
fi
if [[ -z "${GPTQ_ORDERS+x}" ]]; then
  GPTQ_ORDERS=(default activation)
fi

if [[ -z "${SEARCH_CANDIDATES+x}" ]]; then
  SEARCH_CANDIDATES=(identity hadamard dct dst gsr householder)
fi
if [[ -z "${MAIN_OBJECTIVES+x}" ]]; then
  MAIN_OBJECTIVES=(mse cov act_mse)
fi

# jtail is intentionally restricted to auto_abm to avoid redundant tail-mode sweeps.
JTAIL_WEIGHT_MODE="${JTAIL_WEIGHT_MODE:-auto_abm}"
TAIL_LAMBDA="${TAIL_LAMBDA:-0.2}"
TAIL_BINS="${TAIL_BINS:-4}"
TAIL_POWER="${TAIL_POWER:-2.0}"
ACT_SAMPLE_SIZE="${ACT_SAMPLE_SIZE:-1024}"

INCLUDE_FAST_FOOD="${INCLUDE_FAST_FOOD:-0}"
if [[ "${INCLUDE_FAST_FOOD}" == "1" ]]; then
  SEARCH_CANDIDATES+=("fast_food")
fi

EXP_NAMES=()
EXP_FMTS=()
EXP_METHODS=()
EXP_OBS=()
EXP_ORDERS=()
EXP_OBJECTIVES=()
EXP_BASE_LOSSES=()
EXP_TAIL_SOURCES=()
declare -A SEEN_EXPS=()

sanitize_name_part () {
  local value="$1"
  value="${value//[^a-zA-Z0-9_]/_}"
  echo "${value}"
}

add_exp () {
  local name="$1"
  local fmt="$2"
  local method="$3"
  local obs="$4"
  local order="$5"
  local objective="$6"
  local base_loss="$7"
  local tail_source="$8"
  local key="${fmt}|${method}|${obs}|${order}|${objective}|${base_loss}|${tail_source}|${JTAIL_WEIGHT_MODE}"

  if [[ -n "${SEEN_EXPS[$key]:-}" ]]; then
    return 0
  fi
  SEEN_EXPS["${key}"]=1

  EXP_NAMES+=("${name}")
  EXP_FMTS+=("${fmt}")
  EXP_METHODS+=("${method}")
  EXP_OBS+=("${obs}")
  EXP_ORDERS+=("${order}")
  EXP_OBJECTIVES+=("${objective}")
  EXP_BASE_LOSSES+=("${base_loss}")
  EXP_TAIL_SOURCES+=("${tail_source}")
}

build_experiment_specs () {
  local fmt method obs order objective base_loss tail_source
  local local_orders=()
  local jtail_bases=(mse cov cov act_mse)
  local jtail_tails=(weight weight activation activation)

  for fmt in "${FORMATS[@]}"; do
    if [[ "${fmt}" != "nvfp" && "${fmt}" != "mxfp" ]]; then
      echo "[WARN] Skip unsupported format in spec build: ${fmt}"
      continue
    fi

    for method in "${METHODS[@]}"; do
      if [[ "${method}" == "rtn" ]]; then
        local_orders=(default)
      elif [[ "${method}" == "gptq" ]]; then
        local_orders=("${GPTQ_ORDERS[@]}")
      else
        echo "[WARN] Skip unsupported method in spec build: ${method}"
        continue
      fi

      for obs in "${W_OBSERVERS[@]}"; do
        for order in "${local_orders[@]}"; do
          for objective in "${MAIN_OBJECTIVES[@]}"; do
            add_exp \
              "${fmt}_${method}_search_${objective}_${obs}_${order}" \
              "${fmt}" "${method}" "${obs}" "${order}" "${objective}" "" ""
          done

          for i in "${!jtail_bases[@]}"; do
            base_loss="${jtail_bases[$i]}"
            tail_source="${jtail_tails[$i]}"
            add_exp \
              "${fmt}_${method}_jtail_base_${base_loss}_tail_${tail_source}_${JTAIL_WEIGHT_MODE}_${obs}_${order}" \
              "${fmt}" "${method}" "${obs}" "${order}" "jtail" "${base_loss}" "${tail_source}"
          done
        done
      done
    done
  done
}

build_experiment_specs
TOTAL_EXPERIMENTS="${#EXP_NAMES[@]}"

if (( TOTAL_EXPERIMENTS == 0 )); then
  echo "[ERROR] No experiments were generated."
  exit 1
fi

if (( SHARD_COUNT == 4 && TOTAL_EXPERIMENTS != 84 )); then
  echo "[WARN] Expected 84 default experiments, got ${TOTAL_EXPERIMENTS}."
  echo "       This is OK only if you intentionally overrode the matrix in env_var.sh."
fi

PER_SHARD=$(( (TOTAL_EXPERIMENTS + SHARD_COUNT - 1) / SHARD_COUNT ))
START_INDEX=$(( (SHARD_INDEX - 1) * PER_SHARD ))
END_INDEX=$(( START_INDEX + PER_SHARD ))
(( END_INDEX > TOTAL_EXPERIMENTS )) && END_INDEX="${TOTAL_EXPERIMENTS}"

###############################################################################
# 6) Pre-checks
###############################################################################
if [[ "${LIST_ONLY}" != "1" && "${DRY_RUN}" != "1" ]]; then
  python -c "import lm_eval" >/dev/null 2>&1 || {
    echo "[ERROR] Cannot import lm_eval in current env."
    echo "        Install: python -m pip install lm-eval"
    exit 1
  }
fi

###############################################################################
# 7) Resume helper
###############################################################################
is_done () {
  local outdir="$1"
  [[ -f "${outdir}/config.json" ]] || return 1
  if [[ -f "${outdir}/model.safetensors" ]]; then
    return 0
  fi
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

needs_activation_samples () {
  local objective="$1"
  local base_loss="$2"
  local tail_source="$3"
  [[ "${objective}" == "act_mse" ]] && return 0
  [[ "${objective}" == "jtail" && "${base_loss}" == "act_mse" ]] && return 0
  [[ "${objective}" == "jtail" && "${tail_source}" == "activation" ]] && return 0
  return 1
}

run_one () {
  local exp_number="$1"
  local name="$2"
  local fmt="$3"
  local method="$4"
  local obs="$5"
  local order="$6"
  local objective="$7"
  local base_loss="$8"
  local tail_source="$9"

  if [[ "${MAX_EXPERIMENTS}" != "0" && "${RUN_COUNT}" -ge "${MAX_EXPERIMENTS}" ]]; then
    echo "[INFO] MAX_EXPERIMENTS=${MAX_EXPERIMENTS} reached, stop scheduling."
    return 2
  fi

  local gsize
  if [[ "${fmt}" == "nvfp" ]]; then
    gsize=16
  else
    gsize=32
  fi

  local outdir="${OUT_ROOT}/${MODEL_ID}/${name}"
  local logfile="${LOG_DIR}/${MODEL_ID}__${name}.log"
  local cmd_args=()

  cmd_args+=(
    --model_name_or_path "${MODEL_DIR}"
    --dataset_name_or_path "${CALIB_DATASET}"
    --sequence_length "${SEQ_LEN}"
    --num_sequences "${N_SEQS}"
    --seed "${SEED}"
    --dtype "${DTYPE}"
    --format "${fmt}"
    --w_bits 4
    --a_bits 4
    --w_granularity group
    --a_granularity group
    --w_group_size "${gsize}"
    --a_group_size "${gsize}"
    --w_observer "${obs}"
    --quantization_order "${order}"
    --export_quantized_model "${EXPORT_MODE}"
    --save_path "${outdir}"
    --max_shard_size "${MAX_SHARD_SIZE}"
    --fuse_global_scale
    --amp
    --transform_class identity
    --hadamard_group_size "${gsize}"
    --transform_search
    --transform_search_candidates "${SEARCH_CANDIDATES[@]}"
    --transform_search_objective "${objective}"
  )

  if [[ "${method}" == "gptq" ]]; then
    cmd_args+=(--gptq)
  fi

  if [[ "${objective}" == "jtail" ]]; then
    cmd_args+=(
      --transform_search_base_loss "${base_loss}"
      --transform_search_tail_source "${tail_source}"
      --transform_search_tail_lambda "${TAIL_LAMBDA}"
      --transform_search_tail_bins "${TAIL_BINS}"
      --transform_search_tail_weight_mode "${JTAIL_WEIGHT_MODE}"
      --transform_search_tail_weight_power "${TAIL_POWER}"
    )
  fi

  if needs_activation_samples "${objective}" "${base_loss}" "${tail_source}"; then
    cmd_args+=(--transform_search_act_sample_size "${ACT_SAMPLE_SIZE}")
  fi

  if is_done "${outdir}"; then
    echo "[SKIP] #${exp_number}/${TOTAL_EXPERIMENTS} ${name} already exported: ${outdir}"
    echo "${name}" >> "${DONE_LIST}"
    return 0
  fi

  mkdir -p "${outdir}"
  echo "=== EXPORT #${exp_number}/${TOTAL_EXPERIMENTS}: ${name} ==="
  echo "  shard : ${SHARD_INDEX}/${SHARD_COUNT}"
  echo "  model : ${MODEL_DIR}"
  echo "  calib : ${CALIB_DATASET}"
  echo "  out   : ${outdir}"
  echo "  log   : ${logfile}"
  echo

  {
    echo "### START $(date) ###"
    echo "EXPERIMENT_INDEX=${exp_number}/${TOTAL_EXPERIMENTS}"
    echo "SHARD=${SHARD_INDEX}/${SHARD_COUNT}"
    echo "NAME=${name}"
    echo "FORMAT=${fmt}"
    echo "METHOD=${method}"
    echo "OBSERVER=${obs}"
    echo "ORDER=${order}"
    echo "OBJECTIVE=${objective}"
    echo "BASE_LOSS=${base_loss:-<none>}"
    echo "TAIL_SOURCE=${tail_source:-<none>}"
    echo "TAIL_MODE=${JTAIL_WEIGHT_MODE}"
    echo "MODEL_DIR=${MODEL_DIR}"
    echo "CALIB=${CALIB_DATASET}"
    echo "SEQ_LEN=${SEQ_LEN}  N_SEQS=${N_SEQS}  SEED=${SEED}  DTYPE=${DTYPE}"
    echo "EXPORT_MODE=${EXPORT_MODE}"
    echo "GROUP_SIZE=${gsize}"
    echo "SEARCH_CANDIDATES=${SEARCH_CANDIDATES[*]}"
    echo "ACT_SAMPLE_SIZE=${ACT_SAMPLE_SIZE}"
    echo "OFFLOAD_ARGS=${OFFLOAD_ARGS[*]:-<none>}"
    printf 'CMD=python %q ' "${MODEL_QUANT_PY}"
    printf '%q ' "${cmd_args[@]}"
    echo
    echo
  } >> "${logfile}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY_RUN] ${name}"
    return 0
  fi

  set +e
  stdbuf -oL -eL python "${MODEL_QUANT_PY}" \
    "${cmd_args[@]}" \
    "${OFFLOAD_ARGS[@]}" >> "${logfile}" 2>&1
  local rc=$?
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
# 9) Schedule
###############################################################################
echo "[INFO] MODEL_DIR=${MODEL_DIR}"
echo "[INFO] CALIB_DATASET=${CALIB_DATASET}"
echo "[INFO] OUT_ROOT=${OUT_ROOT}/${MODEL_ID}"
echo "[INFO] LOG_DIR=${LOG_DIR}"
echo "[INFO] EXPORT_MODE=${EXPORT_MODE}"
echo "[INFO] SHARD=${SHARD_INDEX}/${SHARD_COUNT}"
echo "[INFO] TOTAL_EXPERIMENTS=${TOTAL_EXPERIMENTS}"
echo "[INFO] SELECTED_RANGE=$((START_INDEX + 1))..${END_INDEX}"
echo "[INFO] SELECTED_COUNT=$((END_INDEX - START_INDEX))"
echo "[INFO] SEARCH_CANDIDATES=${SEARCH_CANDIDATES[*]}"
echo "[INFO] MAIN_OBJECTIVES=${MAIN_OBJECTIVES[*]}"
echo "[INFO] JTAIL_WEIGHT_MODE=${JTAIL_WEIGHT_MODE}"
echo "[INFO] TAIL_LAMBDA=${TAIL_LAMBDA} TAIL_BINS=${TAIL_BINS} TAIL_POWER=${TAIL_POWER}"
echo "[INFO] ACT_SAMPLE_SIZE=${ACT_SAMPLE_SIZE}"
echo

if [[ "${LIST_ONLY}" == "1" ]]; then
  for ((idx = START_INDEX; idx < END_INDEX; idx++)); do
    exp_number=$((idx + 1))
    printf '%03d %s\n' "${exp_number}" "${EXP_NAMES[$idx]}"
  done
  exit 0
fi

STOP_ALL=0
for ((idx = START_INDEX; idx < END_INDEX; idx++)); do
  exp_number=$((idx + 1))
  run_or_continue \
    "${exp_number}" \
    "${EXP_NAMES[$idx]}" \
    "${EXP_FMTS[$idx]}" \
    "${EXP_METHODS[$idx]}" \
    "${EXP_OBS[$idx]}" \
    "${EXP_ORDERS[$idx]}" \
    "${EXP_OBJECTIVES[$idx]}" \
    "${EXP_BASE_LOSSES[$idx]}" \
    "${EXP_TAIL_SOURCES[$idx]}" || STOP_ALL=$?
  [[ "${STOP_ALL}" -eq 2 ]] && break
done

echo
echo "[DONE] Export scheduling finished."
echo "Shard             : ${SHARD_INDEX}/${SHARD_COUNT}"
echo "Selected range    : $((START_INDEX + 1))..${END_INDEX}"
echo "Completed exports : ${RUN_COUNT}"
echo "Outputs           : ${OUT_ROOT}/${MODEL_ID}/"
echo "Logs              : ${LOG_DIR}/"
echo "Done list         : ${DONE_LIST}"
echo "Failed list       : ${FAILED_LIST}"
