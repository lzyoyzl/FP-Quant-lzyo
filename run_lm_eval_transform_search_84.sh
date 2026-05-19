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
DEFAULT_LOG_DIR="lm_eval_logs_llama3_transform_search_84"
if (( SHARD_COUNT > 1 )); then
  DEFAULT_LOG_DIR="lm_eval_logs_llama3_transform_search_84_part${SHARD_INDEX}_of_${SHARD_COUNT}"
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
# 2) Paths and common lm_eval settings
###############################################################################
# export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
# export http_proxy="${http_proxy:-http://127.0.0.1:7990}"
# export https_proxy="${https_proxy:-http://127.0.0.1:7990}"

BASE_MODEL_DIR="${BASE_MODEL_DIR:-/cephfs/shared/model/llama-3-8b-instruct}"
ROTSEARCH_ROOT="${ROTSEARCH_ROOT:-/cephfs/shared/zlouyang/FP-Quant/oyzl_test/results_quant_all_pseudoquant_transform_search_84/llama-3-8b-instruct}"

OUT_DIR="${OUT_DIR:-lm_eval_results_llama3_transform_search_84}"
mkdir -p "${OUT_DIR}"

DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-True}"
BATCH_SIZE="${BATCH_SIZE:-16}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
LIST_ONLY="${LIST_ONLY:-0}"

# By default, only shard 1 evaluates BASE to avoid repeating BASE on all machines.
if [[ -z "${RUN_BASE+x}" ]]; then
  if (( SHARD_INDEX == 1 )); then
    RUN_BASE=1
  else
    RUN_BASE=0
  fi
fi

if [[ -z "${TASKS+x}" ]]; then
  TASKS=(boolq arc_easy arc_challenge piqa winogrande hellaswag)
fi

###############################################################################
# 2.1) Fewshot settings
###############################################################################
FS_BOOLQ=0
FS_ARC_EASY=0
FS_ARC_CHALLENGE=25
FS_PIQA=0
FS_WINOGRANDE=5
FS_HELLASWAG=0

###############################################################################
# 3) Build the same 84 model names as run_export_transform_search_84.sh
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
if [[ -z "${MAIN_OBJECTIVES+x}" ]]; then
  MAIN_OBJECTIVES=(mse cov act_mse)
fi

JTAIL_WEIGHT_MODE="${JTAIL_WEIGHT_MODE:-auto_abm}"

EXP_NAMES=()
declare -A SEEN_EXPS=()

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
  echo "[ERROR] No model names were generated."
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
# 4) Helpers
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

make_model_args () {
  local pretrained="$1"
  echo "pretrained=${pretrained},device=${DEVICE},dtype=${DTYPE},trust_remote_code=${TRUST_REMOTE_CODE}"
}

fewshot_for_task () {
  local task="$1"
  case "${task}" in
    boolq) echo "${FS_BOOLQ}" ;;
    arc_easy) echo "${FS_ARC_EASY}" ;;
    arc_challenge) echo "${FS_ARC_CHALLENGE}" ;;
    piqa) echo "${FS_PIQA}" ;;
    winogrande) echo "${FS_WINOGRANDE}" ;;
    hellaswag) echo "${FS_HELLASWAG}" ;;
    *)
      echo "[ERROR] Unknown task: ${task}" >&2
      return 1
      ;;
  esac
}

build_cmd () {
  local model_args="$1"
  local task="$2"
  local out="$3"
  local fs
  fs="$(fewshot_for_task "${task}")"
  echo "lm_eval --model hf --model_args \"${model_args}\" --tasks ${task} --num_fewshot ${fs} --batch_size ${BATCH_SIZE} --output_path \"${out}\""
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
  local rc=$?
  set -e

  if [[ $rc -ne 0 ]]; then
    echo "[FAIL] ${tag} :: ${task} (rc=${rc}). See: ${logfile}"
    return $rc
  fi

  echo "[OK] ${tag} :: ${task}"
  return 0
}

run_or_continue () {
  local rc
  if run_cmd "$@"; then
    return 0
  else
    rc=$?
    if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
      echo "[WARN] Continue on error enabled; moving to next run."
      return 0
    else
      echo "[ERROR] Stopping due to failure. Set CONTINUE_ON_ERROR=1 to continue."
      exit "${rc}"
    fi
  fi
}

###############################################################################
# 5) Schedule
###############################################################################
echo "[INFO] BASE_MODEL_DIR=${BASE_MODEL_DIR}"
echo "[INFO] ROTSEARCH_ROOT=${ROTSEARCH_ROOT}"
echo "[INFO] OUT_DIR=${OUT_DIR}"
echo "[INFO] LOG_DIR=${LOG_DIR}"
echo "[INFO] SHARD=${SHARD_INDEX}/${SHARD_COUNT}"
echo "[INFO] TOTAL_EXPERIMENTS=${TOTAL_EXPERIMENTS}"
echo "[INFO] SELECTED_RANGE=$((START_INDEX + 1))..${END_INDEX}"
echo "[INFO] SELECTED_COUNT=$((END_INDEX - START_INDEX))"
echo "[INFO] RUN_BASE=${RUN_BASE}"
echo "[INFO] BATCH_SIZE=${BATCH_SIZE}"
echo "[INFO] TASKS=${TASKS[*]}"
echo

if [[ "${LIST_ONLY}" == "1" ]]; then
  for ((idx = START_INDEX; idx < END_INDEX; idx++)); do
    printf '%03d %s\n' "$((idx + 1))" "${EXP_NAMES[$idx]}"
  done
  exit 0
fi

[[ -d "${BASE_MODEL_DIR}" ]] || { echo "[ERROR] BASE_MODEL_DIR not found: ${BASE_MODEL_DIR}"; exit 1; }
[[ -d "${ROTSEARCH_ROOT}" ]] || { echo "[ERROR] ROTSEARCH_ROOT not found: ${ROTSEARCH_ROOT}"; exit 1; }

for task in "${TASKS[@]}"; do
  echo "############################"
  echo "### TASK: ${task}"
  echo "############################"

  if [[ "${RUN_BASE}" == "1" ]]; then
    base_args="$(make_model_args "${BASE_MODEL_DIR}")"
    out="${OUT_DIR}/BASE/${task}_full"
    log="${LOG_DIR}/BASE__${task}.log"
    cmd="$(build_cmd "${base_args}" "${task}" "${out}")"
    run_or_continue "BASE" "${task}" "${cmd}" "${out}" "${log}"
  fi

  for ((idx = START_INDEX; idx < END_INDEX; idx++)); do
    exp_number=$((idx + 1))
    name="${EXP_NAMES[$idx]}"
    model_path="${ROTSEARCH_ROOT}/${name}"
    if [[ ! -d "${model_path}" ]]; then
      echo "[WARN] Missing model dir, skip #${exp_number}: ${model_path}"
      continue
    fi

    model_args="$(make_model_args "${model_path}")"
    out="${OUT_DIR}/ROTSEARCH/${name}/${task}_full"
    log="${LOG_DIR}/ROTSEARCH__${name}__${task}.log"
    cmd="$(build_cmd "${model_args}" "${task}" "${out}")"
    run_or_continue "ROTSEARCH#${exp_number}__${name}" "${task}" "${cmd}" "${out}" "${log}"
  done

  echo
done

echo "[DONE] transform-search lm_eval script completed."
echo "Shard          : ${SHARD_INDEX}/${SHARD_COUNT}"
echo "Selected range : $((START_INDEX + 1))..${END_INDEX}"
echo "Results        : ${OUT_DIR}/"
echo "Logs           : ${LOG_DIR}/"
