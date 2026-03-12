#!/usr/bin/env bash
set -euo pipefail
source env_var.sh

MODEL="${MODEL:-/cephfs/shared/model/llama-3-8b-instruct}"
DTYPE="${DTYPE:-auto}"
NUM_SEQUENCES=1024

# 优先用你已经生成的 jsonl 校准集；不行再回退 fineweb-edu
if [[ -f "$OUT_CALIB/fineweb_calib_1024x2048_text.jsonl" ]]; then
  CALIB="$OUT_CALIB/fineweb_calib_1024x2048_text.jsonl"
else
  CALIB="fineweb-edu"
fi

run_one () {
  local name="$1"
  shift
  echo "=== RUN: $name ==="
  python /cephfs/shared/zlouyang/FP-Quant/model_quant.py \
    --model_name_or_path="${MODEL}" \
    "$@" \
    --dataset_name_or_path="${CALIB}" \
    --num_sequences="${NUM_SEQUENCES}" \
    --sequence_length=2048 \
    --dtype="${DTYPE}" \
    --lm_eval_batch_size=auto \
    --save_path="outputs/${name}" \
    --cpu_offload_activations \
    --cpu_offload_modules \
    --fuse_global_scale \
    --amp
}

# ---------- NVFP4 ----------
# (1) RTN
run_one "nvfp_rtn_w4a4_g16" \
  --format=nvfp --w_bits=4 --a_bits=4 --w_group_size=16 --a_group_size=16 \
  --transform_class=identity --w_observer=minmax --quantization_order=default

# (2) RTN + Hadamard
run_one "nvfp_rtn_ht_w4a4_g16" \
  --format=nvfp --w_bits=4 --a_bits=4 --w_group_size=16 --a_group_size=16 \
  --transform_class=hadamard --hadamard_group_size=16 \
  --w_observer=minmax --quantization_order=default

# (3) GPTQ
run_one "nvfp_gptq_w4a4_g16" \
  --format=nvfp --w_bits=4 --a_bits=4 --w_group_size=16 --a_group_size=16 \
  --transform_class=identity --w_observer=minmax --quantization_order=default \
  --gptq

# (4) MR-GPTQ
run_one "nvfp_mrgptq_w4a4_g16" \
  --format=nvfp --w_bits=4 --a_bits=4 --w_group_size=16 --a_group_size=16 \
  --transform_class=hadamard --hadamard_group_size=128 \
  --w_observer=mse --quantization_order=actorder \
  --gptq

# ---------- MXFP4 ----------
run_one "mxfp_rtn_w4a4_g32" \
  --format=mxfp --w_bits=4 --a_bits=4 --w_group_size=32 --a_group_size=32 \
  --transform_class=identity --w_observer=minmax --quantization_order=default

run_one "mxfp_rtn_ht_w4a4_g32" \
  --format=mxfp --w_bits=4 --a_bits=4 --w_group_size=32 --a_group_size=32 \
  --transform_class=hadamard --hadamard_group_size=32 \
  --w_observer=minmax --quantization_order=default

run_one "mxfp_gptq_w4a4_g32" \
  --format=mxfp --w_bits=4 --a_bits=4 --w_group_size=32 --a_group_size=32 \
  --transform_class=identity --w_observer=minmax --quantization_order=default \
  --gptq

run_one "mxfp_mrgptq_w4a4_g32" \
  --format=mxfp --w_bits=4 --a_bits=4 --w_group_size=32 --a_group_size=32 \
  --transform_class=hadamard --hadamard_group_size=128 \
  --w_observer=mse --quantization_order=actorder \
  --gptq

