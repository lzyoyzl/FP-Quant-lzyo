python /cephfs/shared/zlouyang/FP-Quant/model_quant.py \
  --model_name_or_path=/cephfs/shared/model/llama-3-8b-instruct \
  --dataset_name_or_path="${OUT_CALIB}/fineweb_calib_1024x2048_tokens.pt" \
  --num_sequences=1024 \
  --sequence_length=2048 \
  --dtype=auto \
  --format=mxfp \
  --w_bits=4 --a_bits=4 \
  --w_group_size=32 --a_group_size=32 \
  --w_granularity=group --a_granularity=group \
  --w_observer=minmax \
  --gptq \
  --quantization_order=default \
  --transform_search \
  --transform_search_candidates identity hadamard dct dst gsr householder \
  --export_quantized_model=pseudoquant \
  --save_path=outputs-COV/mxfp_gptq_rotsearch_smoke \
  --cpu_offload_modules \
  --fuse_global_scale \
  --amp

python /cephfs/shared/zlouyang/FP-Quant/model_quant.py \
  --model_name_or_path=/cephfs/shared/model/llama-3-8b-instruct \
  --dataset_name_or_path="${OUT_CALIB}/fineweb_calib_1024x2048_tokens.pt" \
  --num_sequences=1024 --sequence_length=2048 --dtype=auto  --seed 0 \
  --format=mxfp --w_bits=4 --a_bits=4 \
  --w_group_size=32 --a_group_size=32 --w_granularity=group --a_granularity=group \
  --w_observer=mse \
  --gptq --quantization_order=default \
  --transform_class=hadamard --hadamard_group_size=32 \
  --export_quantized_model=pseudoquant \
  --save_path=outputs-test/mxfp_gptq_had_g32_nosearch_probe_1 \
  --fuse_global_scale --amp

python /cephfs/shared/zlouyang/FP-Quant/model_quant.py \
  --model_name_or_path=/cephfs/shared/model/llama-3-8b-instruct \
  --dataset_name_or_path="${OUT_CALIB}/fineweb_calib_1024x2048_tokens.pt" \
  --num_sequences=1024 --sequence_length=2048 --dtype=auto --seed 0 \
  --format=mxfp --w_bits=4 --a_bits=4 \
  --w_group_size=32 --a_group_size=32 --w_granularity=group --a_granularity=group \
  --w_observer=mse \
  --gptq --quantization_order=default \
  --transform_search \
  --transform_search_objective=mse \
  --transform_search_candidates hadamard \
  --export_quantized_model=pseudoquant \
  --save_path=outputs-test/mxfp_gptq_hadonly_search_probe_1 \
  --fuse_global_scale --amp \
  --cpu_offload_activations  \  
  --cpu_offload_modules \


lm_eval --model hf \
  --model_args "pretrained=/cephfs/shared/zlouyang/FP-Quant/oyzl_test/outputs-test/mxfp_gptq_had_g32_nosearch_probe,device=cuda,dtype=bfloat16,trust_remote_code=True" \
  --tasks mmlu_cot_llama \
  --batch_size 64 \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --output_path lm_eval_results_probe/mxfp_gptq_had_g32_nosearch_probe/mmlu_cot_llama

lm_eval --model hf \
  --model_args "pretrained=/cephfs/shared/zlouyang/FP-Quant/oyzl_test/outputs-test/mxfp_gptq_hadonly_search_probe,device=cuda,dtype=bfloat16,trust_remote_code=True" \
  --tasks mmlu_cot_llama \
  --batch_size 64 \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --output_path lm_eval_results_probe/mxfp_gptq_hadonly_search_probe/mmlu_cot_llama














