# lm_eval 结果汇总

- 生成时间：`2026-01-27 17:05:02`
- 结果目录：`/cephfs/shared/zlouyang/FP-Quant/oyzl_test/lm_eval_results_tasks_llama_2_3-Qwen_3`

## 总览（主指标）

| model_id | variant | boolq (acc) | arc_easy (acc_norm/acc) | arc_challenge (acc_norm/acc) | piqa (acc) | winogrande (acc) |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3-8B | BASE | 0.8661 (acc,none) | 0.8342 (acc,none) | 0.6493 (acc,none) | 0.7661 (acc,none) | 0.7056 (acc,none) |
| Qwen3-8B | mxfp_gptq_w4a4_hadamard_h128_mse_default | 0.8572 (acc,none) | 0.8043 (acc,none) | 0.6126 (acc,none) | 0.7595 (acc,none) | 0.6819 (acc,none) |
| Qwen3-8B | mxfp_mrgptq_w4a4_hadamard_h128_mse_activation | 0.8547 (acc,none) | 0.8203 (acc,none) | 0.6246 (acc,none) | 0.7563 (acc,none) | 0.6906 (acc,none) |
| llama-2-7b-hf | BASE | 0.7936 (acc,none) | 0.7563 (acc,none) | 0.4957 (acc,none) | 0.7818 (acc,none) | 0.7435 (acc,none) |
| llama-2-7b-hf | mxfp_gptq_w4a4_hadamard_h128_mse_default | 0.7734 (acc,none) | 0.7534 (acc,none) | 0.4855 (acc,none) | 0.7704 (acc,none) | 0.7198 (acc,none) |
| llama-2-7b-hf | mxfp_mrgptq_w4a4_hadamard_h128_mse_activation | 0.7575 (acc,none) | 0.7517 (acc,none) | 0.4701 (acc,none) | 0.7720 (acc,none) | 0.7167 (acc,none) |
| llama-3-8b-hf | BASE | 0.8214 (acc,none) | 0.8072 (acc,none) | 0.5572 (acc,none) | 0.7878 (acc,none) | 0.7822 (acc,none) |
| llama-3-8b-hf | mxfp_gptq_w4a4_hadamard_h128_mse_default | 0.7908 (acc,none) | 0.7731 (acc,none) | 0.5162 (acc,none) | 0.7666 (acc,none) | 0.7324 (acc,none) |
| llama-3-8b-hf | mxfp_mrgptq_w4a4_hadamard_h128_mse_activation | 0.7957 (acc,none) | 0.7795 (acc,none) | 0.4991 (acc,none) | 0.7753 (acc,none) | 0.7245 (acc,none) |

## 任务明细（所有指标）

### boolq

| model_id | variant | metrics | source_json |
| --- | --- | --- | --- |
| Qwen3-8B | BASE | acc,none=0.866055, acc_stderr,none=0.005957 | Qwen3-8B/BASE/boolq_full/__cephfs__shared__model__Qwen3-8B/results_2026-01-27T12-27-31.599208.json |
| Qwen3-8B | mxfp_gptq_w4a4_hadamard_h128_mse_default | acc,none=0.857187, acc_stderr,none=0.006119 | Qwen3-8B/mxfp_gptq_w4a4_hadamard_h128_mse_default/boolq_full/__cephfs__shared__zlouyang__FP-Quant__oyzl_test__results_quant__Qwen3-8B__mxfp_gptq_w4a4_hadamard_h128_mse_default/results_2026-01-27T12-30-42.040244.json |
| Qwen3-8B | mxfp_mrgptq_w4a4_hadamard_h128_mse_activation | acc,none=0.854740, acc_stderr,none=0.006163 | Qwen3-8B/mxfp_mrgptq_w4a4_hadamard_h128_mse_activation/boolq_full/__cephfs__shared__zlouyang__FP-Quant__oyzl_test__results_quant__Qwen3-8B__mxfp_mrgptq_w4a4_hadamard_h128_mse_activation/results_2026-01-27T12-31-58.548662.json |
| llama-2-7b-hf | BASE | acc,none=0.793578, acc_stderr,none=0.007079 | llama-2-7b-hf/BASE/boolq_full/__cephfs__shared__model__llama-2-7b-hf/results_2026-01-27T12-16-30.951513.json |
| llama-2-7b-hf | mxfp_gptq_w4a4_hadamard_h128_mse_default | acc,none=0.773394, acc_stderr,none=0.007322 | llama-2-7b-hf/mxfp_gptq_w4a4_hadamard_h128_mse_default/boolq_full/__cephfs__shared__zlouyang__FP-Quant__oyzl_test__results_quant__llama-2-7b-hf__mxfp_gptq_w4a4_hadamard_h128_mse_default/results_2026-01-27T12-20-12.000129.json |
| llama-2-7b-hf | mxfp_mrgptq_w4a4_hadamard_h128_mse_activation | acc,none=0.757492, acc_stderr,none=0.007496 | llama-2-7b-hf/mxfp_mrgptq_w4a4_hadamard_h128_mse_activation/boolq_full/__cephfs__shared__zlouyang__FP-Quant__oyzl_test__results_quant__llama-2-7b-hf__mxfp_mrgptq_w4a4_hadamard_h128_mse_activation/results_2026-01-27T12-21-19.209427.json |
| llama-3-8b-hf | BASE | acc,none=0.821407, acc_stderr,none=0.006699 | llama-3-8b-hf/BASE/boolq_full/__cephfs__shared__model__llama-3-8b-hf/results_2026-01-27T12-22-18.813016.json |
| llama-3-8b-hf | mxfp_gptq_w4a4_hadamard_h128_mse_default | acc,none=0.790826, acc_stderr,none=0.007114 | llama-3-8b-hf/mxfp_gptq_w4a4_hadamard_h128_mse_default/boolq_full/__cephfs__shared__zlouyang__FP-Quant__oyzl_test__results_quant__llama-3-8b-hf__mxfp_gptq_w4a4_hadamard_h128_mse_default/results_2026-01-27T12-25-18.195661.json |
| llama-3-8b-hf | mxfp_mrgptq_w4a4_hadamard_h128_mse_activation | acc,none=0.795719, acc_stderr,none=0.007052 | llama-3-8b-hf/mxfp_mrgptq_w4a4_hadamard_h128_mse_activation/boolq_full/__cephfs__shared__zlouyang__FP-Quant__oyzl_test__results_quant__llama-3-8b-hf__mxfp_mrgptq_w4a4_hadamard_h128_mse_activation/results_2026-01-27T12-26-25.794261.json |

### arc_easy

| model_id | variant | metrics | source_json |
| --- | --- | --- | --- |
| Qwen3-8B | BASE | acc,none=0.834175, acc_stderr,none=0.007632, acc_norm,none=0.809343, acc_norm_stderr,none=0.008060 | Qwen3-8B/BASE/arc_easy_full/__cephfs__shared__model__Qwen3-8B/results_2026-01-27T12-39-39.261663.json |
| Qwen3-8B | mxfp_gptq_w4a4_hadamard_h128_mse_default | acc,none=0.804293, acc_stderr,none=0.008141, acc_norm,none=0.787879, acc_norm_stderr,none=0.008389 | Qwen3-8B/mxfp_gptq_w4a4_hadamard_h128_mse_default/arc_easy_full/__cephfs__shared__zlouyang__FP-Quant__oyzl_test__results_quant__Qwen3-8B__mxfp_gptq_w4a4_hadamard_h128_mse_default/results_2026-01-27T12-40-56.439376.json |
| Qwen3-8B | mxfp_mrgptq_w4a4_hadamard_h128_mse_activation | acc,none=0.820286, acc_stderr,none=0.007878, acc_norm,none=0.791246, acc_norm_stderr,none=0.008340 | Qwen3-8B/mxfp_mrgptq_w4a4_hadamard_h128_mse_activation/arc_easy_full/__cephfs__shared__zlouyang__FP-Quant__oyzl_test__results_quant__Qwen3-8B__mxfp_mrgptq_w4a4_hadamard_h128_mse_activation/results_2026-01-27T12-42-12.158005.json |
| llama-2-7b-hf | BASE | acc,none=0.756313, acc_stderr,none=0.008809, acc_norm,none=0.738636, acc_norm_stderr,none=0.009016 | llama-2-7b-hf/BASE/arc_easy_full/__cephfs__shared__model__llama-2-7b-hf/results_2026-01-27T12-32-53.176297.json |
| llama-2-7b-hf | mxfp_gptq_w4a4_hadamard_h128_mse_default | acc,none=0.753367, acc_stderr,none=0.008845, acc_norm,none=0.735269, acc_norm_stderr,none=0.009053 | llama-2-7b-hf/mxfp_gptq_w4a4_hadamard_h128_mse_default/arc_easy_full/__cephfs__shared__zlouyang__FP-Quant__oyzl_test__results_quant__llama-2-7b-hf__mxfp_gptq_w4a4_hadamard_h128_mse_default/results_2026-01-27T12-34-07.227918.json |
| llama-2-7b-hf | mxfp_mrgptq_w4a4_hadamard_h128_mse_activation | acc,none=0.751684, acc_stderr,none=0.008865, acc_norm,none=0.723064, acc_norm_stderr,none=0.009182 | llama-2-7b-hf/mxfp_mrgptq_w4a4_hadamard_h128_mse_activation/arc_easy_full/__cephfs__shared__zlouyang__FP-Quant__oyzl_test__results_quant__llama-2-7b-hf__mxfp_mrgptq_w4a4_hadamard_h128_mse_activation/results_2026-01-27T12-35-17.672709.json |
| llama-3-8b-hf | BASE | acc,none=0.807239, acc_stderr,none=0.008094, acc_norm,none=0.775253, acc_norm_stderr,none=0.008565 | llama-3-8b-hf/BASE/arc_easy_full/__cephfs__shared__model__llama-3-8b-hf/results_2026-01-27T12-36-15.071106.json |
| llama-3-8b-hf | mxfp_gptq_w4a4_hadamard_h128_mse_default | acc,none=0.773148, acc_stderr,none=0.008594, acc_norm,none=0.755051, acc_norm_stderr,none=0.008825 | llama-3-8b-hf/mxfp_gptq_w4a4_hadamard_h128_mse_default/arc_easy_full/__cephfs__shared__zlouyang__FP-Quant__oyzl_test__results_quant__llama-3-8b-hf__mxfp_gptq_w4a4_hadamard_h128_mse_default/results_2026-01-27T12-37-24.520736.json |
| llama-3-8b-hf | mxfp_mrgptq_w4a4_hadamard_h128_mse_activation | acc,none=0.779461, acc_stderr,none=0.008508, acc_norm,none=0.766414, acc_norm_stderr,none=0.008682 | llama-3-8b-hf/mxfp_mrgptq_w4a4_hadamard_h128_mse_activation/arc_easy_full/__cephfs__shared__zlouyang__FP-Quant__oyzl_test__results_quant__llama-3-8b-hf__mxfp_mrgptq_w4a4_hadamard_h128_mse_activation/results_2026-01-27T12-38-37.724714.json |

### arc_challenge

| model_id | variant | metrics | source_json |
| --- | --- | --- | --- |
| Qwen3-8B | BASE | acc,none=0.649317, acc_stderr,none=0.013945, acc_norm,none=0.672355, acc_norm_stderr,none=0.013716 | Qwen3-8B/BASE/arc_challenge_full/__cephfs__shared__model__Qwen3-8B/results_2026-01-27T13-50-05.764400.json |
| Qwen3-8B | mxfp_gptq_w4a4_hadamard_h128_mse_default | acc,none=0.612628, acc_stderr,none=0.014236, acc_norm,none=0.642491, acc_norm_stderr,none=0.014005 | Qwen3-8B/mxfp_gptq_w4a4_hadamard_h128_mse_default/arc_challenge_full/__cephfs__shared__zlouyang__FP-Quant__oyzl_test__results_quant__Qwen3-8B__mxfp_gptq_w4a4_hadamard_h128_mse_default/results_2026-01-27T13-58-26.843834.json |
| Qwen3-8B | mxfp_mrgptq_w4a4_hadamard_h128_mse_activation | acc,none=0.624573, acc_stderr,none=0.014151, acc_norm,none=0.653584, acc_norm_stderr,none=0.013905 | Qwen3-8B/mxfp_mrgptq_w4a4_hadamard_h128_mse_activation/arc_challenge_full/__cephfs__shared__zlouyang__FP-Quant__oyzl_test__results_quant__Qwen3-8B__mxfp_mrgptq_w4a4_hadamard_h128_mse_activation/results_2026-01-27T14-03-10.638041.json |
| llama-2-7b-hf | BASE | acc,none=0.495734, acc_stderr,none=0.014611, acc_norm,none=0.525597, acc_norm_stderr,none=0.014592 | llama-2-7b-hf/BASE/arc_challenge_full/__cephfs__shared__model__llama-2-7b-hf/results_2026-01-27T12-45-57.731629.json |
| llama-2-7b-hf | mxfp_gptq_w4a4_hadamard_h128_mse_default | acc,none=0.485495, acc_stderr,none=0.014605, acc_norm,none=0.518771, acc_norm_stderr,none=0.014601 | llama-2-7b-hf/mxfp_gptq_w4a4_hadamard_h128_mse_default/arc_challenge_full/__cephfs__shared__zlouyang__FP-Quant__oyzl_test__results_quant__llama-2-7b-hf__mxfp_gptq_w4a4_hadamard_h128_mse_default/results_2026-01-27T12-58-27.765477.json |
| llama-2-7b-hf | mxfp_mrgptq_w4a4_hadamard_h128_mse_activation | acc,none=0.470137, acc_stderr,none=0.014585, acc_norm,none=0.504266, acc_norm_stderr,none=0.014611 | llama-2-7b-hf/mxfp_mrgptq_w4a4_hadamard_h128_mse_activation/arc_challenge_full/__cephfs__shared__zlouyang__FP-Quant__oyzl_test__results_quant__llama-2-7b-hf__mxfp_mrgptq_w4a4_hadamard_h128_mse_activation/results_2026-01-27T13-04-35.185085.json |
| llama-3-8b-hf | BASE | acc,none=0.557167, acc_stderr,none=0.014516, acc_norm,none=0.581911, acc_norm_stderr,none=0.014414 | llama-3-8b-hf/BASE/arc_challenge_full/__cephfs__shared__model__llama-3-8b-hf/results_2026-01-27T13-14-19.818109.json |
| llama-3-8b-hf | mxfp_gptq_w4a4_hadamard_h128_mse_default | acc,none=0.516212, acc_stderr,none=0.014604, acc_norm,none=0.546075, acc_norm_stderr,none=0.014549 | llama-3-8b-hf/mxfp_gptq_w4a4_hadamard_h128_mse_default/arc_challenge_full/__cephfs__shared__zlouyang__FP-Quant__oyzl_test__results_quant__llama-3-8b-hf__mxfp_gptq_w4a4_hadamard_h128_mse_default/results_2026-01-27T13-24-53.261566.json |
| llama-3-8b-hf | mxfp_mrgptq_w4a4_hadamard_h128_mse_activation | acc,none=0.499147, acc_stderr,none=0.014611, acc_norm,none=0.537543, acc_norm_stderr,none=0.014570 | llama-3-8b-hf/mxfp_mrgptq_w4a4_hadamard_h128_mse_activation/arc_challenge_full/__cephfs__shared__zlouyang__FP-Quant__oyzl_test__results_quant__llama-3-8b-hf__mxfp_mrgptq_w4a4_hadamard_h128_mse_activation/results_2026-01-27T13-37-11.734126.json |

### piqa

| model_id | variant | metrics | source_json |
| --- | --- | --- | --- |
| Qwen3-8B | BASE | acc,none=0.766050, acc_stderr,none=0.009877, acc_norm,none=0.778020, acc_norm_stderr,none=0.009696 | Qwen3-8B/BASE/piqa_full/__cephfs__shared__model__Qwen3-8B/results_2026-01-27T14-14-46.669788.json |
| Qwen3-8B | mxfp_gptq_w4a4_hadamard_h128_mse_default | acc,none=0.759521, acc_stderr,none=0.009971, acc_norm,none=0.758977, acc_norm_stderr,none=0.009979 | Qwen3-8B/mxfp_gptq_w4a4_hadamard_h128_mse_default/piqa_full/__cephfs__shared__zlouyang__FP-Quant__oyzl_test__results_quant__Qwen3-8B__mxfp_gptq_w4a4_hadamard_h128_mse_default/results_2026-01-27T14-16-08.625564.json |
| Qwen3-8B | mxfp_mrgptq_w4a4_hadamard_h128_mse_activation | acc,none=0.756257, acc_stderr,none=0.010017, acc_norm,none=0.763874, acc_norm_stderr,none=0.009909 | Qwen3-8B/mxfp_mrgptq_w4a4_hadamard_h128_mse_activation/piqa_full/__cephfs__shared__zlouyang__FP-Quant__oyzl_test__results_quant__Qwen3-8B__mxfp_mrgptq_w4a4_hadamard_h128_mse_activation/results_2026-01-27T14-17-47.630589.json |
| llama-2-7b-hf | BASE | acc,none=0.781828, acc_stderr,none=0.009636, acc_norm,none=0.784004, acc_norm_stderr,none=0.009601 | llama-2-7b-hf/BASE/piqa_full/__cephfs__shared__model__llama-2-7b-hf/results_2026-01-27T14-04-30.860215.json |
| llama-2-7b-hf | mxfp_gptq_w4a4_hadamard_h128_mse_default | acc,none=0.770403, acc_stderr,none=0.009813, acc_norm,none=0.781284, acc_norm_stderr,none=0.009645 | llama-2-7b-hf/mxfp_gptq_w4a4_hadamard_h128_mse_default/piqa_full/__cephfs__shared__zlouyang__FP-Quant__oyzl_test__results_quant__llama-2-7b-hf__mxfp_gptq_w4a4_hadamard_h128_mse_default/results_2026-01-27T14-08-11.451103.json |
| llama-2-7b-hf | mxfp_mrgptq_w4a4_hadamard_h128_mse_activation | acc,none=0.772035, acc_stderr,none=0.009788, acc_norm,none=0.773123, acc_norm_stderr,none=0.009772 | llama-2-7b-hf/mxfp_mrgptq_w4a4_hadamard_h128_mse_activation/piqa_full/__cephfs__shared__zlouyang__FP-Quant__oyzl_test__results_quant__llama-2-7b-hf__mxfp_mrgptq_w4a4_hadamard_h128_mse_activation/results_2026-01-27T14-09-22.952592.json |
| llama-3-8b-hf | BASE | acc,none=0.787813, acc_stderr,none=0.009539, acc_norm,none=0.805767, acc_norm_stderr,none=0.009230 | llama-3-8b-hf/BASE/piqa_full/__cephfs__shared__model__llama-3-8b-hf/results_2026-01-27T14-10-36.690498.json |
| llama-3-8b-hf | mxfp_gptq_w4a4_hadamard_h128_mse_default | acc,none=0.766594, acc_stderr,none=0.009869, acc_norm,none=0.778020, acc_norm_stderr,none=0.009696 | llama-3-8b-hf/mxfp_gptq_w4a4_hadamard_h128_mse_default/piqa_full/__cephfs__shared__zlouyang__FP-Quant__oyzl_test__results_quant__llama-3-8b-hf__mxfp_gptq_w4a4_hadamard_h128_mse_default/results_2026-01-27T14-12-01.525240.json |
| llama-3-8b-hf | mxfp_mrgptq_w4a4_hadamard_h128_mse_activation | acc,none=0.775299, acc_stderr,none=0.009738, acc_norm,none=0.785092, acc_norm_stderr,none=0.009584 | llama-3-8b-hf/mxfp_mrgptq_w4a4_hadamard_h128_mse_activation/piqa_full/__cephfs__shared__zlouyang__FP-Quant__oyzl_test__results_quant__llama-3-8b-hf__mxfp_mrgptq_w4a4_hadamard_h128_mse_activation/results_2026-01-27T14-13-36.360936.json |

### winogrande

| model_id | variant | metrics | source_json |
| --- | --- | --- | --- |
| Qwen3-8B | BASE | acc,none=0.705604, acc_stderr,none=0.012809 | Qwen3-8B/BASE/winogrande_full/__cephfs__shared__model__Qwen3-8B/results_2026-01-27T14-28-18.276195.json |
| Qwen3-8B | mxfp_gptq_w4a4_hadamard_h128_mse_default | acc,none=0.681926, acc_stderr,none=0.013089 | Qwen3-8B/mxfp_gptq_w4a4_hadamard_h128_mse_default/winogrande_full/__cephfs__shared__zlouyang__FP-Quant__oyzl_test__results_quant__Qwen3-8B__mxfp_gptq_w4a4_hadamard_h128_mse_default/results_2026-01-27T14-29-31.214605.json |
| Qwen3-8B | mxfp_mrgptq_w4a4_hadamard_h128_mse_activation | acc,none=0.690608, acc_stderr,none=0.012991 | Qwen3-8B/mxfp_mrgptq_w4a4_hadamard_h128_mse_activation/winogrande_full/__cephfs__shared__zlouyang__FP-Quant__oyzl_test__results_quant__Qwen3-8B__mxfp_mrgptq_w4a4_hadamard_h128_mse_activation/results_2026-01-27T14-30-44.159360.json |
| llama-2-7b-hf | BASE | acc,none=0.743489, acc_stderr,none=0.012274 | llama-2-7b-hf/BASE/winogrande_full/__cephfs__shared__model__llama-2-7b-hf/results_2026-01-27T14-18-40.009914.json |
| llama-2-7b-hf | mxfp_gptq_w4a4_hadamard_h128_mse_default | acc,none=0.719811, acc_stderr,none=0.012622 | llama-2-7b-hf/mxfp_gptq_w4a4_hadamard_h128_mse_default/winogrande_full/__cephfs__shared__zlouyang__FP-Quant__oyzl_test__results_quant__llama-2-7b-hf__mxfp_gptq_w4a4_hadamard_h128_mse_default/results_2026-01-27T14-19-57.032230.json |
| llama-2-7b-hf | mxfp_mrgptq_w4a4_hadamard_h128_mse_activation | acc,none=0.716654, acc_stderr,none=0.012665 | llama-2-7b-hf/mxfp_mrgptq_w4a4_hadamard_h128_mse_activation/winogrande_full/__cephfs__shared__zlouyang__FP-Quant__oyzl_test__results_quant__llama-2-7b-hf__mxfp_mrgptq_w4a4_hadamard_h128_mse_activation/results_2026-01-27T14-21-45.480508.json |
| llama-3-8b-hf | BASE | acc,none=0.782163, acc_stderr,none=0.011601 | llama-3-8b-hf/BASE/winogrande_full/__cephfs__shared__model__llama-3-8b-hf/results_2026-01-27T14-23-49.641126.json |
| llama-3-8b-hf | mxfp_gptq_w4a4_hadamard_h128_mse_default | acc,none=0.732439, acc_stderr,none=0.012442 | llama-3-8b-hf/mxfp_gptq_w4a4_hadamard_h128_mse_default/winogrande_full/__cephfs__shared__zlouyang__FP-Quant__oyzl_test__results_quant__llama-3-8b-hf__mxfp_gptq_w4a4_hadamard_h128_mse_default/results_2026-01-27T14-24-50.287157.json |
| llama-3-8b-hf | mxfp_mrgptq_w4a4_hadamard_h128_mse_activation | acc,none=0.724546, acc_stderr,none=0.012556 | llama-3-8b-hf/mxfp_mrgptq_w4a4_hadamard_h128_mse_activation/winogrande_full/__cephfs__shared__zlouyang__FP-Quant__oyzl_test__results_quant__llama-3-8b-hf__mxfp_mrgptq_w4a4_hadamard_h128_mse_activation/results_2026-01-27T14-26-04.834674.json |
