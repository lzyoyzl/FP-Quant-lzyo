## BASE

| model | algo | rotate | h | observer | order | winogrande(acc) | hellaswag(acc_norm) | avg(2 tasks) |
| ----- | ---- | ------ | - | -------- | ----- | --------------- | ------------------- | ------------ |
| BASE  | BASE |        |   |          |       | 0.7553          | 0.7891              | 0.7722       |

## mxfp

| model                                          | algo   | rotate   | h   | observer | order      | winogrande(acc) | hellaswag(acc_norm) | avg(2 tasks) |
| ---------------------------------------------- | ------ | -------- | --- | -------- | ---------- | --------------- | ------------------- | ------------ |
| mxfp_gptq_w4a4_hadamard_h128_minmax_activation | gptq   | hadamard | 128 | minmax   | activation | 0.7269          | 0.7574              | 0.7421       |
| mxfp_gptq_w4a4_hadamard_h128_minmax_default    | gptq   | hadamard | 128 | minmax   | default    | 0.7301          | 0.7563              | 0.7432       |
| mxfp_gptq_w4a4_hadamard_h128_mse_default       | gptq   | hadamard | 128 | mse      | default    | 0.7435          | 0.7583              | 0.7509       |
| mxfp_gptq_w4a4_identity                        | gptq   | identity |     |          | default    | 0.7159          | 0.7423              | 0.7291       |
| mxfp_gptq_w4a4_identity_minmax_activation      | gptq   | identity |     | minmax   | activation | 0.7127          | 0.7446              | 0.7287       |
| mxfp_gptq_w4a4_identity_mse_default            | gptq   | identity |     | mse      | default    | 0.7159          | 0.7394              | 0.7276       |
| mxfp_mrgptq_w4a4_hadamard_h128_mse_activation  | mrgptq | hadamard | 128 | mse      | activation | 0.7459          | 0.7587              | 0.7523       |
| mxfp_rtn_w4a4_hadamard_h128                    | rtn    | hadamard | 128 | minmax   | default    | 0.7190          | 0.7471              | 0.7330       |
| mxfp_rtn_w4a4_hadamard_h32                     | rtn    | hadamard | 32  | minmax   | default    | 0.7269          | 0.7514              | 0.7391       |
| mxfp_rtn_w4a4_identity                         | rtn    | identity |     | minmax   | default    | 0.7080          | 0.7407              | 0.7244       |

## nvfp

| model                                          | algo   | rotate   | h   | observer | order      | winogrande(acc) | hellaswag(acc_norm) | avg(2 tasks) |
| ---------------------------------------------- | ------ | -------- | --- | -------- | ---------- | --------------- | ------------------- | ------------ |
| nvfp_gptq_w4a4_hadamard_h128_minmax_activation | gptq   | hadamard | 128 | minmax   | activation | 0.7395          | 0.7659              | 0.7527       |
| nvfp_gptq_w4a4_hadamard_h128_minmax_default    | gptq   | hadamard | 128 | minmax   | default    | 0.7561          | 0.7651              | 0.7606       |
| nvfp_gptq_w4a4_hadamard_h128_mse_default       | gptq   | hadamard | 128 | mse      | default    | 0.7451          | 0.7688              | 0.7570       |
| nvfp_gptq_w4a4_identity                        | gptq   | identity |     |          | default    | 0.7403          | 0.7659              | 0.7531       |
| nvfp_gptq_w4a4_identity_minmax_activation      | gptq   | identity |     | minmax   | activation | 0.7459          | 0.7684              | 0.7571       |
| nvfp_gptq_w4a4_identity_mse_default            | gptq   | identity |     | mse      | default    | 0.7419          | 0.7617              | 0.7518       |
| nvfp_mrgptq_w4a4_hadamard_h128_mse_activation  | mrgptq | hadamard | 128 | mse      | activation | 0.7553          | 0.7657              | 0.7605       |
| nvfp_rtn_w4a4_hadamard_h128                    | rtn    | hadamard | 128 | minmax   | default    | 0.7293          | 0.7629              | 0.7461       |
| nvfp_rtn_w4a4_hadamard_h16                     | rtn    | hadamard | 16  | minmax   | default    | 0.7293          | 0.7579              | 0.7436       |
| nvfp_rtn_w4a4_identity                         | rtn    | identity |     | minmax   | default    | 0.7474          | 0.7670              | 0.7572       |
