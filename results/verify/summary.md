| run | device | workers | mode | compile | precision | trainable | first step | train img/s | test acc | wall | CPU fallback ops |
|---|---|---|---|---|---|---|---|---|---|---|---|
| verify_compile_b | neuron | 1 | lastn | yes | fp32 | 14.20M | 105s | 185.7 | - | 112s | none |
| verify_compile_b_acc | neuron | 1 | lastn | yes | fp32 | 14.20M | 10s | 65.7 | 96.0% | 177s | none |
| verify_compile_s | neuron | 1 | lastn | yes | fp32 | 3.56M | 196s | 408.5 | - | 200s | none |
| verify_ddp4_acc | neuron | 4 | lastn | yes | fp32 | 3.56M | 8s | 100.7 | 88.9% | 112s | none |
| verify_ddp4_speed | neuron | 4 | lastn | yes | fp32 | 3.56M | 99s | 1162.7 | - | 114s | none |
| verify_eager_s | neuron | 1 | lastn | no | fp32 | 3.56M | 175s | 204.5 | - | 182s | none |
| verify_linear_lr1e-2 | neuron | 1 | linear | yes | fp32 | 0.01M | 88s | 92.2 | 87.4% | 207s | none |
