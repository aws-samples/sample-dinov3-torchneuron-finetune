| run | device | workers | mode | compile | precision | trainable | first step | train img/s | test acc | wall | CPU fallback ops |
|---|---|---|---|---|---|---|---|---|---|---|---|
| compile_full_1core | neuron | 1 | full | yes | fp32 | 21.60M | 146s | 123.9 | 89.4% | 235s | none |
| compile_lastn_1core | neuron | 1 | lastn | yes | fp32 | 3.56M | 204s | 438.1 | - | 207s | none |
| compile_lastn_1core_acc | neuron | 1 | lastn | yes | fp32 | 3.56M | 7s | 235.1 | 92.9% | 54s | none |
| compile_lastn_bf16_1core | neuron | 1 | lastn | yes | bf16 | 3.56M | 79s | 639.9 | - | 81s | none |
| compile_lastn_ddp4 | neuron | 4 | lastn | yes | fp32 | 3.56M | 90s | 110.1 | 88.9% | 194s | none |
| compile_linear_1core | neuron | 1 | linear | yes | fp32 | 0.01M | 78s | 99.6 | 77.4% | 188s | none |
| cpu_lastn | cpu | 1 | lastn | no | fp32 | 3.56M | 11s | 12.8 | - | 33s | - |
| eager_lastn_1core | neuron | 1 | lastn | no | fp32 | 3.56M | 158s | 224.8 | - | 165s | none |
