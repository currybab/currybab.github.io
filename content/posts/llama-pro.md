+++
categories = ['papers']
title = 'Llama Pro 정리'
date = 2024-03-26T14:08:18+09:00
tags = ['train']
+++

efficiently and effectively improving the model’s knowledge without catastrophic forgetting

### Block Expansion

![full_model](https://github.com/currybab/currybab.github.io/assets/7679722/f612acf4-b655-43e7-89ee-c851127db96b)

![attention_layer](https://github.com/currybab/currybab.github.io/assets/7679722/d0e59095-aa2a-4d2a-a835-f5391579df9c)

### Pretrain Detail

- base model: LLaMA-2 7B
- expand blocks from 32 to 40
    - P=1, M=4, N=8
- constrct data for coding and math
    - Proof-Pile-2: 55B, weight 1.0
        - AlgebraicStack: 11B 
        - OpenWebMath: 15B
        - ArXiv: 29B
    - The-Stack-Dedup
        - Python: 22B, weight 1.5
- training params
    - batch size: 1024 
    - sequence length: 4096 
    - learning rate: 2e-4 (with Cosine learning rate scheduler, a warmup ratio of 6%)
    - bf16 mixed precision
    - weight decay: 0.1
    - gradient clipping: 1.0
    - apply the flash-attention mechanism
    - trained for a total of 15,900 (approximately 2830 H800 GPU hours)

### SFT Detail

- combine five data sources
    - ShareGPT
    - WizardLM_evol_instruct_V2
    - SlimOrca
    - MetaMath
    - Evol-CodeAlpaca
- final sft dataset consists of approximately 1M samples
- training params
    - batch size: 128
    - sequence length: 4096
    - learning rate of 2e-5 (with Cosine learning rate scheduler, 0.03 warmup ratio)
    - bf16 mixed precision

### Ablation Study

- Lora, 일반적인 fine tuning 방법과 비교하였을 때 우수함을 보여줌.
    - Overall Performance (OP): 이 지표는 모델이 여러 작업이나 데이터 세트에 대해 학습한 후의 전반적인 성능을 평가합니다. OP는 모델이 새로운 정보를 학습하면서도 이전에 학습한 정보를 유지하고 활용할 수 있는 능력을 종합적으로 측정합니다. 따라서, 높은 OP 점수는 모델이 새로운 작업을 학습하는 동안 이전 작업에 대한 성능을 잘 유지하고 있다는 것을 의미합니다.
    - Backward Transfer (BWT): BWT는 모델이 새로운 작업을 학습함에 따라 이전에 학습한 작업의 성능에 미치는 영향을 평가합니다. 긍정적인 BWT 점수는 새로운 학습이 이전 학습에 긍정적인 영향을 미쳤음을 나타내며, 이는 새로운 정보를 통해 모델이 이전 작업에서 더 잘 수행될 수 있음을 의미합니다. 반면, 부정적인 BWT 점수는 새로운 학습이 이전 작업의 성능을 저하시켰음을 나타내며, 이는 일반적으로 '잊어버림(forgetting)'이라고 불리는 현상을 나타냅니다.
- 다양한 expansion block 수 및 위치에 대해서 현재의 방식으로 8블록을 늘리는게 좋았다고 보여줌.
