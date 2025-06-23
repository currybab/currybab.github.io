+++
categories = ['papers']
title = 'Llama Pro 정리'
date = 2024-03-26T14:08:18+09:00
tags = ['train']
+++

해당 paper에서는 catastrophic forgetting을 효과적이고 효율적으로 해결하기 위한 구조로 Llama Pro 모델을 제안한다.
catastrpohic forgetting이란 LLM을 training할 때에 오래된 정보를 잊는 것을 의미한다.
예를 들어 code-llama는 llama2를 code-specific한 데이터셋으로 추가 훈련을 시킨 모델인데 코딩을 더 잘 이해하지만 그만큼 llama2에 비해서 일반적인 벤치마크에서 성능이 감소하였다.
해당 논문에서는 효과적인 post training 방법으로 block expansion을 제안한다.

![attention_layer](https://github.com/currybab/currybab.github.io/assets/7679722/d0e59095-aa2a-4d2a-a835-f5391579df9c)

## 사전 지식: LLaMA 블록

라마 블록은 Multi Head Self Attention 블록과 (SwiGlu와 residual connection이 있는) position-wise FFN으로 이루어져있다.
라마 블록의 입력을 $x$, 출력을 $y$라고 하면,
$$ x\prime = x + MHSA(RMSNorm(x)) $$
$$ y = x\prime + FFN(RMSNorm(x\prime)) $$
입력 $x$가 sequence length $n$과 hidden dimension $d$를 가지고 있으면 $n \times d$ 차원을 갖게 된다. 출력 $y$ 역시도 같은 차원을 가진다. MHSA는 다음과 같이 정의된다
$$ MHSA(Q,K,V) = Concat(head_1,...,head_h)W^{O} $$
$$ head_{i} = Attention(x W_{i}^{Q}, x W_{i}^{K}, x W_{i}^{V}) $$
$$ Attention(Q_i, K_i, V_i) = Softmax(\frac{Q_{i}K_{i}^{T}}{\sqrt{d_{k}}}) V_{i} $$
FFN 블록에서 라마는 SwiGLU 활성화 함수를 사용한다. $\otimes$는 element-wise multiplication을 의미한다.
$$ SwiGLU(x, W, V) = SiLU(xW) \otimes (xV) $$
$$ FFN(x) = SwiGLU(x, W_1, W_2)W_3 $$
$$ SiLU(x) = x \otimes \sigma(x) $$

## Block Expansion

![full_model](https://github.com/currybab/currybab.github.io/assets/7679722/f612acf4-b655-43e7-89ee-c851127db96b)

identity block이 추가된 이후에 모델이 기존 모델과 같은 값을 내어야한다. 
즉 identity block은 $ \phi(x) = x $를 만족한다.(입력과 출력이 동일)
또한 이 블록을 기존 블록들에 교차해서 넣는다.

Shen 등에 따르면 identity block에서 Norm 모듈의 scale parameter을 0으로 초기화 하는 것을 제안했는데 라마 프로에서는 이 방법이 잘 work하지 않았다. 이유로는 역전파 동안 손실 함수 L의 기울기가 RMSNorm 가중치 w에 대해 0이 되기 때문으로 이것이 RMSNorm의 훈련을 막기 때문이다.
$$ \frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \frac{\partial FFN(RMSNorm(x \prime))}{\partial RMSNorm(x \prime)} \frac{\partial RMSNorm(x \prime)}{\partial w} = 0 $$
그래서 라마프로에서는 RMSNorm을 변형하는 대신에 $W^{O}$항(o_proj)과 $W_{3}$항(down_proj)의 weight를 0으로 초기화 하였다. 이렇게 함으로써 초기에 잔여 연결만을 통과시킴으로써 기존 모델과 동일한 출력을 가질 수 있었다.

모델이 추가적인 도메인 지식을 수용하면서 일반 지식을 유지할 수 있는 능력을 향상시키기 위해서 LLM의 블록 수를 증가시키기 위해 블록 확장을 사용하였다. 또한 원래의 블록을 freeze하고 새로 추가된 모델만을 파인튜닝함으로써 모델의 일반적인 능력을 보존하였다.


## Training
### Pretrain Detail

- base model로는 LLaMA-2 7B를 사용했다
- 32개 decoder block을 40개까지 늘렸다. 블록 4개당 1개씩 더 생긴 셈이다.
- 코딩과 수학에 적합한 데이터셋을 구축했다.
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

- 5개의 데이터 소스를 합쳐서 사용하였다.
    - ShareGPT
    - WizardLM_evol_instruct_V2
    - SlimOrca
    - MetaMath
    - Evol-CodeAlpaca
- 약 100만개의 sft dataset을 사용하였다.
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

