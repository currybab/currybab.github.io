+++
categories = ['papers']
title = 'MiniCPM 정리'
date = 2024-04-07T13:32:17+09:00
math = true
tags = ['train']
+++

소형 언어모델에 관심이 계속 생겨서 앞으로 관련된 좋은 학습 방법을 공부해보려고 한다.
오늘 정리해 볼 것은 MiniCPM의 [technical report](https://shengdinghu.notion.site/MiniCPM-Unveiling-the-Potential-of-End-side-Large-Language-Models-d4d3a8c426424654a4e80e42a711cb20)이다.
2.4B의 모델로도 상당히 좋은 벤치마크를 중국어와 영어 모두에서 기록하였다고 한다.

뭔가 정리하고 싶었는데 사실상 번역이 된것 같기도 하고...

## Introduction

MiniCPM은 edge-side llm 시리즈로 embedding layer를 제외한 2.4B 파라미터를 가진 MiniCPM-2B를 기반으로 함.

Mistral-7B와 근사한 벤치마크 성적을 기록하였으며 중국어, 코딩, 수학에서는 더 뛰어난 성능을 가졌다고 함. 
또한 Llama2-13B, MPT-30B, Falcon-40B와 같은 모델들보다 더 뛰어난 성능을 기록하였다고 함. 
또한 (user 경험과 유사한 벤치마크인) MTBench에서 대표적인 오픈소스 모델들보다 더 뛰어난 성적을 거둠.

공개한 모델의 목록은 다음과 같음.
- MiniCPM-2B-SFT/DPO (instruction finetuning)
- MiniCPM-V (비전 모델)
- MiniCPM-2B-SFT/DPO-Int4

한계점은 다음과 같음.
- 모델 크기의 제약으로 인해 특히 DPO 모델에서 생성된 더 길고 정교한 응답의 경우 환각적 문제가 나타날 수 있다고 함.
- 학술 연구 목적으로 모델의 일반성을 보장하기 위해 신원 관련 학습을 거치지 않았음. 또한, 공개적으로 사용 가능한 ShareGPT 말뭉치를 학습 데이터의 일부로 활용했기 때문에 이 모델은 GPT 시리즈 모델과 유사한 신원 관련 정보를 생성할 수 있음.
- 모델 크기에 제한을 받는 이 모델은 프롬프트의 영향을 크게 받기 때문에 여러 번 시도했을 때에 일관성 없는 결과가 나올 수 있음.
- 모델 용량의 제약으로 인해 모델의 지식 리콜 정확도가 다소 제한적. 향후 RAG 메소드를 이용해 보강할 예정.

대체로 모델 크기에 따른 제약사항을 적어둠.

## Model Wind Tunnel Experiment

해당 단어를 claude에게 물어보니 실제 크기의 물체 대신에 축소 모형을 사용하여 공기역학적 특성을 실험하는 방법이라고 한다. 
대형 모델의 경우 실험 비용이 많이 들고 configuration 튜닝 없이는 최적의 성능을 달성하기 어렵다고 해서 해당 실험을 구성하였다고 한다.
해당 실험은 크게 다섯 가지 측면에서 진행되었다.
- Hyper-parameters
- Batch size
- Learning Rate
- Learning Rate Scheduler
- Data Strategy

### 1. Scaling Up Models with Stable Hyperparameters

- 기존 학습에서 각 모델에 대한 하이퍼파라미터를 조정하는 작업은 대규모 모델에서는 불가능함.
- $\mu P$ method에서 영감을 받아 모델에서 다양한 파라미터 모듈 간의 연결 가중치를 조정하고 모델 초기화를 수정했음. 이러한 조정 중 일부는 [Cerebras-GPT](https://arxiv.org/abs/2304.03208)와 유사.
- 구체적인 파라미터는 0.009B 모델 규모에서 약 400개의 베이지안 파라미터 검색을 통해 얻었음.

|Name|Specific Operation|
|---|---|
|Embedding Output Scaling|embedding의 output에 12를 곱함|
|Residual Connection Scaling|각 레이어의 각 residual connection에서 1.4/sqrt(num_layers) = 1.4/sqrt(40)로 증분을 스케일링|
|Initialization of Tensors|각 2차원 텐서 파라미터의 초기화 표준 편차를 0.1/sqrt(dim_model/256) = 1/30으로 설정하고 다른 파라미터의 초기화를 0.1로 설정|
|Learning Rate Scaling of Tensors|각 2차원 텐서 파라미터의 학습 속도를 다른 부분의 학습 속도(또는 전체 학습 속도)의 1/(dim_model/256) = 1/9배로 조정|
|lm_head Scaling|출력 로그를 1/(dim_model/256) = 원래 값의 1/9배로 조정|


### 2. Optimal Batchsize

- 배치 사이즈는 모델의 수렴 속도와 리소스 소비 사이의 균형을 결정.
    - 너무 크면 데이터 양과 계산 비용이 크게 증가하여 손실이 발생.
    - 너무 작으면 많은 훈련 단계가 필요하고 손실 함수의 감소가 제한될 수 있음.
- 2020년 OpenAI가 손실 함수와 토큰 수 사이의 관계를 [연구](https://arxiv.org/abs/2001.08361)
    - 실험에서 그들은 더 많은 step을 소비하는 것이 더 많은 시간을 소비하는 것과 같다고 가정함.
    - 이 가정하에 OpenAI는 너무 많은 token이나 step을 소모하지 않고 특정 손실에 도달하는 crtical batchsize(임계 배치 크기, 모델 매개변수 수의 제곱근에 따라 증가하는 것으로 나타남)를 정의.
- 하지만 저자들은 gradient checkpointing strategy를 사용하면서 A100과 같은 GPU를 사용할 때 메모리가 아닌 계산 속도가 주요 병목 현상임을 관찰함.
    - 배치 크기를 두배로 늘리는 것은 단일 step에 걸리는 시간을 두배로 늘리는 것과 같음.
    - 이러한 관찰을 통해 "너무 많은 step을 소비하지 않는다"를 포기하고 토큰 수량을 최소화 하여 손실을 최소화하는 방향으로 전환.
- 실험 결과
    - 각각 0.009B, 0.03B, 0.17B 모델에 대해 6개의 배치 크기로 훈련 실험을 진행.
    - C4 데이터 세트에서 손실 오프셋에 따른 최적의 배치 크기의 추세를 관찰(그림의 빨간색 선).
![Untitled (1)](https://github.com/currybab/currybab.github.io/assets/7679722/20a5d3ac-0b70-423d-8390-718eca6283f6)

위 그림에서 빨간 선을 로그스케일하고 연결하면 아래 그림과 같이 됨.

![Untitled (2)](https://github.com/currybab/currybab.github.io/assets/7679722/3feaf646-564c-4e31-92e1-d82d4b2a952c)

Batch size와 C4 손실 간의 관계를 다음과 같이 구할 수 있었음.
$$BS = \frac{1.2110 \times 10^9}{L^{6.2393}}$$

이 패턴에 따라 2B 모델이 약 2.5의 C4 손실에 도달하려면 4M의 배치 크기가 적합할 것으로 예상했다고 함.


### 3. Optimal Learning Rate

- 하이퍼파라미터를 안정화하는 매개변수화 방식을 사용했기 때문에 모델의 가장 중요한 하이퍼파라미터인 학습률은 모델 규모가 확장되어도 큰 변화를 겪지 않을 것으로 예상. 
- 0.04B, 0.1B, 0.3B, 0.5B에서 6가지 학습 속도 실험을 진행했음. 
- 그 결과 모델 규모가 10배 증가했음에도 불구하고 최적의 학습률은 눈에 띄는 변화를 보이지 않고 약 0.01을 유지하는 것으로 나타남. 
- 한 2.1B 규모로 간단한 검증을 실시한 결과, 학습률 0.01이 실제로 가장 낮은 손실을 달성하는 것으로 확인됨.

![loss_vs_lr](https://github.com/currybab/currybab.github.io/assets/7679722/272b310b-84f7-493f-a6a9-afe0b8d06dca)


### 4. Optimal Learning Rate Scheduler (WSD Scheduler)

Learning Rate Scheduler는 학습의 여러 단계에서 사용되는 학습 속도를 조정하는 기능으로, 모델 성능에 매우 중요한 역할을 함. 
- 현재 일반적으로 사용되는 learning rate scheduler는 코사인 어닐링으로, 워밍업 단계 이후 최고점에 도달한 후 학습 속도를 서서히 낮추는 방식임. 
- 거의 모든 대규모 모델은 코사인 학습 속도 스케줄러(약칭 코사인 LRS)를 사용함.

먼저 코사인 스케줄러가 뛰어난 성능을 보이는 이유를 조사하기 위해 다양한 실험을 진행.
- 0.036B 모델을 사용하여 cutoff step $T$를 조정해가면서 실험함.
- 결과 ![cosine_lrs_cutoff](https://github.com/currybab/currybab.github.io/assets/7679722/fa75e060-ce3e-44c8-82da-821b90f01ae7)
- 그림에서 $S$ 단계까지 훈련된 모델의 경우 Cosine LRS의 컷오프 단계 $T$를 $S$로 설정하면 항상 최적의 성능을 얻을 수 있지만, 더 많거나 더 적은 단계로 설정하면 최적의 결과가 나오지 않음.

하지만 continuous training 시나리오를 생각했을 때에 코사인 스케줄러에서 더 많은 이슈들이 있었다고 함. 
- 코사인 스케줄러의 cutoff 단계 이후에 최대 학습률의 0.1배로 계속 학습하면, continuous training동안 수렴이 매우 느려짐.
- 코사인의 cutoff 단계 이후에 코사인 LRS를 다시 시작하면, 손실이 오랫동안 증가하며 그동안 모델을 사용할 수 없는 상태가 됨.
- 코사인 LRS는 두 가지 이유로 인해 미리 정해진 단계 수를 지정할 때 뛰어난 성능을 발휘하는 것으로 추측함.
    - 1 : $T=S$인 코사인 LRS는 선형 LRS, Noam LRS, $T<S$인 코사인 LRS에 비해 학습률이 높은 훈련 기간이 더 김. 이 단계는 모델이 더 나은 글로벌 최적값을 찾는 데 도움이 될 수 있음.
    - 2 : $T=S$인 코사인 LRS는 $T>S$인 코사인 LRS 및 상수 LRS에 비해 학습률 감소의 어닐링 단계가 더 철저함. 이 단계에서는 모델이 더 나은 로컬 최적점을 찾을 수 있도록 하는 고유한 동적 현상이 발생할 수 있음.
- 두가지를 결합하여 Warmup-Stable-Decay (WSD) scheduler를 제안함.
    - $lr(s)=\begin{cases}\frac{s}{W}*\eta,\ when\ s< W\\\eta,\ when\ W<s<S\\f(s-S)*\eta,\ when\ S<s<S+D\end{cases}$
    ![cosine_vs_wsd](https://github.com/currybab/currybab.github.io/assets/7679722/235896a7-922c-4ab8-b938-29af8a95be04)
    - 3개의 stage로 구성
        - warmup stage
        - stable training stage
        - decay stage
    - 다음과 같은 4가지 장점이 있다고 함.
        - 지속적으로 훈련할 수 있음.
        - 언제든지 꺼낼 수 있습니다.
        - 코사인 LRS보다 성능이 뛰어남.
        - 명시적으로 구분할 수 있는 훈련 단계가 있어 다양한 데이터 전략의 사용이 용이함.
- 예상대로 Decay 단계(어닐링 단계)에서 학습률이 감소함에 따라 손실이 크게 급격히 감소하고 $T=S$ 단계에서 코사인 LRS와 같거나 더 낮아지는 것을 발견. 
    - 동시에 감쇠 이전의 모델을 재사용하여 높은 학습률로 훈련을 계속할 수 있음.
    - $S'$ 단계를 더 수행한 후에는 어닐링을 수행하여 $T'=S'$에서 코사인 LRS와 동일한 효과를 얻을 수 있었음.
    - 10% 어닐링 길이는 코사인 LRS에 도달하거나 이를 능가하기에 충분했음.
    ![Untitled (3)](https://github.com/currybab/currybab.github.io/assets/7679722/8f1b97f4-c088-4bb5-93f6-5e2d932984ff)
- 차트에서 분홍색 선은 0.17B 모델, 녹색 선(직렬)은 WSD를 사용한 0.036B 모델, 노란색 선은 코사인 스케줄링을 사용한 0.036B 모델을 나타내며, 코사인 주기는 학습 데이터가 모델 파라미터의 80배일 때 설정됨. 어닐링 엔드포인트 라인을 대략적으로 추정하여 0.17B 모델의 엔드포인트와 일치하는 것을 확인할 수 있음(자세한 내용은 이 섹션의 6번째 부분 참조).
![Untitled (4)](https://github.com/currybab/currybab.github.io/assets/7679722/9d1209bf-e178-4678-8604-6fa438a8cc84)


### 5. Batchsize Scheduling

- 훈련 중에 배치사이즈를 늘리는 것에 대한 연구임.
- 배치 크기를 더 크게 설정하면 손실율이 낮아질 수 있음.
- 그러나 안타깝게도 정식 실험에서는 배치 크기를 늘린 후 어닐링 단계에서의 손실 감소 효과가 다소 감소했음.
- 따라서 배치 크기를 늘리는 훈련 방법을 채택하지 않았음.


### 6. How many times can continuous training of a fixed-size model reach that of a large model?

- WSD learning rate scheduler는 모든 단계에서 어닐링할 수 있고 해당 단계에서 모델의 최적 성능을 얻을 수 있으므로, 최상의 시나리오에서 Chinchilla-optimal model을 능가할 수 있는 최대 파라미터 크기인 N 크기의 모델을 지속적으로 학습할 수 있는지 살펴볼 기회를 가졌음.
    - Chinchilla-optimal model은 학습에 필요한 최적 데이터 세트의 수는 모델 파라미터 수량의 20배 크기.
- 먼저, countinuous training 중에 모델의 성능이 계산에 따라 어떻게 변화하는지 추정함.
    - exponential($L(C) = \alpha e^{-\beta C} + L_0$)과  power-law($L(C) = \beta C^{-\alpha} + L_0$)로 피팅해봄.
    ![Untitled (5)](https://github.com/currybab/currybab.github.io/assets/7679722/aff03bcb-f65a-48a5-87e4-fb897e3172d2)
    - power-law form이 더 잘 맞은듯 보임.
        - 피팅을 통해 0.036B 모델의 C4 Loss가 3.27에 도달할 수 있을 것이라고 계산함.
        - 비교를 위해 Chinchilla-optimal model의 세팅에 맞추어 0.17B 모델도 함께 훈련함.
        - WSD scheduler를 활용한 0.036B 모델의 C4 Loss가 3.37에 도달했는데 이는 Chinchilla-optimal model의 3.34와 거의 유사함.
        - WSD 스케줄러로 훈련된 모델은 동일한 양의 계산을 수행할 때 약 5배의 파라미터 수를 달성할 수 있다고 생각함.
        - 또한 지속적인 학습을 통해 더 큰 규모의 친칠라 최적화 모델을 능가할 수 있다고 봄.
        - 이들은 data volume과 the non-embedding parameter quantity의 비율이 20이 아닌 100(구체적인 숫자는 아님)이 될 수 있다고 주장.
- MiniCPM을 통해 이를 검증했음. MiniCPM의 최종 C4 Loss는 2.41을 달성하였고 이는 9B의 Chinchilla Model의 C4 Loss인 2.40에 근접함.
    - 주석으로는 어닐링단계에서 고품질 데이터 학습을 진행했기 때문에 사전 훈련 데이터만 적용해서 하면 9B의 수치를 넘는다고 표현함.


### 7. Continuous Training-Friendly Data Strategy

- WSD LRS 모델의 어닐링 단계에서 손실이 크게 감소하기 때문에 이 단계에서 고품질 데이터를 도입하면 다음과 같은 두 가지 이점이 있다고 추측
    - SFT 단계에서 고품질 데이터를 추가하는 것과 비교하여, 어닐링 단계에서 데이터를 도입하면 보다 철저한 모델 학습이 가능하다.
    - 사전 학습 단계에서 고품질 데이터를 추가하는 것과 비교하면 적은 데이터로 학습을 더 잘 지원한다. 
    그렇지 않으면 미리 정해진 훈련 단계가 없는 지속적인 사전 훈련 과정에서 작은 데이터가 너무 많이 반복되어 부정적인 영향을 미칠 수 있다.
- 이러한 두 가지 추측을 바탕으로 다음과 같이 제안함.
    - 사전 훈련 단계에서는 일반적이고 대규모의 거친 품질의 사전 훈련 데이터만 사용.
    - 어닐링 단계에서는 SFT의 고품질 데이터와 함께 광범위한 고품질의 지식 및 역량 데이터를 사전 훈련 데이터에 혼합하여 어닐링에 사용.
- 직접 SFT와 비교하여 우리 방법의 장점을 검증하기 위해 중간 체크포인트부터 두 가지 실험을 진행했습니다.
    - 실험 A: 사전 훈련 데이터만을 사용해 어닐링한 후 4B 토큰 SFT를 사용했습니다.
    - 실험 B: 앞서 언급한 고품질 데이터 + 사전 학습 데이터에 혼합된 SFT 데이터를 사용한 어닐링과 4B 토큰 SFT를 사용한 어닐링.
    - | |CEval|CMMLU|MMLU|GSM8K|Math|HumanEval|MBPP|
      |---|---|---|---|---|---|---|---|
      |A|40.0|41.5|44.6|27.7|5.1|27.7|24.4|
      |B|52.6|51.1|50.9|42.3|5.4|30.4|30.3|
    - 어닐링 초기에 고품질 데이터를 도입하는 것이 어닐링 후 SFT 단계에서 추가하는 것보다 훨씬 더 많은 이점이 있는 것으로 나타남. 
    - 따라서 모델 기능의 전문화 및 향상은 어닐링 단계부터 시작하는 것이 좋음.


## Vocabulary

- vocab size: 122,753
    - 6.57M Chinese documents
    - 6.69M English documents
    - 3M code documents
    - 2.5M mathematical documents
- sentencepiece library for Byte Pair Encoding (BPE) 사용

## Two-Stage Pretraining

### 1. Stable Training Stage

- 1테라바이트의 중복 제거된 데이터를 활용, 대부분의 데이터는 오픈 데이터 세트에서 가져옴.
- 실험 중에 발견된 최적의 구성인 WSD LRS를 사용했으며, 배치 크기는 393만 개, 최대 학습률은 0.01로 설정.

![Data Mixture of Stable Stage](https://github.com/currybab/currybab.github.io/assets/7679722/9b154c83-8253-4888-a695-663de1964488)

### 2. Annealing Phase

구글에 검색해보니 "어닐링(annealing, 풀림) 은 재결정화 온도 이상의 고온에서 오랫동안 금속을 노출시켜 금속을 더 부드럽게 만드는 작업이다."라고 나옴.

![decay_data_mixture](https://github.com/currybab/currybab.github.io/assets/7679722/80281dd9-c2ac-4d58-8453-6e4438554444)

- 263,000스텝(약 1조 개의 데이터 포인트)에서 어닐링이 시작됨.
- 어닐링 과정에서도 손실 함수가 급격히 감소하는 것을 볼 수 있음. 
- 또한 다양한 작업 데이터와 SFT 데이터에서도 손실이 눈에 띄게 감소한다고 함.
- 어닐링 단계에서는 WSD 스케줄러를 특별하게 변형하였고 지수 어닐링을 사용함.
$$f(s-S）= \eta \times 0.5^{(s-S)/T}$$

어닐링 단계에서 C4 손실의 변화 (아래 그림)
![c4_loss_annealing_phase](https://github.com/currybab/currybab.github.io/assets/7679722/7179ee70-a7a5-470a-a0b1-6da09b9f17be)

## Alignment

- 어닐링 단계에서 SFT 데이터를 통합했음에도 불구하고 별도의 SFT 단계가 필요하다는 것을 알게 됨. 즉, 어닐링과 SFT 단계는 모두 필수적인 과정. 
    - 사전 훈련 데이터를 제외하고 어닐링 단계와 유사한 SFT 데이터를 활용하여 약 6B tokens으로 SFT 훈련을 진행. 
    - SFT의 학습 속도는 어닐링 종료 시점에 맞춰 1e-3으로 설정, WSD 스케줄러도 사용.
- SFT 이후에는 모델의 추가 인간 선호도 조정을 위해 DPO를 사용. 
    - 이 단계에서는 모델의 코드와 수학적 기능을 향상시키기 위해 기본 정렬 데이터셋으로 UltraFeedback을 활용하고, 내부 선호도 데이터셋을 구축. 
    - 1e-5의 학습률로 한 번의 DPO 학습을 수행했으며 코사인 스케줄러를 활용. 
- DPO 및 데이터 설정에 대한 자세한 내용은 해당 연구자들이 이전에 작성한 [UltraFeedback paper](https://arxiv.org/pdf/2310.01377.pdf)을 참고하라고 함.

## Comprehensive Evaluation

### Benchmark - SFT Model: MiniCPM-sft

저자들의 open source인 [UltraEval](https://github.com/OpenBMB/UltraEvalhttps://github.com/OpenBMB/UltraEval)을 사용함.
데이터셋트로는 다음을 사용함.
- English, using MMLU
- Chinese, utilizing CMMLU and C-Eval
- Code, employing HumanEval and MBPP
- Mathematics, incorporating GSM8K and MATH
- Question-Answering, covering HellaSwag, ARC-E, ARC-C
- Logic, utilizing BBH


전반적으로 언급된 데이터 세트에서 MiniCPM은 영어에서는 Mistral-7B-v0.1과 비슷한 성능을 보였지만 중국어에서는 Mistral-7B-v0.1보다 훨씬 뛰어난 성능을 보였음.

대부분의 7B 규모 모델보다 성능이 뛰어나거나 비슷한 수준이며, 10B 규모 이상의 일부 모델보다 성능이 뛰어남.

비슷한 사이즈의 소규모 모델과 비교하면 특정 영어 평가 데이터 세트를 제외한 모든 테스트 세트에서 사용 가능한 모든 모델보다 성능이 우수했음.

특이한 점으로는 Phi-2의 평가 결과는 Mistral-7B를 능가하지만 실제 사용자 경험은 동등한 수준에 이르지 못하는 것으로 나타났다고 함.

> QA 작업을 테스트할 때는 일반적으로 두 가지 접근 방식이 사용됩니다. 첫 번째는 질문의 연속으로 옵션을 확장할 때 선택 기준이 되는 난해성(PPL)을 사용하는 것입니다. 두 번째는 모델이 직접 답변 옵션을 출력하는 직접 생성 방식입니다. 이 두 가지 방법을 사용하여 얻은 결과에는 상당한 차이가 있음을 관찰했습니다. 실제로 MiniCPM은 직접 생성과 PPL 테스트에서 비슷한 성능을 보였으며, 직접 생성에서 더 나은 성능을 보였습니다. 반면, Mistral-7B-v0.1은 PPL 테스트에서는 더 나은 성능을 보였지만 직접 생성에서는 더 낮은 성능을 보였습니다. 이러한 현상을 해결하기 위해 각 모델에 대한 점수를 보고할 때 가장 높은 점수를 얻은 평가 방법의 점수를 채택하여 비교의 공정성을 보장합니다.


### Benchmark - DPO Model: MiniCPM-dpo

preference alignment를 위해 DPO를 적용한 이후 MT-Bench 점수가 6.89(SFT)에서 7.25로 올랐다고 함.(Llama2-70B-Chat를 넘는 수치임)

주석으로 해당 MT-bench 점수에 대해서 유의사항을 적어놓았음.
> UltraFeedback과 같은 일부 SFT 데이터와 강화 학습 데이터는 MTBench의 평가에 유리할 수 있음. 
예를 들어, 모델의 생성 스타일을 개선하여 GPT-4 평가자의 선호도에 더 부합할 수 있으며, Zephyr-7B도 유사한 편향성을 보일 수 있음. 
따라서 연구자들은 리더보드의 평가 결과를 차분히 살펴볼 것을 권장함.
또한, 작은 2B 모델로서 많은 작업에서 여전히 Llama2-70B-Chat보다 약할 수 있음.

![dpo_benchmark](https://github.com/currybab/currybab.github.io/assets/7679722/caf1d2d1-17d6-4be5-968b-e05ce4a98645)