+++
title = 'Energy Based Model'
date = 2024-06-01T21:37:33+09:00
categories = ['generation modeling']
math = true
tags = ['train']
+++

[Cornell CS 6785: Deep Generative Models. Lecture 11: Energy-Based Models](https://www.youtube.com/watch?v=W0rCcPKF4Yc)를 보고 정리한 내용입니다.

## Energy Based Model

### Motivation

확률 분포 p(x)를 표현하는 것은 생성 모델링에서 중요한 도전 과제임.
확률 분포는 두가지 공통된 특징을 가짐.

1. non-negativity: $ p(x) >= 0 $
2. sum-to-one: $ \sum_{x} p(x) = 1 $ or $ \int p(x) dx=1 $

sum-to-one이 아주 중요한 특성인데 전체 volume이 1로 정해져있기때문에 train시에 데이터셋에 대해 likelihood를  maximize하다보면 포함되지 않은 다른 데이터들은 확률이 줄어든다는 것을 의미한다.

먼저 non-negative 함수 $g_{\theta}(x)$ 를 다루는 것은 어렵지 않다. 아무런 신경망에 대해서 제곱이나 exponetial을 취하면 된다.

sum-to-one의 경우 보통은 $ \sum_{x} g_{\theta}(x) != 1 $이기 때문에 $g_{\theta}(x)$는 유효한 확률 분포가 아닐 것이다.

#### solution: $g_{\theta}(x)$의 명시적 정규화

$$ p_{\theta}(x) = \frac{1}{Volume(g_{\theta})} g_{\theta}(x) = \frac{1}{\int g_{\theta}(x) dx} g_{\theta}(x) $$

이전에 배운 모델에 대해서 volume을 분석해보면,

![이전에 배운 모델](https://github.com/currybab/currybab.github.io/assets/7679722/d97b2352-fee0-4a26-aee6-202be6941b9c)

1. autoregressive : 왜 normalized objects의 곱으로 표현할 수 있다는 건지 잘 모르겠음. $ p_{\theta}(x) = \prod p_{\theta}(x_i | x_{<i})$ 니까 특정 시점까지의 확률을 x라두고 그 이후에 구하려는 지점을 y라고 두면, 저런 표현식이 나오나...? 
2. latent variables : gaussian 분포들의 합으로 볼 수 있으므로 mixture of normalized objects로 나타낼 수 있음

하지만 volume 혹은 normalization constant를 분석적으로 계산하기 힘든 경우라면..?

### Definitions

$$ p_{\theta}(x) = \frac{1}{\int exp(f_{\theta}(x)) dx} exp(f_{\theta}(x)) = \frac{1}{Z(\theta)} exp(f_{\theta}(x))$$

$f_{\theta}(x)$는 임의의 어떤 데이터 포인트에서 어떤 score를 가지는 함수이고 이것을 exponential을 취함으로써 non-negative하게 함. 
하지만 이것의 합은 energy based model이기 때문에 sum을 해도 1이 아님.
하지만 명시적으로 전체가질수 있는 스코어를 이제 합해서 나눠줌으로써 normalize하기 때문에 확률 함수로써 동작할 수 있음.

$ Z(\theta) $를 partition function, volume, normalization constant으로 부름.

Why exponential ?

1. 확률의 매우 큰 변동을 포착하고자 합니다. 자연 스케일에서 작업하고자 하는 로그 확률입니다. 그렇지 않으면 매우 비연속적인 ($f_\theta$)가 필요합니다.
2. 많은 보통의 분포(exponential families)가 이런 형태로 표현될 수 있다.
3. 이러한 분포는 통계 물리학에서 비교적 일반적인 가정(최대 엔트로피 이론, 열역학 제2법칙) 하에서 발생합니다. -$f_\theta(x)$가 시스템의 에너지라고 불린다고함.(에너지 기반 모델이라 불리는 이유)
 
에너지 기반 모델의 장점: 원하는 아무 함수 $f_\theta(x)$를 사용할 수 있음.

단점: $ Z(\theta) $를 구하는 것이 매우 힘듬.
- 확률분포함수로부터 Sampling이 힘듬.
- 확률분포함수의 likelihood를 evaluating과 optimizing이 힘듬(학습이 힘듬)
- 현재 형태에서는 feature learning이 없음. (latent variables들을 추가할 수는 있음)

그렇지만 어떤 task들은 $ Z(\theta) $ 없이도 동작한다. 다음강좌(스코어 베이스 모델)인듯 한데 여기서는 $ Z(\theta) $를 사용하지 않고도 EBM을 학습하는 알고리즘을 가르쳐준다고 함.

### Motivating Applications

이상값 탐지의 한 형태 혹은 어떤 종류의 밀도 추정에서 쓰일 수 있음.
![not need volume](https://github.com/currybab/currybab.github.io/assets/7679722/5db4c38f-240a-42ca-8e70-42169bd79f34)

1. 이상값 탐지: 새로운 값이 주어졌을 때 위값을 계산하여 임계값을 초과하면 이상치로 탐지함.
2. 노이즈 제거 및 추정: 최적화 알고리즘을 통해 높은 값을 갖는 값을 찾음. 후에 ising model에서도 예를 든다고 함.

### Conditional Energy-based Models

![Conditional Energy-based Models](https://github.com/currybab/currybab.github.io/assets/7679722/fc6d48c0-1af0-4c75-9f76-380f56b0ea79)

## Representation

### Ising Models

> Ising 모델은 통계 물리학과 컴퓨터 과학에서 중요한 역할을 하는 수학적 모델로, 주로 자성체(magnetic materials)의 상전이(phase transition)를 설명하기 위해 고안되었습니다. 이 모델은 스핀(spin) 시스템을 통해 자성체의 자발적인 자화(magnetization)를 연구하는 데 사용됩니다.

![ising model](https://github.com/currybab/currybab.github.io/assets/7679722/d57e47e1-45a0-4da8-9fe3-c5bfd04c3791)

joint probability distribution의 첫째항은 node, 둘째항은 edge에 관한 것으로 볼 수 있음.

### Restricted Boltzmann Machines

RBM은 latent variable을 가진 energy based model임.

![RBM](https://github.com/currybab/currybab.github.io/assets/7679722/055bffe6-d678-4141-b653-96ae1f48958a)

restricted라고 불리는 이유는 visible-visible과 hidden-hidden 간에 연결이 없기 때문임.
또한 nerual network 모델과 비슷하게 생김.

### Deep Boltzmann Machines

Stacked RBM은 처음으로 working했던 deep generative model임.

![deep RBM](https://github.com/currybab/currybab.github.io/assets/7679722/0dea39f4-e140-4091-a6a0-b41a7e66daf7)

bottom layer v는 pixel value임. 위에 있는 레이어들(h)은 높은 레벨의 feature를 나타냄.(corners, edges,...etc)

![samples](https://github.com/currybab/currybab.github.io/assets/7679722/e0232ac6-88dc-46f0-8694-3b047742fa2a)

## Learning

### Likelihood based learning

일반적으로 energy base model의 likelihood-based learning은 intractable함.
(log likelihood를 계산하는 것조차 어려움.)

1. 최대 우도 학습의 근사치 적용
- Variational inference: optimization based approximations
- MCMC-based 방법들: sampling based approximations (이번 강의에서 다룸)

### Exponential Families

$$ p(x;\theta) = \frac{exp(\theta^{T} f(x))}{Z(\theta)} $$

- energy based model은 exponential familes들과 밀접한 관계가 있음.
- vector $f(x)$는 충분 통계량 벡터라고 부름. 데이터를 특징화하거나 요약함: 동일한 f(x)를 가지는 두 x는 동등.
- Example: 가우시안: $ f(x) = (x, x^2), \theta = (\frac{\mu}{\sigma^2}, \frac{-1}{\sigma^2}) $.
- Exponential, Binomial, Cauchy, Beta, Drichlet, Ising, conditional RBMs and many others 역시 위와 같이 표현할 수 있나봄.

#### Learning 

주어진 데이터셋 D에 대해, 우리는 $\theta$를 최대 우도 학습법을 통해 추정할 것임. 
우리는 log-likelihood 함수가  오목하며(concave) 다음과 같음을 보임.

$$ \frac{1}{|D|} \log p(D;\theta) = \frac{1}{|D|} \sum_{x \in D} \theta^T f(x) - \log Z(\theta) $$

첫번째 항은 $\theta$에 대해 linear하기 때문에 다루기 쉬움. 

$$ \log Z(\theta) = \log \sum_{x} exp(\theta^T f(x)) $$

두번째 항은 위와 같은데 optimize하기 힘들뿐 아니라 evaluate하기도 힘듬.

#### Gradient Based Learning

우리가 energy model을 gradient based 최대 우도 학습법을 통해 학습시킬수 있을까 확인해봄.
첫번째 항은 계산하기 쉬움. 두번째 항을 풀어보면,

![gradient of Z](https://github.com/currybab/currybab.github.io/assets/7679722/f5f40558-d64a-476c-b0dd-8ba7f9d81c10)

전체 데이터에 대한 분포에 대한 f(x)의 기댓값이 됨.

#### Moment Matching

위의 과정을 통해서 우리는 log-likelihood 함수의 gradient가 아래와 같이 됨을 보임.

![moment matching](https://github.com/currybab/currybab.github.io/assets/7679722/bb9adf4b-ac44-4d89-9471-babf124267cf)

이는 데이터셋(D)에 대한 f(x)의 경험적 평균과 전체 데이터 분포 p에 대한 f(x)의 기대값 사이의 차이를 계산하는 것임. 
이 값이 0이 되었을 때 내 모델이 optimize가 되었다고 생각할 수 있음.
하지만 여전히 오른쪽 항이 계산하기 쉽지 않기 때문에 여전히 문제가 있음.

샘플링 기반 근사법: 분포 p로부터 샘플 x를 근사하여 생성하고, 샘플을 사용하여 몬테카를로 방법으로 $E_{x~p}[f(x)]$를 근사하여 분포를 학습.

### Markov Chain Monte Carlo(MCMC)

확률 분포 p로부터 x를 생성할때 Markov Chain을 시뮬레이션함.

#### Definition of Markov Chain

![definition of markov chain](https://github.com/currybab/currybab.github.io/assets/7679722/7e73ec89-2520-431f-be78-a2cda2c3a9e5)

#### Stationary Distribution

![stationary distribution](https://github.com/currybab/currybab.github.io/assets/7679722/fededa0f-7748-44cb-84b5-10e7d72c1e0b)

- detailed balance(상세 균형): stationary distribution $\pi$가 존재하기 위한 충분조건.
$$ \pi(i) T(j|i) = \pi(j) T(i|j) $$
상태 (i)에서 상태 (j)로 전이하는 확률과 상태 (j)에서 상태 (i)로 전이하는 확률이 정지 분포에서 균형을 이루는 것을 의미함.
상세 조건의 중요한 점은, 이 조건이 성립하면 정지 분포가 존재할 뿐만 아니라, 그 정지 분포를 쉽게 계산할 수 있다는 것임. 
따라서 많은 확률 모델에서 상세균형 조건을 이용하여 정지 분포를 구하는 데 사용됨.

#### Method

- 상태가 모델 내 모든 변수들에 대한 특정 값들의 조합으로 정의함.
- 정지 분포가 모델 확률과 동일함.

1. 전이 연산자 (T) 정의:
    - MCMC 알고리즘은 마르코프 연쇄를 정의하는 전이 연산자 (T)를 지정합니다. 이 연산자는 현재 상태에서 다음 상태로의 전이 확률을 결정합니다.
    - 초기 변수 할당 (x_0)를 설정합니다. 이는 마르코프 연쇄를 시작할 초기 상태를 의미합니다.

2. Burn-in 단계:
    - 마르코프 연쇄를 x_0에서 시작하여 B번의 burn-in 단계를 실행합니다.
    - Burn-in 단계는 초기 상태의 영향을 줄이기 위해 사용됩니다. 이 단계에서는 수렴하지 않은 초기 상태를 제거하고, 연쇄가 목표 분포에 더 가까워지도록 합니다.

3. Sampling 단계:
    - Burn-in 단계 이후, 마르코프 연쇄를 N번의 샘플링 단계 동안 실행하고, 방문한 모든 상태를 수집합니다.
    - 이 단계에서 수집된 상태들은 목표 분포 p로부터의 샘플을 형성합니다.

B가 충분히 크다고 가정하면, Burn-in 단계 이후의 상태들은 초기 상태의 영향을 거의 받지 않게 됨.
따라서 샘플링 단계에서 수집된 상태들은 목표 확률 분포(p)로부터의 샘플로 간주될 수 있음.
 
#### Metropolis-Hastings 적용

두 구성 요소를 가짐.
- 전이 커널 (Q(x'|x)):
    - 전이 커널은 현재 상태 (x)에서 다음 상태 (x')로의 전이 확률을 정의합니다.
    - 이 전이 커널은 사용자가 지정하며, 일반적으로 간단한 형태를 가집니다. 예를 들어, $x + \text{noise}$와 같이 현재 상태에 약간의 노이즈를 추가하는 방식이 있을 수 있습니다.

- 수락 확률 (A(x'|x)):
    - 전이 커널 (Q)에 의해 제안된 이동을 수락할 확률을 정의합니다.
    - 수락 확률은 다음과 같이 정의됩니다:
    $$A(x\prime| x) = \min \left( 1, \frac{P(x\prime) Q(x | x\prime)}{P(x) Q(x\prime | x)} \right)$$
    - 여기서 ( P(x) )는 목표 분포의 확률 밀도 함수입니다.
    - 중요한 점은 비율 ($\frac{P(x')}{P(x)}$)가 normalize constant $Z(\theta)$를 알 필요가 없다는 것입니다. 이는 목표 분포가 정상화 상수를 모를 때에도 샘플링을 할 수 있게 해줍니다.
    - 이 수락 확률은 우리가 분포에서 더 높은 확률을 가지는 지점으로 이동하도록 장려합니다. 예를 들어, ( Q )가 균일 분포일 때의 수락 확률 공식을 고려해보면 이해할 수 있습니다.
    - 만약 ( Q )가 낮은 확률 영역으로의 이동을 제안하면, 우리는 일정한 비율로 그 이동을 수락하게 됩니다.
 

MH 알고리즘은 다음 단계를 통해 작동합니다:
1. 새로운 상태 (x') 제안:
    - 마르코프 연쇄의 각 단계에서 전이 커널 Q에 따라 새로운 상태 x'를 제안합니다.
2. 수락 여부 결정:
    - 제안된 새로운 상태 x'를 수락할 확률 $\alpha$를 계산합니다.
    - 확률 $\alpha$로 제안된 상태 x'를 수락하고, 그렇지 않으면 현재 상태에 머무릅니다. 즉, 확률 $1 - \alpha$로 현재 상태를 유지합니다.
 
이 과정을 반복하면, 마르코프 연쇄는 목표 분포 ( P(x) )로 수렴하게 됩니다. 이를 통해 우리는 목표 분포로부터 샘플을 생성할 수 있습니다.

### (Persistent) Contrastive Divergence

로그-우도(log-likelihood)를 최대화하기 위해 그래디언트 디센트(gradient descent)를 사용하고, MCMC를 통해 샘플을 얻어 그래디언트를 계산하는 방법.

1. MCMC 체인을 실행하여 ($p(x; \theta_t)$)로부터 샘플을 얻음
2. MCMC 샘플을 사용하여 그래디언트($\nabla \log p(\text{data}; \theta_t)$)를 계산
3. 그래디언트 스텝을 통해 파라미터 업데이트 $\theta_{t+1} = \theta_t + \alpha \cdot \nabla \log p(\text{data}; \theta_t)$(여기서 a는 학습률)

Persistent CD는 CD 알고리즘의 변형으로, MCMC 체인을 매번 재시작하지 않고 이전 체인을 계속 사용함. 이는 다음과 같은 이유로 효과적임.
- $\theta_t$와 $\theta_{t+1}$는 매우 유사하므로, 이전 체인 $p(x; \theta_t)$에서 얻은 샘플은 새로운 체인 $p(x; \theta_{t+1})$에서도 좋은 샘플이 됨.
- 새로운 체인을 이전 샘플에서 초기화하고, $\theta_{t+1}$에서 몇 번의 스텝만 실행하여 샘플을 얻음. 이는 체인의 수렴 속도를 높이고 계산 효율성을 향상시킴.

## EBM 정리

$$ p_\theta(x)= \frac{1}{\int\exp(f_\theta(x))} \exp(f_\theta(x)) = \frac{1}{Z(\theta)} \exp(f_\theta(x)) $$

### 장점 (Pros)
- 임의의 함수 $f_\theta(x)$를 사용할 수 있음:
    - 모델 내에서 거의 모든 형태의 함수 $f_\theta(x)$를 사용할 수 있습니다. 이는 모델의 유연성을 높여줍니다.
- 다른 model family와 결합 가능:
    - 이 모델은 다른 확률 모델과 결합하여 더 복잡한 모델을 만들 수 있습니다. 예를 들어, 심층 신경망(deep neural networks)과 결합할 수 있습니다.
- 다양한 모델을 통합하는 프레임워크:
    - 많은 기존 모델들을 이 통일된 프레임워크 내에서 일반화할 수 있습니다. 이는 다양한 모델들을 하나의 일관된 방식으로 이해하고 분석할 수 있게 해줍니다.

### 단점 (Cons)
- 샘플링이 일반적으로 어렵다:
    - 모델로부터 샘플을 생성하는 것이 일반적으로 계산적으로 어려울 수 있습니다. 특히, $Z(\theta)$를 계산하는 것이 어려운 경우가 많습니다.
- 우도 평가(학습)가 어렵다:
    - 모델의 우도를 평가하는 것이 일반적으로 어렵습니다. 이는 모델 학습 과정에서 계산적으로 비효율적일 수 있습니다.
- 잠재 변수 추론이 어려움:
    - 모델 내의 잠재 변수(latent variable)를 추론하는 것도 어려운 문제입니다. 이는 모델을 학습하고 예측하는 과정에서 추가적인 복잡성을 가져옵니다.
 
