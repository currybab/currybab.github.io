+++
title = 'Autoregressive Models'
date = 2024-05-05T12:25:22+09:00
draft = true
+++

[Cornell CS 6785: Deep Generative Models. Lecture 3: Autoregressive Models](https://www.youtube.com/watch?v=Y3cJFaM8w2w&list=PL2UML_KCiC0UPzjW9BjO-IW6dqliu9O4B&index=3)

## The Task of Generation Modeling

![generative modeling](https://github.com/currybab/currybab.github.io/assets/7679722/0ee6f626-f170-4d6d-b55b-d5bef9c638c4)

(강아지의) 이미지들 $x$에 대해 확률 분포 $p(x)$로부터 다음과 같은 목표를 달성할 수 있어야함.
- Generation: $p(x)$에서 추출한 sample $x_new$가 강아지 같아야 함.
- Representation Learning: 이미지들이 갖고 있는 공통적인 특징에 대해 배울 수 있어야함.
- Density Estimation: $x$가 강아지 같을수록 $p(x)$가 높은 값을 가져야하며 아니라면 낮은 값을 가져야함.

1. 첫번째 스텝: 모델링 스테이지 - define model family - How to represent $p(x)$
2. 두번째 스텝: How to learn it

## Basic Autoregressive Models

이 강의에서 주요 주제는 modeling handwrite digits임(mnist)
- each image: 28 * 28 = 784 pixels
- each pixel: 0 or 1
- GOAL: Learn probabilty distribution $p(x)=p(x_1,...,x_{784})$ over $x \in \{0,1\}^{784}$ such that when $x \sim p(x)$, x looks like a digit

### Recall: Neural Models for Classification

input features $X\in\{0,1\}^n$ 에 대해서 binary classification($Y\in\{0,1\}$)을 하는 상황을 고려해보자.
- 이산 모델을 파라미터화한 모델을 다음과 같다고 가정함. $ p(Y=1|\text{x};\alpha) = f(\text{x}; \alpha)$
- 로지스틱 회귀가 $ f(\text{x};\alpha)$의 한 예가 될 수 있음.

$z(\alpha,\text{x}) = \alpha_0 + \sum_{i=1}^{n} \alpha_i \text{x}_i$라고 하면
$ \sigma(z) =1/(1+e^{-z}) $ 일때, $ p_{logit}(Y=1|\text{x};\alpha) = \sigma(z(\alpha, \text{x}))$ 이 됨. 

- MLP에서는 $\text{h}(A,\text{b},\text{x})$를 input feature들의 non-linear transformation으로 보고 다음과 같이 적용할 수 있음. $p_{MLP}(Y=1|\text{x};\alpha, A, \text{b}) = \sigma(\alpha_0 + \sum_{i=1}^{h} \alpha_i h_i)$
    - flexibility가 증가함
    - 파라미터가 증가함

### An Autoregressive Model From Logistic Regression

logistic regression으로 부터 AR model을 construct해보려고함. 가장 간단한 공식부터 시작함. 체인룰에 의해서,
$$ p(x_1, ..., x_{784}) = p(x_1)p(x_2|x_1)p(x_3|x_1,x_2)...p(x_n|x_1,...,x_{n-1}) $$
위에서 얻어진 공식을 logistic regression 모델을 통해서 단순화 시킬수 있음.
$$ p(x_1, ..., x_{784}) = p_{CPT}(x_1; \alpha^1)p_{logit}(x_2|x_1; \alpha^2)p_{logit}(x_3|x_1,x_2; \alpha^3)...p_{logit}(x_n|x_1,...,x_{n-1};\alpha^n) $$
각각의 $p_n$을 입력값 $x_1$부터 $x_{n-1}$까지 활용하는 logstic regression을 적용함(modeling assumption). 이를 좀더 직접적으로 표현하면
$$ p_{CPT}(X_1 = 1; \alpha^1) = \alpha ^ 1,  p(X_1 = 0) = 1 - \alpha^1 $$
$$ p_{logit}(X_2 = 1 | x_1 ; \alpha^2) = \sigma(\alpha_0^2 + \alpha_1^2 x_1)$$
$$ p_{logit}(X_3 = 1 | x_1, x_2 ; \alpha^3) = \sigma(\alpha_0^3 + \alpha_1^3 x_1 + \alpha_2^3 x_2)$$
와 같이 볼수 있음. 각각의 $p_n$이 n개의 parameter를 갖게 되고 이는 총 $ O(n^2)$에 해당하므로 기존의 exponential을 고려했을때 매우 줄었음을 알 수 있음.

우리의 모델에서 $p_{logit} (x_i|x_1,...,x_{i-1}; \alpha^i)$ 각 항은 conditional Bernoulli임.
이를 나타내기 위해 $\hat{x}$를 사용하면,
$$ \hat{x}_i = p(X_i=1|x_1,..., x_{i-1}; \alpha^i) = p(X_i=1|x_{<i}; \alpha^i) = \sigma(\alpha_0^i + \sum_{j=1}^{i-1} \alpha_j^i x_j)$$

우리가 $ \hat{x}_i$를 생각하는 방법은
1. pixel i의 확률
2. pixel이 1~i-1까지 주어졌을때, 다음 pixel 예측
3. partial input이 주어졌을때, $x_i$의 reconstruction.

### Fully Visible Sigmoid Belief Network (FVSBN)

우리가 지금까지 한것을 FVSBN이라고 부름.
![fvsbn](https://github.com/currybab/currybab.github.io/assets/7679722/6fb11027-9c3b-4c16-a466-161728bb751b)

- density 구하기 (= $p(x_1,...,x_{784})$)
    - 모든 condition factor를 곱함.
    - $p(X_1=0, X_2=1, X_3=1, X_4=0) = (1-\hat{x_1}) \hat{x_2} \hat{x_3} (1-\hat{x_4})$ 각각의 hat은 위의 공식을 통해 구할 수 있음.

- sample 추출하기
    - $ \bar{x_1} \sim p(x_1)$ 샘플링 (np.random.choice([1,0], p=[$\hat{x_1}$, 1-$\hat{x_1}$]))
    - $ \bar{x_2} \sim p(x_2 | x_1=\bar{x_1})$ 샘플링 
    - $ \bar{x_3} \sim p(x_3 | x_1=\bar{x_1}, x_2=\bar{x_2})$ 샘플링 
    - parameter 수 = 1 + 2 + 3 + ... + n <= n^2 / 2

### From Logistic Regression to MLP

![mlp](https://github.com/currybab/currybab.github.io/assets/7679722/b94d0940-7bfb-4fd7-bedb-ab80590181b1)

#### weight sharing

![weight sharing](https://github.com/currybab/currybab.github.io/assets/7679722/16ce5672-83cd-4438-b01c-e854c2ccf231)

### Masked Autoencoder for Distribution Estimation (MADE)

오토인코더를 활용하여 i보다큰 input weight를 masking함으로써 ar 모델을 만들수 있다.
![made](https://github.com/currybab/currybab.github.io/assets/7679722/37a653c5-6900-475f-a40c-76d01f11536c)

## Recurrent Neural Network as Autorgressive Models

![RNN](https://github.com/currybab/currybab.github.io/assets/7679722/5bcfc150-91d3-4e81-9711-91b3cfc4909f)

## Modern Autoregressive Models

WaveNet, PixelCNN