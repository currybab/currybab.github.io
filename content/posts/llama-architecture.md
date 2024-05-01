+++
title = 'LLaMA Architecture 정리'
date = 2024-04-28T00:02:57+09:00
+++

### Reference

- [LLaMA explained: KV-Cache, Rotary Positional Embedding, RMS Norm, Grouped Query Attention, SwiGLU](https://www.youtube.com/watch?v=Mn_9W1nCFLo)
- [pytorch-llama-notes](https://github.com/hkproj/pytorch-llama-notes/)
- [Transformer Engine Doc](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/te_llama/tutorial_accelerate_hf_llama_with_te.html)

해당 글은 위의 유튜브를 보면서 추가적으로 궁금했던 부분들을 더 찾아보고 붙여 놓은 글입니다.

## LLaMA vs GPT2

![transformer_vs_llama](https://github.com/currybab/currybab.github.io/assets/7679722/78fa6c89-072b-42ba-a1eb-97d2dd7aa69e)

최초의 트랜스포머 아키텍처와 비교하면 사실 normalization이 각 블록의 앞으로 온점이 추가적으로 다르다.
원래는 뒤에 위치했었는데 이는 아마 Layer Normalization 이전에 Batch Normalization이 주로 레이어 뒤쪽에 위치했기 때문이 아닐까...
어쨌든 GPT 2부터는 레이어 앞쪽으로 위치한다. layer normalization의 위치를 다룬 논문([On Layer Normalization in the Transformer Architecture](https://arxiv.org/pdf/2002.04745.pdf))이 GPT 2 이후에 나온 것은 조금 신기하긴 하다. 
그 외에도 LayerNorm이 RMSNorm으로 대체 되었다.

Postional Encoding이 임베딩 레이어 이후에 전체적으로 추가되던 것이 query와 key 부분에 추가되는 Rotary Positional Encoding으로 대체되었다. 또한 self attention 부분이 multihead attention(MHA)에서 grouped query attention(GQA)으로 변경되었다. (GQA의 경우 34B와 70B 모델만 적용) 마지막으로 feed forward layer에서 relu 활성화 함수를 적용하던 부분이 SwiGLU 이후에 feed forward layer를 진행하는 것으로 변경되었다. 

## RMSNorm

기존에는 Layer Normalization을 활용하였다.
$$ y = \frac{x-E[x]}{\sqrt{Var[x] + \epsilon}} * \gamma + \beta $$
Batch Normalization의 수식과 같은데 Batch Normalization는 feature별로 normalize를 진행하는 것이고 Layer Normalization은 data별로 normalize를 진행한다.

Root Mean Square Layer Normalization에서는 LayerNorm의 성공이 invariance의 re-centering과 re-scaling에서 왔다고 하는데 해당 논문의 저자들은 re-scaling의 영향이 더 크다고 주장한다.
그래서 re-scaling에만 집중한 normalize 방법으로 RMSNorm을 제시한다.

$$ \bar{a}_i = \frac{a_i}{RMS(\textbf{a})} \gamma_i, \quad where \enspace RMS(\textbf{a}) = \sqrt{\frac{1}{n} \sum_{i=1}^{n} {a_i}^2}  $$

LayerNorm에 비해 연산량이 적고 실제로 잘 동작했기 때문에 RMSNorm으로 대체되었다.

## Rotary Positional Encoding

"Attention is all you need" 논문에서 제시됬던 방법은 absolute positional encoding이다. 해당 방법은 token embedding에 절대적인 위치에 대한 값을 추가하였다.
추후에 "Self-Attention with relative positional representations" 논문에서 relative positional encoding이 제시되었다. 해당 방법에서는 query와 key를 내적할 때 두 토큰간의 거리를 계산해서 넣어주었다.

RoPE는 "ROFORMER: ENHANCED TRANSFORMER WITH ROTARY POSTIONAL ENCODING" 논문에서 방법이 제시되었다. 
해당 논문은 내적의 특성을 통해서 어떻게 relative postion을 자동으로(?) 넣어줄 수 있을까에 대해서부터 시작한 방법이다.
그 결과로 $e^{i\theta}$를 query와 key부분에 곱하여 내적을 하면 relative positional encoding과 같은 효과를 누릴수 있다고 제시하였다.

추가적으로 long-term decay가 관찰 되었는데 이는 직관적으로 거리가 먼 토큰이면 서로 연관이 있을 가능성이 떨어진다는 것과 일치한다.

## self attention review

![self-attention full](https://github.com/currybab/currybab.github.io/assets/7679722/376f69b6-160c-4790-81dc-82eb10c37274)

## KV Cache

inference 시에 우리는 마지막 나오는 토큰만 알면 되는데 기존 구조에서는 모든 토큰의 값을 구했다. 
이를 효율적으로 해결하는 방법이 KV cache이다.
$Attention(Q,K,V)$를 계산할 때 마지막 값만 우리는 관심이 있다.
i번째 토큰 $T_i$가 query 값으로 들어올때 key 값으로 $T_1$부터 $T_i$ 부분만 계산하면 되는것이다.
그 다음부분은 어차피 mask에 의해서 영향을 받지 않고 Output의 $i-1$까지는 이미 알고 있는 부분이기 때문에 계산할 필요가 없다.
따라서 해당 행은 $1 \times i$ 크기를 갖게 되고 마찬가지로 value 역시도 $T_1$부터 $T_i$까지만 갖고 있으면 된다.($i \times d$ 크기)
이렇게 Query 값만 변화하는 채로 Key, Value를 캐싱해서 계산하면 되기 때문에 이름이 KV Cache인듯 하다.


## Grouped Multi-Query Attention

GPU가 계산이 memory보다 빠르기 때문에 memory의 bandwidth가 bottleneck이 된다.(약 40배)
따라서 연산횟수를 줄이는것 만큼이나 메모리 접근 및 이동을 최소화하는 것이 중요하다.

### Vanilla MHA

- 연산수: $ O(bnd^2) $
- 메모리: $ O(bnd + bhn^2 + d^2) $
- 메모리 / 연산수: $ O(\frac{1}{d} + \frac{1}{bn}) $
- 1보다 작으므로 이경우에는 문제가 되지 않음.

### MHA with KV Cache

- 연산수: $ O(bnd^2) $
- 메모리: $ O(bn^2d + nd^2) $
- 메모리 / 연산수: $ O(\frac{n}{d} + \frac{1}{b})$
- $\frac{n}{d}$가 1보다 클 수 있으므로 문제가 될 수 있음.

### MQA with KV Cache

- $K$, $V$가 각각 h개의 차원을 가져서 나눠졌던 부분을 삭제함.
- 서로다른 Query Header가 같은 Key, Value값을 보고 있음을 의미함.
- 연산수: $ O(bnd^2) $
- 메모리: $ O(bnd + bn^2k + nd^2) $
- 메모리 / 연산수: $ O(\frac{1}{d} + \frac{n}{dh} +\frac{1}{b})$
- 적은 성능 저하를 겪고 퍼포먼스를 얻음.

### GQA

MHA와 MQA 사이에서 quality와 speed 사이의 적당한 balance를 찾은 방법이라고 보면 됨.
![mha_gqa_mqa](https://github.com/currybab/currybab.github.io/assets/7679722/1d707a56-58dd-4781-935c-7e54eaaf0d8b)

## SwiGLU Activation Function

기존 Transformer에서 사용되는 ReLU 함수 대신 SwiGLU 활성화 함수를 사용했다.SwiGLU가 잘 작동하지만 효과적인 이유는 완전히 파악된것은 아니라고 한다. 실제로 후에 나온 Gemma에서는 GEGLU를 사용했다.
