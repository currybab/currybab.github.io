+++
title = "25년 11월 4주차 최근 본 것들"
date = "2025-11-18T00:45:11+09:00"

#
# description is optional
#
# description = "An optional description for SEO. If not provided, an automatically created summary will be used."

+++

- [Lecture 84: Numerics and AI](https://www.youtube.com/watch?v=ua2NhlenIKo)
  - 여러 quantization 관련 고찰에 대한 내용. 아직 잘 다루지는 못하지만 nvfp4 해커톤 때문에 편하게 들을 수 있었던 것 같다.
- [The 1 Billion Token Challenge: Finding the Perfect Pre-training Mix](https://huggingface.co/blog/codelion/optimal-dataset-mixing)
  - 50% finePDFs: 30% DCLM-baseline: 20% FineWeb-Edu
  - 여러 어닐링 전략(데이터셋간의 전환시 사용함)을 사용하는 것보다 그냥 정적 혼합이 낫다.
  - 커리큘럼 학습보다 그냥 정적 혼합이 낫다.
- [Beyond Quantization: Bringing Sparse Inference to PyTorch](https://pytorch.org/blog/beyond-quantization-bringing-sparse-inference-to-pytorch/)
  - 위에서 언급한 "Numerics and AI"에서 끝에 질문중에 sparse에 대한 Q&A가 있었던 것으로 기억한다. 아직 메인분야는 아니고 아무래도 sparse가 단순한 quantization이 아닌 좀 더 복잡한 접근이라는 점이 사람들이 접근하기 힘든 영역이라고 했었던 것으로 기억한다.
  - 처음에는 이해가 가지 않았었는데 이게 relu연산이랑 연관을 지어 말하니 좀 이해가 되는 것 같기도 하다.
  - 아마 relufication이 말했던 방법 이였던 것 같다.
  - 학습 없이도 CETT나 CATS 같은 방법으로 가능하다는 점이 흥미롭긴하다.
  - 최근에도 qat같은 식으로 quantization이 발전하는 것으로 볼때 sparse를 더 효율적으로 다루게 되면 relufication도 당연히 받아들여지지 않을까 싶기도하다.
- 
