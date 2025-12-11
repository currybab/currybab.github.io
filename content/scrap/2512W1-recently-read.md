+++
title = "25년 12월 1주차 최근 본 것들"
date = "2025-12-01T19:23:38+09:00"

#
# description is optional
#
# description = "An optional description for SEO. If not provided, an automatically created summary will be used."

tags = []
+++

- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)
    - 행렬곱 커널에 대해서 단계별로 성능 최적화하는 글. tensor core를 활용하지 않고 cuda core만을 사용하여 성능을 최적화하는 방법을 설명함.
    - Warp level에서도 타일링을 적용할 수 있다는 아이디어가 나에게는 신선했음.
- [NVIDIA Tensor Core Evolution: From Volta To Blackwell](https://newsletter.semianalysis.com/p/nvidia-tensor-core-evolution-from-volta-to-blackwell)
    - 볼타부터 blackwell아키텍쳐까지의 텐서코어의 진화에 대해서 설명한 글
    - 사실 텐서코어의 진화가 gpu 아키텍쳐 진화의 핵심인 것 같다.
    - 요새 TPU가 짱이니 GPU가 짱이니 말이 많은데, 특히 범용성과 성능을 기준으로 떠도는 말이 많은데.. 텐서코어가 결국 핵심이고 텐서코어는 matrix multiplication을 최적화하려는 수단이기 때문에 이렇게 발전이 되었다면 GPU도 이미 행렬곱에 최적화된 아키텍쳐가 되어있는 것 같다.
    - 그동안 CUDA 코어만을 활용해서 leetgpu 문제를 풀고 그랬는데 gpumode의 nvfp4 해커톤을 참여하다보니 다양한 자료를 보고 알게 되는 것 같다.
