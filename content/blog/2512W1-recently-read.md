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
- [Implementing a fast Tensor Core matmul on the Ada Architecture](https://www.spatters.ca/mma-matmul)
- [Efficient GEMM in CUDA](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md)
- [Continuous batching](https://huggingface.co/blog/continuous_batching)
- [I wrote a kernel that makes sparse LLMs faster and smaller on consumer GPUs even at low sparsity.](https://www.reddit.com/r/LocalLLaMA/comments/1pbag5i/i_wrote_a_kernel_that_makes_sparse_llms_faster/)
- [Notes About Nvidia GPU Shared Memory Banks](https://feldmann.nyc/blog/smem-microbenchmarks)
