+++
title = "25년 12월 3주차 최근 본 것들"
date = "2025-12-15T21:09:20+09:00"

#
# description is optional
#
# description = "An optional description for SEO. If not provided, an automatically created summary will be used."

tags = []
+++

- [AI Kernel Generation: What's working, what's not, what's next – Natalie Serrino, Gimlet Labs](https://youtu.be/6guQG_tGt0o?si=1w2z0I5HF0Ibm_x5)
    - AI agent를 통해 torch에서 gpu kernel 최적화를 다룸.
    - 아직은 해당 커널을 열심히 연구한 사람들에게 밀리는 수준이지만, 이미 개발된 최적화를 다루는건 잘한다고 한다.
    - 우리나라도 비싼 GPU 많이 들어온다는데 GPU 커널 최적화가 많이 중요해질 것 같다. 궁금하다.
- [Code World Model: Building World Models for Computation – Jacob Kahn, FAIR Meta](https://www.youtube.com/watch?v=sYgE4ppDFOQ)
    - CWM에 대한 개념 및 접근법을 설명한 영상
    - AI 코드 서비스에 내장 되었을 때의 결과가 궁금하다.
- [Stanford CS230 | Autumn 2025 | Lecture 9: Career Advice in AI](https://www.youtube.com/watch?v=AuZoDsNmG_s)
    - 이 정신 없는 AI 시대에서 어떤 마인드를 갖고 일을 해야 할지에 대해 방향을 잡아주는 좋은 영상.
- [Notes About Nvidia GPU Shared Memory Banks](https://feldmann.nyc/blog/smem-microbenchmarks)
    - Shared Memory Bank에 대한 좋은 설명글
- [Lecture 86: Getting Started with CuTe DSL](https://www.youtube.com/watch?v=9-dfte_N3yk)
    - CuTe DSL의 기본적인 사용법에 대한 영상
    - nvfp4 해커톤을 참여하고 있는데 첫 과제는 cuda cpp로 했었는데 이제는 CuTe DSL로 넘어가야할 것 같아서 시청 했음.
- [Mini-SGLang: Efficient Inference Engine in a Nutshell](https://lmsys.org/blog/2025-12-17-minisgl/)
    - 5000줄로만 구현된 SGLang의 경량화된 구현체로 Tensor Parallelism, Overlap Scheduling, Chunked Prefill, Radix Cache, JIT CUDA kernel 등이 구현되어 있다고 한다.
    - 안그래도 vllm과 SGLang 구현에 대해 곧 뜯어보고 싶다고 생각했는데 너무 좋은 플젝이 적절한 타이밍에 나와 감사하다.
