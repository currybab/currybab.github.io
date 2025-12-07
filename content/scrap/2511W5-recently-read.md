+++
title = "25년 11월 5주차 최근 본 것들"
date = "2025-11-26T22:52:48+09:00"

#
# description is optional
#
# description = "An optional description for SEO. If not provided, an automatically created summary will be used."

+++

- [Learning CUDA by optimizing matrix-vector multiplication (SGEMV) for cuBLAS-like performance - A worklog](https://maharshi.bearblog.dev/optimizing-sgemv-cuda/)
  - gemv 커널에서 있는 현상들을 잘 분석한 글
- [Making GPUs Actually Fast: A Deep Dive into Training Performance](https://www.youtube.com/watch?v=pHqcHzxx6I8)
  - Jane street 에서 쓰는 간단한 성능 최적화 방법들을 정리한 영상.
- [I Trained an LLM to Dream. It Remembers Everything.](https://www.youtube.com/watch?v=YA3hAGtfMs4)
  - 꿈을 통해 기억한다는 논문을 보고 영감을 받아서 context를 유지하는 모델을 만드는 영상.
  - 사실 context 유지가 정말 되는것보다 forgetting이 더 많을것 같긴한데 개인 연구자로써 저렇게 해볼 수 있다는 영감을 받았다. 
- [INTELLECT-3: A 100B+ MoE trained with large-scale RL](https://www.primeintellect.ai/blog/intellect-3)
  - glm-4.5-air base 모델을 사후 학습(sft + rl)을 통해 좋은 성능을 내는 모델을 만든 prime intellect 팀
  - 사전학습이 물론 중요하지만 현실적으로 모든 팀이 굳이 할 필요는 없다고 생각한다. 어차피 규모의 싸움이기 때문에 이미 대형랩에서 잘 공개해준 base 모델로 충분할 거라는 생각을 한다.
  - 이제 더 중요한 것은 각 사용 환경에 따른 사후학습이 더 중요한 순간이 온 것 같다고 생각한다. 너무 경험해보고 싶은 일이고 부럽다.
