+++
title = "Pmpp Lec07 Profiling 요약"
date = "2025-11-11T22:42:04+09:00"

#
# description is optional
#
# description = "An optional description for SEO. If not provided, an automatically created summary will be used."

tags = ["pmpp"]
+++

source: [Lecture 07 - Profiling](https://www.youtube.com/watch?v=zHY7iF_2RyU)

### Today

- Profiling
  - bottleneck을 찾는 방법
- Running example
  - Vector addition

## NVProf

- profiler api를 추가함 `#include <cuda_profiler_api.h>`
- 메인 함수 마지막에 `cudaProfilerStop()`을 호출함
  - 애플리케이션이 종료되기전에 수집한 모든 정보가 저장됨.
- 프로파일러가 타이머 정보를 알수 있기때문에 이전처럼 굳이 따로 시간을 측정할 필요 없음
- 실행법 `nvprof ./{binary}`
  - GPU활동과 API 호출(CPU에서 일어나는 일)로 구분됨.
  - 각 활동에 대해 얼마나 많이 호출되는지 얼마나 걸리는지 알수 있음.
  - 전체 실행시간중 차지하는 시간도 나옴.
- 프로파일링 정보를 visual profiler를 사용할 수 있도록 저장할 수도 있음.
  - `nvprof -o {output_file} ./{binary}`
- 점유율, 캐시, 메모리 대역폭 등과 같은 다른 정보들이 필요하면 `nvprof -m all -o {output_file} ./{binary}`
  - all 대신 특정한 지표들만 선택하는것도 가능함.
  - 실제로 커널을 여러번 실행하고 매번 다른 프로파일링 지표를 측정함.

## NVVP

- `nvvp {output_file}`
- 새 세션을 만들어서 비주얼 프로파일러 자체를 사용하여 프로파일링 할 수도 있고
- nvprof를 통해 생성한 프로파일링 지표를 가져올 수도 있음.
- 세가지 지표 분석을 할 수 있음
  - memory bandwidth : 가장 bottleneck이 될 확률이 높음
  - compute 
  - lantency
