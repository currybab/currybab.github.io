+++
title = "Pmpp Lec08 Convolution 요약"
date = "2025-11-13T21:18:33+09:00"

#
# description is optional
#
# description = "An optional description for SEO. If not provided, an automatically created summary will be used."

tags = ["pmpp"]
+++

source: [Lecture 08 - Convolution](https://www.youtube.com/watch?v=xEVyTZG1wlk)

### Today

- Parallel pattern: convolution
- New feature: constant memory

## Convolution

- 어떠한 인풋이 있을 때 컨볼루션의 출력은 출력의 모든 요소가 해당 입력 요소와 주변 입력의 가중합이 됨.
- blur는 컨볼루션 연산의 특수한 경우였음.(가중치가 같은 경우)
- 각 입력 요소에 대해 다른 가중치를 가질 수 있으며 convolution mask로 결정됨.(합성곱 커널이라고 부르는 경우도 있음)

### Application of convolution

- 신호 처리, 이미지 처리, 비디오 처리 등등에 많이 사용됨.
- 신호나 픽셀을 변환하는데 더 바람직한 값으로 변경하는데 사용됨
  - Gaussian blur, sharpening, edge detection 등등
  - 변환은 마스크의 가중치에 따라 달라짐.
- 2D를 예제로 사용하지만 1D나 3D로 확장할 수 있음.

## Convolution 병렬화

- 병렬화 접근 법: 출력 요소당 하나의 스레드를 갖는 것, 인접한 입력 요소들과 마스크를 통해 결과를 계산함.
- 마스크 저장법
  - 관찰
    - 이 마스크는 일반적으로 작음. 
    - 모든 스레드에 대해 마스크는 동일함.
    - 실행 내내 마스크가 변하지는 않음.
  - 상수 메모리에 저장하여 빠른 액세스를 제공함으로써 최적화할 수 있음.

## 상수 메모리 사용법

- `__constant__ float mask_c[MASK_DIM][MASK_DIM];`로 global varibale 선언을 함
  - 초기화 하는 방법
    - GPU 실행 중에는 쓸 수 없음.
    - CPU에서 GPU 메모리로 복사함. `cudaMemcpyToSymbol(dest, src, size);`
- 64KB만 할당 할 수 있음.
  - 다른 input도 constant이지만 constant memory에 넣기에는 너무 큼.

### 상수 메모리의 동기

- 데이터가 일정할 때 더 효율적인 캐시를 구성하기가 쉬움.
  - 캐시에 변경사항이나 덮어쓰기를 지원하지 않아도 됨. (비싼 작업들임)
  - 쓰기 가능한 캐시가 있으면 일관성 프로토콜을 제공해서 가지고 있는 데이터가 서로 다른 캐시에서 같은 값을 제공하도록 해야함.
- 크기가 작음. 
  - 캐시의 미스율이 작음

## 합성곱에서의 데이터 재사용 

- 같은 블록의 스레드는 동일한 입력 요소 중 일부를 로드함.
  - 특히 다음 스레드는 거의 중복된 요소를 로드함.
- 각 스레드가 입력 요소 하나를 공유 메모리에 로드하여 다른 스레드가 접근할 수 있도록 함. (Tiling)
  - challenge: 이 경우에는 입력 타일이 실제로 출력 타일 보다 큼.
    - input tile dimension = output tile dimension + 2 * mask radius
  - solution: 입력 크기만큼 많은 스레드를 사용하여 로드하고 계산할때 이 스레드중 일부만을 사용함.
    - 그러면 입력시에 모든 타일이 활성화 됨.
    - 입력 타일과 출력 타일 사이의 크기 차이가 있을 때 쓸 수 있음.

## 메모리 비율 계산

- M*M mask를 고려
- 타일링 없이: 
  - 모든 스레드가 m^2 * 4 만큼의 데이터를 로드함.
  - 각 스레드가 m^2 * 2 만큼의 데이터를 계산함.
  - Ratio: 0.5 OP/B 
- 타일링 포함:
  - 타일이 tile_size^2 * 4 byte 만큼의 로드를함
  - 타일이 (tile_size - m + 1)^2 * m^2 * 2 만큼의 계산을 함.
  - Ratio: 0.5 * M^2 * (1 - (M - 1) / T)^2 OP/B
    - M이 5, T가 32일때 약 9.57 OP/B로 19배 향상된 것

## 경계 조건

- 출력을 계산하기 위해 입력 타일을 로드할 때 두가지 방법이 있음.
  - 원래 전역 메모리가 인바운드였는지 확인하는법
  - ghost elements라고 불리는 그냥 0을 채워넣는 방법.
