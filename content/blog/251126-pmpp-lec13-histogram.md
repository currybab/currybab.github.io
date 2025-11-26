+++
title = "Pmpp Lec13 Histogram"
date = "2025-11-26T13:01:02+09:00"

#
# description is optional
#
# description = "An optional description for SEO. If not provided, an automatically created summary will be used."

tags = ["pmpp", "gpu", "cuda"]
+++

Source: [Lecture 13 - Histogram](https://youtu.be/BiYieuVwUbg?si=aD0nVM-3R_Z03u6u)

### Today

- Parallel patterns: histogram
- New feature: atomic operations
- Optimization: privatization

## Histogram

- 데이터셋의 분포를 근사화함
  - 입력 데이터가 가질수 있는 값의 범위를 bin(또는 bucket)들로 나누고,
  - 각 bin에 해당하는 데이터의 개수를 셈
- 예제: 이미지의 색상 히스토그램(color histogram)은 각 픽셀 값(또는 값의 범위)마다 픽셀의 개수를 세는 것

### Color Histogram 

- Sequential

```c
for (int i = 0; i < width * height; i++) {
    unsigned char b = image[i];
    ++bins[b];
}
```

- Parallel
    - 이미지의 모든 입력 픽셀에 스레드를 할당하고 모든 스레드가 해당 벤드를 찾아 업데이트하기.(가장 기본)

```c
    __global__ void histogram(unsigned char* image, unsigned int* bins, unsigned int width, unsigned int height) {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i < width * height) {
            unsigned char b = image[i];
            ++bins[b]; // !!!! Data Race error
            // atomicAdd(&bins[b], 1);
        }
    } 
```

## Data Races

- 여러 스레드가 순서 없이 동시에 동일한 메모리 위치에 액세스하고 적어도 하나의 액세스가 쓰기일때 data race가 발생함.
  - 데이터 경쟁은 예측할 수 없는 프로그램 출력으로 이어짐.

### Mutual Exclusion(상호 배타)

- 데이터 경쟁을 피하려면, 동일한 메모리 위치에 동시적인 read-modify-write 작업들의 순서를 강제하기 위해 상호 배타적으로 만들어야함.
- 병렬성을 죽이지만 동일한 출력에 대한 경합이 발생하면 피할 수 없는 일.
- CPU에서는 lock(mutex)를 사용하여 달성할 수 있음.

### Locks and SIMD Execution

- thread 0과 1이 같은 와프에 있고 같은 락을 획득하려는 상황에서.
- thread 0이 락을 획득하고 thread 1은 락을 획득할 수 없음.
- thread 1은 thread 0이 락을 해제할 때까지 기다림.
- 하지만 같은 워프에 있음
  - 이는 다음 명령이나 다음 작업을 실행하려고 할때 함께 실행됨을 의미함.
  - 스레드 1이 묶여있기 때문에 스레드 0이 실행될 수 없음
  - 스레드 0은 스레드 1이 simd 를 완료할떄까지  기다려야 함.(데드락!)
- 잠금을 이용한 SIMD 실행은 deadlock을 일으킬 수 있음.

## Atomic Operations (원자적 연산)

- 원자적 연산은 GPU에서 단일 ISA 명령으로 read-modify-write를 수행함.
- 하드웨어가 작업이 끝날때까지 다른 스레드가 메모리 위치에 액세스할 수 없음을 보장함.
- 만약 동일한 메모리 위치에 동시에 원자적 연산을 수행한다면 하드웨어에 의해 직렬화됨.
  
### Atomic Operations in CUDA

- Atomic Add: `T atomicAdd(T* address, T val);`
  - T는 int, unsigned int, float, double 등이 가능함.
  - adress에 있던 값을 읽고 val을 더하고 새 값을 address에 저장하고 원래 저장되어 있던 old value return함.
  - 단일 ISA 명령으로 변환됨. (이런 종류를 내장함수(intrinsics라고 얘기함)
- sub, min, max, inc, dec, and, or, xor, exchange, compare and swap 등이 있음.
  
## High Latency

- 서로 다른 스레드 블록에서 여러 스레드가 모두 동일한 픽셀을 갖고 있다면 동일한 bin을 업데이트하게됨.
  - global memory에 대한 원자적 연산은 high latency를 갖게 됨.
    - 긴 시간이 걸리는 읽고 쓰기가 완료되기를 기다려야 함.
    - 다른 스레드가 같은 위치에 접근하고 있다면 기다려야 함.(경합 발생 확률이 높음)

## Privatization

- 공유 출력에서 경쟁하는 여러 스레드가 있을 때 적용할 수 있음.
  - 순서가 변경될 수 있기 때문에 출력에 대한 연산은 결합적이고 교환적이어야 함.
- 각 스레드 블록에 대해 히스토그램의 프라이빗 복사본을 만듬.
- 각 스레드 블록은 히스토그램의 private copy를 업데이트함.
- 그러면 히스토그램의 자체 개인 복사본 내에서만 경쟁하게 되고, 다른 스레드블록의 모든 스레드와 경쟁하지는 않음.
- 스레드 블록이 완료되면, 히스토그램의 private copy에 축적된 값을 global memory에 atomic add함.
  - private copy가 너무 많으면 리덕션 트리 접근 방식을 고려해볼 수 있음.
- 다른 스레드 블록의 스레드 간의 경쟁을 줄이는 일.

## Coarsening

- 더 적은 스레드 블록을 사용하면 private copy 수가 줄어들고, global copy를 업데이트하는데 더 적은 global memory atomic 연산으로 이어짐.
- 블록의 수를 줄이기 위해 thread coarsening을 적용하고 각 스레드가 여러 입력을 계산하도록 함
  - 입력을 로드할 때 coalesced(병합)하게 해야함.
