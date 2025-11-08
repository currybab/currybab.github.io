+++
title = "Pmpp Lec05 Memory and Tiling 요약"
date = "2025-11-08T18:22:13+09:00"

#
# description is optional
#
# description = "An optional description for SEO. If not provided, an automatically created summary will be used."

tags = ['pmpp', 'gpu']
categories = ['pmpp']
+++

source: [Lecture 05 - Memory and Tiling](https://www.youtube.com/watch?v=31ZyYkoClT4&list=PLRRuQYjFhpmubuwx-w8X964ofVkW1T8O4&index=6)

### Today

- Memory in the GPU architecture
- Memory in the CUDA programming model
- Optimizing memory access

## Performance Metrics

- 프로세서 설계자가 프로세서를 설계할 때 다양한 지표를 통해 보통 이 프로세서의 성능에 대한 정보를 제공함.
    - FLOPS Rate: 프로세서가 초당 실행할 수 있는 부동 소수점 연산의 수
        - 프로세서의 코어가 단위 시간당 얼마나 많은 연산을 실행할 수 있는가, 달성할 수 있는 최대 속도
    - Memory Bandwidth: 프로세서의 메모리가 이 프로세서의 코어에 공급할 수 있는 초당 바이트 수
        - V100의 경우 14 TFlops, 900 GB/s
- 지표가 보장되는 것은 아님
    - 이 이상의 성능을 달성하는것이 물리적으로 불가능을 의미함
    - 모든 코어에서 매 사이클마다 부동 소수점 연산을 실행한다면 이런 수치가 나올것.
    - 프로세서의 성능을 가늠하는 좋은 지표가 됨.
    - 또한 코드의 성능을 비교하여 이러한 최고치에 근접 했는지, 아니면 한참 멀리 떨어져 있는지 확인하는 좋은 참고 자료가 됨.
    - 아주 멀리 떨어져 있다면 코드가 얻을 수 있는 잠재적인 성능 측면에서 어디에 서있는지에 대해 알 수 있음

## Performance Bounds

- Compute-bound(연산 제약적): Flops rate에 의해 성능이 제한됨.
    - 프로세서의 코어들은 항상 완전히 활용되고 있는 상태임.
- Memory-bound(메모리 제약적): Memory Bandwidth에 의해 성능이 제한됨.
    - 메모리를 기다리느라 프로세서 코어는 유휴 상태일 수 있음.
- Desired Compute-to-Memory Ratio (OP/B) : V100의 경우 15.6
    - 컴퓨팅 대 전역 메모리 접근 비율
    - 최고의  성능을 달성하기위해서는 부동 소수점(4바이트)당 60번의 부동 소수점 연산을 해야함.
  
## Example: Vector Addition

```c
    z[i] = x[i] + y[i];
```

- 벡터 덧셈 커널은 0.125 OP/B로 수행함.
    - 2개의 FP value를 로드하고(8 바이트) 1 FP add operation을 수행함.
    - 저장은 무시함.
    - 코어를 완전히 활용하기 위한 값인 15.6과 매우 멈.
- 벡터 덧셈은 매우 메모리 제약적임을 알 수 있음.
    - 따라서 생각보다 큰 속도 향상을 보지 못했었음.
    - 대부분의 ALU가 유휴 상태였기 때문에...
    - 더 많은 연산을 투입해도 개선 효과는 미미할 것
    - 우리가 최적화적으로 더 할수 있는게 딱히 없음.(너무 간단한 커널이여서)
  
## Example: Matrix-Matrix Multiplication

```c
    for (unsigned int i = 0; i < N; i++) {
        sum += A[row*N + i] * B[i*N + col];
    }
```

- 행렬 곱셉 커널은 0.25 OP/B로 수행함. 
    - 두번의 floating point 연산이 있음. 곱셈 한번, 덧셈 한번
    - 벡터 덧셈 연산과 다르게 행렬 곱셈 연산은 실제로 데이터 재사용 가능성이 매우 높음.
- 행렬 곱셈을 수행하는데 필요한 총 부동 소수점 연산 수는 얼마인가? 
    - N * N 행렬에 대해서 
        - 데이터 로드: (2 input matrices) * (N^2 values) * (4 B) = 8 N^2 B
        - 연산 수: (N^2 dot products) (N adds + N muls each) = 2 N^3 OP
        - 잠재적 compute-to-memory ratio: 2 N^3 OP / 8 N^2 B = 0.25 N OP/B
  
### Reuse in Matrix-Matrix Multiplication

- 관찰: 입력의 각 요소가 출력의 모든 row/column에 사용됨
- 커널에서 메모리 접근을 최적화할 기회가 있으므로 global memory 로부터 로드한 입력데이터를 가능한 많이 재사용할 수 있음.
    - 각 출력요소를 계산하기 위해 전역 메모리에서 데이터를 로드할 필요없이 어딘가에 로드할 수 있는 방법이 있을지도 모름.
    - 일부 스레드만 이 데이터를 로드하고 다른 스레드들이 전역 메모리로 갈 필요 없이 이 데이터에 접근할 수 있도록 함.
    - 그러면 global memory에 접근하는 것을 줄여 성능을 높일 수 있음.

## GPU 아키텍쳐 에서의 메모리

- SM이 스트리밍 프로세서를 가지고 있음.
  - 공유 제어 로직을 가지고 있고, 로컬 메모리에 대한 접근을 공유함.
- 글로벌 메모리에 대한 접근은 일반적으로 약 500 cycle정도를 소모함.
- SM은 레지스터를 갖고 있고 이를 접근하는데는 1cycle이 걸림.
- SM에는 L1 캐시가 있음, 전역 메모리에 있는 데이터를 다시 액세스할 경우를 대비하여 데이터를 보관함.
- 또 공유 메모리가 있음. 프로그램에 의해 관리됨. L1 캐시와 shared memory는 5 cycle을 소모함.
- 또 다른 메모리로 constant cache가 있음. (딱히 다루지는 않음, 상수 데이터를 저장하고 5cycle정도 소모함.)
- L2 캐시도 있음. On chip memory이지만 SM에 있지는 않음. 전역 메모리를 위한 캐시임.

## CUDA 프로그래밍 모델에서의 메모리

- 그리드는 블록으로 나뉘고 블록은 스레드로 나뉨.
- 각 스레드는 자체 전용 레지스터에 접근할 수 있음.
- 동일한 스레드 블록에 있는 각 스레드는 동일한 공유 메모리에 접근할 수 있음.
    - 같은 블록에 있는 모든 스레드들은 모두 동일한 shared memory에 접근할 수 있음.
    - 다른 블록의 공유 메모리 데이터에는 접근할 수 없음.
- 그리드 내의 모든 스레드가 동일한 전역 메모리에 접근할 수 있음.
- 또 모두 동일한 상수 메모리에 접근할 수 있음.
- l1, l2 캐시는 하드웨어 레벨에서 관리되기 때문에 프로그래머가 직접 관리할 필요가 없음.
- 공유 메모리를 활용하여 전역모드로 이동해야 하는 양을 줄일 수 있음.

## CUDA Type Qualifies

- 쿠다는 프로그래머가 특정 데이터가 어디에 상주하기를 원하는지 지정하는데 도움이되는 다양한 한정자를 제공함.
- `cudaMalloc(...)`은 host에서 global memory에 데이터를 할당하는 함수임.
- `__device__ int globalVar;`: global memory, gird scope, application lifetime
- `__device__ __constant__ int constVar;`: constant memory, grid scope, application lifetime
- `__device__ __shared__ int sharedVar;`: shared memory, block scope, block lifetime
- `int localVar;`: register, thread scope, thread lifetime
- `int localArr[N];`: global memory, thread scope, thread lifetime (잘 안 씀)

## Reuse in Matrix-Matrix Multiplication again

- 행렬 곱셈을 병렬화하고 싶을 때, 우리가 했던 것은 출력 행렬을 타일 또는 블록으로 나누고 각 타일에 스레드 블록을 할당했음.
    - 그리고 해당 타일의 각 요소에 스레드를 할당했음.
- 관찰
    - 스레드 블록 안에서 같은 행에 있는 출력요소는 모두 A의 동일한 행을 로드함.
    - 비슷하게 같은 열에 있는 출력요소는 모두 B의 동일한 열을 로드함.
    - 같은 블록에 있는 스레드들이 공유 메모리를 활용할 수 있다는 것을 알았기 때문에 입력을 공유할 수 있음.
- 운이 좋다면, 스레드가 l1 캐시에서 데이터를 찾을 수 있음.
- 운이 없다면 데이터가 l1캐시에서 제거 되어서 데이터를 찾을 수 없음.
  - GPU에서는 훨씬 더 많은 스레드를 가지고 있기 때문에 실제로 l1캐시에서 데이터를 찾을 수 없을 확률이 높음.
- 데이터 재사용률이 매우 높다는 것을 알고 있다면 하드웨어 캐시에 의존하지 않고 공유 메모리를 사용하여 직접 캐싱을 시도하는 것이 나음.
- 솔루션:
    - 스레드들이 협력하여 데이터의 일부를 로드하도록 하게 함
    - 사용해야 하는 모든 스레드가 더 많은 데이터를 로드하기 전에 공유메모리를 통해 접근하도록 보장함.
    - 입력을 데이터 타일로 나누기 때문에 tiling이라고 불리는 최적화 방법임.
  
## Tiled Matrix-Matrix Multiplication

- Step 1: 각 입력의 첫번째 타일을 공유 메모리에 로드하는 것.
- Step 2: 출력 행렬에 할당된 스레드들이 공유메모리에 접근하여 타일들의 partial dot product를 계산함. (스레들이 서로 끝나는 것을 기다림)
- Step 3: 글로벌 메모리로 가서 다음 타일을 가져옴. 다시 step2를 수행함.
- 이제 요소를 처리하기 위해 행의 모든 요소를 로드하기 전에 타일들의 수만큼만 가져오고 다른 요소들은 다른 스레들이 가져오게 될것임.
  - 모든 스레드들은 각각 타일당 한번씩 로드하게 될 것임.

### Boundary Conditions

- 기존과는 다르게 단순하게 N을 기준으로 스레드를 비활성화 시킬수는 없음.
  - 각 스레드들이 앞쪽에 채워진 공간에 대해서 로딩에 책임을 질 수 있기 때문에

## CPU에서의 타일링

- cpu에서도 가능함
  - shared memory는 없고 캐시에 의존함.
  - 스레드가 적기 때문에 캐시가 상대적으로 커서 신뢰할 수 있음.
