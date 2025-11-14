+++
title = "Pmpp Lec09 Stencil 요약"
date = "2025-11-14T19:54:11+09:00"

#
# description is optional
#
# description = "An optional description for SEO. If not provided, an automatically created summary will be used."

tags = ["gpu", "pmpp", "CUDA"]
+++

source: [Lecture 09 - Stencil](https://www.youtube.com/watch?v=NOoSyDCVRU0&t=9s)

## Stencil

- 그리드 지점의 값은 해당 지점 이웃의 부분집합을 기반으로 계산되는 구조화된 그리드에서 수행되는 계산의 한 종류
  - 5-point stencil (2D)
  - 7-point stencil (3D)
  - stencil은 특별한 경우의 convolution으로 볼 수 있음.
- 출력 값은 해당 입력 요소와 차원별 이웃의 입력값들을 기반으로 계산됨. 
- Parallelization Approach
  - 출력 그리드 포인트당 하나의 스레드를 할당.(convolution과 비슷하게)
  - 내부 출력값에 대해서만 스텐실을 계산할 것
    - 접근하는 입력값들이 항상 유효하게 됨
    - 이러면 boundary 체크를 하지 않아도 됨.

### Stencil Code

```c
#define BLOCK_DIM 8

__global__ void stencil_kernel(float* in, float* out, unsigned int N) {
    unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
        out[i * N * N + j * N + k] = C0 * in[i * N * N + j * N + k]
                                    + C1 * (in[i * N * N + j * N + (k - 1)]
                                        + in[i * N * N + j * N + (k + 1)]
                                        + in[i * N * N + (j - 1) * N + k]
                                        + in[i * N * N + (j + 1) * N + k]
                                        + in[(i - 1) * N * N + j * N + k]
                                        + in[(i + 1) * N * N + j * N + k]);
    }
}
```

## Data Reuse in Stencil (tiling)

- 출력의 인접한 요소들이 동일한 입력 요소들을 사용하고 있음.
- 재사용률이 convolution 만큼 높지는 않음
- 타일을 공유 메모리에 로드하여 액세스함.
  - input tile dimension 과 output tile dimension 이 다름.
  - output tile dimension = input tile dimension - 2
  - 컨볼루션과 비슷하게 input tile 만큼의 쓰레들르 실행하여 로드하고 일부만 계산하는데 사용함.

```c
#define BLOCK_DIM 8
#define IN_TILE_DIM BLOCK_DIM
#define OUT_TILE_DIM (IN_TILE_DIM - 2)

__global__ void stencil_kernel(float* in, float* out, unsigned int N) {
    unsigned int i = blockIdx.z * OUT_TILE_DIM + threadIdx.z - 1;
    unsigned int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    unsigned int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;

    // 입력 타일 로드
    __shared__ float in_s[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];
    if (i >= 0 && i < N && j >= 0 && j < N && k >= 0 && k < N) {
        in_s[threadIdx.z][threadIdx.y][threadIdx.x] = in[i * N * N + j * N + k];
    }
    __syncthreads();
    
    if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
        if (threadIdx.x >= 1 && threadIdx.x < blockDim.x - 1 
            && threadIdx.y >= 1 && threadIdx.y < blockDim.y - 1 
            && threadIdx.z >= 1 && threadIdx.z < blockDim.z - 1) {
            out[i * N * N + j * N + k] = C0 * in_s[threadIdx.z][threadIdx.y][threadIdx.x]
                                        + C1 * (in_s[threadIdx.z][threadIdx.y][threadIdx.x - 1]
                                            + in_s[threadIdx.z][threadIdx.y][threadIdx.x + 1]
                                            + in_s[threadIdx.z][threadIdx.y - 1][threadIdx.x]
                                            + in_s[threadIdx.z][threadIdx.y + 1][threadIdx.x]
                                            + in_s[threadIdx.z - 1][threadIdx.y][threadIdx.x]
                                            + in_s[threadIdx.z + 1][threadIdx.y][threadIdx.x]);
        }
    }
}
```

### ratio analyze (왜 성능 향상이 없었는가)

- Original kernel
  - 각 스레드가 8OPs (6FP adds and 2FP muls)
  - 각 스레드가 28B를 global memory로 로드 (7 FP values)
  - Ratio: (8 OPs) / (28 B) = 0.29 OP/B
- Tiled kernel
  - input tile size를 T라고 하면 output tile size는 T-2
  - 각 블록은 (8 OPs) * (T - 2)^3 OPs 연산을 함.
  - 각 블록은 (4 Byte) * T^3 만큼 로드함.
  - Ratio: (8 OPs * (T - 2)^3) / (4 B * T^3) = 2 * (1 - 2/T)^3
    - T가 8일때 0.84 OP/B
    - 사실 개선 폭이 오버헤드에 비해 큰편이 아님. 
    - T를 증가시키는 것이 ratio를 개선하는데 도움이 됨.
      - T가 32일때 1.65 OP/B로 2배 개선됨.
      - T를 늘리는 아이디어 뒤의 직관은 경계요소는 data 재사용이 적음. 사이즈를 늘리면 상대적으로 경계 값의 수가 적어짐.
  
### Input tile size 늘리기

- 두 가지 이유로 그냥 늘릴 수는 없음.
  - 입력타일 크기가 하드웨어에 의해 제한된 블록 크기와 같기 때문.
  - input size를 늘리면 shared memory를 초과할 수 있음. (64KB)
    - limit을 초과하지 않더라도 shared memory를 많이 사용하면 점유율에 영향이 갈 수 있음.
- 스레드 수를 늘리지 않고도 큰 출력을 만드는 법
  - thread coarsening을 사용하여 더 큰 입출력 타일을 처리하는 것.
  - 각 단계에 필요한 input tile의 slice만 shared memory에 유지함. (제한을 피하기 위해)

## Stencil with Thread Coarsening Code

```c
__global__ void stencil_kernel(float* in, float* out, unsigned int N) {
    int iStart = blockIdx.z * OUT_TILE_DIM;
    int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;

    __shared__ float inPrev_s[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ float inCurr_s[IN_TILE_DIM][IN_TILE_DIM];
    __shared__ float inNext_s[IN_TILE_DIM][IN_TILE_DIM];
    if (iStart - 1 >= 0 && iStart - 1 < N && j >= 0 && j < N && k >= 0 && k < N) {
        inPrev_s[threadIdx.y][threadIdx.x] = in[(iStart - 1) * N * N + j * N + k];
    }
    if (iStart >= 0 && iStart < N && j >= 0 && j < N && k >= 0 && k < N) {
        inCurr_s[threadIdx.y][threadIdx.x] = in[iStart * N * N + j * N + k];
    }
    __syncthreads();

    for (int i = iStart; i < iStart + OUT_TILE_DIM; i++) {
        if (i + 1 >= 0 && i + 1 < N && j >= 0 && j < N && k >= 0 && k < N) {
            inNext_s[threadIdx.y][threadIdx.x] = in[(i + 1) * N * N + j * N + k];
        }
        __syncthreads();
        if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
            if (threadIdx.x >= 1 && threadIdx.x < blockDim.x - 1 
            && threadIdx.y >= 1 && threadIdx.y < blockDim.y - 1) {
                out[i * N * N + j * N + k] = C0 * inCurr_s[threadIdx.y][threadIdx.x]
                                        + C1 * (inPrev_s[threadIdx.y][threadIdx.x]
                                            + inNext_s[threadIdx.y][threadIdx.x]
                                            + inCurr_s[threadIdx.y][threadIdx.x - 1]
                                            + inCurr_s[threadIdx.y][threadIdx.x + 1]
                                            + inCurr_s[threadIdx.y - 1][threadIdx.x]
                                            + inCurr_s[threadIdx.y + 1][threadIdx.x]);
            }
        }
        __syncthreads();
        inPrev_s[threadIdx.y][threadIdx.x] = inCurr_s[threadIdx.y][threadIdx.x];
        inCurr_s[threadIdx.y][threadIdx.x] = inNext_s[threadIdx.y][threadIdx.x];
    }
}
```

### Unshared Slices

- 관찰
  - 오직 현재 슬라이스만 스레드 블록 전체에서 진정으로 공유됨. 이전과 다음 슬라이스는 로드한 스레드만이 필요로 함.
- 최적화
  - 이전과 다음 슬라이스는 레지스터에 넣어 shared memory를 절약함.
  - 현재 평면이 되면 shared memory로 복사함(레지스터에서)
  - 한 평면을 위한 충분한 공유 메모리만 있으면 됨
  - 서로 다른 스레드의 레지스터가 모여서 타일을 이룸 (레지스터 타일링)
    - 레지스터 타일링이 새로운 것은 아님. 행렬곱에서 C tile의 값이 스레드의 레지스터에 저장되었다가 출력요소를 만드는데 사용됨.

### Stencil with Register Tiling Code

```c
__global__ void stencil_kernel(float* in, float* out, unsigned int N) {
    int iStart = blockIdx.z * OUT_TILE_DIM;
    int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;

    float inPrev_s;
    __shared__ float inCurr_s[IN_TILE_DIM][IN_TILE_DIM];
    float inNext_s;
    if (iStart - 1 >= 0 && iStart - 1 < N && j >= 0 && j < N && k >= 0 && k < N) {
        inPrev_s = in[(iStart - 1) * N * N + j * N + k];
    }
    if (iStart >= 0 && iStart < N && j >= 0 && j < N && k >= 0 && k < N) {
        inCurr_s[threadIdx.y][threadIdx.x] = in[iStart * N * N + j * N + k];
    }
    __syncthreads();

    for (int i = iStart; i < iStart + OUT_TILE_DIM; i++) {
        if (i + 1 >= 0 && i + 1 < N && j >= 0 && j < N && k >= 0 && k < N) {
            inNext_s = in[(i + 1) * N * N + j * N + k];
        }
        __syncthreads();
        if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
            if (threadIdx.x >= 1 && threadIdx.x < blockDim.x - 1 
            && threadIdx.y >= 1 && threadIdx.y < blockDim.y - 1) {
                out[i * N * N + j * N + k] = C0 * inCurr_s[threadIdx.y][threadIdx.x]
                                        + C1 * (inPrev_s
                                            + inNext_s
                                            + inCurr_s[threadIdx.y][threadIdx.x - 1]
                                            + inCurr_s[threadIdx.y][threadIdx.x + 1]
                                            + inCurr_s[threadIdx.y - 1][threadIdx.x]
                                            + inCurr_s[threadIdx.y + 1][threadIdx.x]);
            }
        }
        __syncthreads();
        inPrev_s = inCurr_s[threadIdx.y][threadIdx.x];
        inCurr_s[threadIdx.y][threadIdx.x] = inNext_s;
    }
}
```

### Q&A

- 실제로 많이 하는 방법중하나는 차원 하나를 나누지 않고 32*32 블록을 사용하여 처음부터 끝까지 전체공간을 통과함 
  - 훨씬 더 큰 출력 타일을 제공할 수 있음. (T를 늘리는 방법)
