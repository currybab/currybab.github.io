+++
title = 'pmpp lecture 02 요약'
date = 2025-11-05T21:42:14+09:00
tags = ['pmpp', 'gpu']
categories = ['pmpp']
+++

### parallelism의 종류

- task parallelism
  - 같거나 다른 데이터들에 대해 다른 연산을 수행
  - 워드 같은 곳에서 편집기에서 작업을 하는 동안 스펠 체커가 돌아가는 경우
  - (많지 않은) 제한된 양의 병렬 수행만 가능함.
- data parallelism
  - 다른 데이터에 대해 같은 연산을 수행
  - 화면에 어떤 pixel을 어떻게 표시할지 연산하는 연산
  - 엄청난 양의 데이터에서 엄청난 양의 연산을 할 수 있음(GPU에 적합) 

### GPU가 있는 system의 구조

CPU (host)
main memory (host memory)

GPU (device)
GPU memory (global memory)

- cpu와 gpu는 분리된 메모리를 가지고 있음. 서로의 메모리에 접근이 불가능함.
- nvlink나 pcie 같은 interconnect를 통해 메모리간에 데이터를 주고받음.
- gpu kernel을 실행하기 위해서는 다음과 같은 과정을 거침.
  1. gpu memory 할당
  2. gpu memory에 데이터 복사
  3. gpu kernel 실행
  4. gpu memory에서 데이터 복사
  5. gpu memory 해제

### CUDA Memory Management API
- Allocating memory
  - `devPtr`: pointer to pointer to allocated device memory
  - `size`: requested allocation size in bytes
```c
  cudaError_t cudaMalloc(void **devPtr, size_t size);
```
  
- Deallocating memory
  - `devPtr`: pointer to device memory to free
```c
  cudaError_t cudaFree(void *devPtr);
```

- Copying memory
  - `dst`: destination memory address
  - `src`: source memory address
  - `count`: size in bytes to copy
  - `kind`: type of transfer
    - `cudaMemcpyHostToHost`
    - `cudaMemcpyHostToDevice`
    - `cudaMemcpyDeviceToHost`
    - `cudaMemcpyDeviceToDevice`
    - `cudaMemcpyDefault`: let the runtime choose the type of transfer
```c
  cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
```

- Return type: `cudaError_t` - helps with error checking

### Grid, block, thread (2단계 구조)

- grid: GPU에 있는 스레드의 배열. GPU에 그리드를 실행하여 작업을 수행한다고 말함.
- block: 이 스레드들은 그룹을 지어 쓰레드 블록이라 부름
  - 같은 스레드 블록 내에 있는 스레들은 서로 협력할 수 있음.
- 스레드 그리드를 수행하는 것을 하고 싶다면,(커널을 호출하려면)
  - 그리드에 몇개의 블록이 있는지 지정하고 (N / block size)
  - 각 블록에 몇개의 스레드가 있는지 지정해야함.  (보통 1024, 512, 256, 128,...)
- 동일한 그리드내의 쓰레드들은 커널이라 불리는 같은 함수를 실행함.
```c
  vecadd_kernel<<< numBlocks, threadsPerBlock >>>();
```

### Grid Dimension
- `gridDim.x`: number of blocks in grid
- `blockIdx.x`: position of block in grid
- `blockDim.x`: number of threads in block
- `threadIdx.x`: position of thread in block
- `unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;`: global index of thread 

### 컴파일
- `nvcc vecadd.cu -o vecadd`
- nvcc가 host c/c++ 코드와 gpu 코드를 분리함.
- host c/c++코드는 host c/c++ compiler로 컴파일됨.
- gpu 코드는 ptx라고 불리는 virtual isa 코드로 컴파일됨.
- ptx 코드는 gpu에 따라 다르게 컴파일되어 device assembly를 컴파일함. (device just-in-time compiler 사용)

### 어떤 코드가 GPU로 가고 CPU로 가는가
- keyword를 통해 구분함
- `__host__`: Host에서 호출하고 Host에서 실행되는 코드 (default)
- `__global__`: Host나 GPU에서 호출하고 GPU에서 실행되는 코드 (커널 함수)
- `__device__`: GPU에서 호출하고 GPU에서 실행되는 코드
- 왜 `__host__`가 필요한가?
  - host와 device 모두에서 실행되는 코드가 있을 수 있음.
  - 그경우에 `__host__ __device__`를 통해 두곳에서 전부 사용할 수 있음.

### 비동기 커널 호출
- 커널 호출은 기본적으로 비동기적으로 실행 됨
- gpu에서 실행되는 동안 cpu가 작업을 병렬로 실행할 수 있게 해줌.
- `cudaError_t cudaDeviceSynchronize()`를 통해 동기화를 할 수 있음.

### 에러 체크
- 쿠다 api는 cudaError_t 형태로 error code를 반환함.
- 디바이스 동기화를 통해 에러를 체크하거나 `cudaError_t cudaGetLastError()`를 통해 마지막에 발생한 에러를 확인할 수 있음.


source: [Lecture 02 - Data Parallel Programming](https://www.youtube.com/watch?v=iE-xGWBQtH0&list=PLRRuQYjFhpmubuwx-w8X964ofVkW1T8O4&index=2)
