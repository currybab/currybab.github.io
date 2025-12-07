+++
title = "Pmpp Lec20 Intra Warp Synchronization"
date = "2025-11-30T02:09:47+09:00"

#
# description is optional
#
# description = "An optional description for SEO. If not provided, an automatically created summary will be used."

tags = ["pmpp", "cuda", "gpu"]
+++

Source: [Lecture 20 - Intra Warp Synchronization](https://www.youtube.com/watch?v=g5ZKBH6UQvE)

## 복습: Warp

- SM에서 스케줄링의 단위
    - 사이즈는 장치별로 다르지만 현재까지는 항상 32개의 스레드였음.
- 워프내 스레드의 독특한 점은 SIMD 모델에 따라 실행된다는 것. (Single Instruction, Multiple Data)
    - 하나의 명령어를 가져와 해당 워프의 모든 스레드에 대해 다른 데이터로 실행한다는 의미

## Intra-Warp Synchronization

- 우리는 동일한 워프 내 스레드 간의 특별한 관계를 활용하여 스레드 간에 빠른 동기화를 할 수 있음.
- CUDA에는 작업간 동기화를 위한 내장 함수가 있음. 
    - 스레드간에 데이터를 섞는(공유하는) 것
        - 스레드는 레지스터 중 하나의 값을 가져올 수 있음.
        - 공유 메모리를 거치지 않고
    - 스레드간의 voting

## Warp Shuffle Functions

- 내장된 워프 셔플 명령은 스레드가 동일한 워프의 다른 스레드와 데이터를 공유할 수 있게 함.
    - 원래는 공유 메모리를 써서 해야했음. 
- 종류
    - `__shfl_sync()`: 특정 레인(워프 내의 스레드 중 하나)에서 복사하는 역할을 함.
    - `__shfl_up_sync()`: 더 낮은 ID를 가진 레인에서 셔플링할 스레드와 비교하여 셔플링함.
    - `__shfl_down_sync()`: 더 높은 ID를 가진 레인에서 셔플링할 스레드와 비교하여 셔플링함.
    - `__shfl_xor_sync()`: 자신만의 레인 ID를 XOR하여 다른 레인에서 셔플링할 스레드와 비교하여 셔플링함.
- 생성한 값을 다른 스레드들이 읽는 작업의 예
    - scan
    - reduction


## Reduction with Warp Shuffle

- 마지막 워프가 남았을 때, 셔플을 이용함.

```c
    #define BLOCK_DIM 1024
    #define WARP_SIZE 32
    
    __global__ void reduce_kernel(float* input, float* partialSums, unsigned int N) {
        unsigned int segment = (blockIdx.x * blockDim.x) * 2;
        unsigned int i = segment + threadIdx.x;

        __shared__ float input_s[BLOCK_DIM];
        input_s[threadIdx.x] = input[i] + input[i + BLOCK_DIM];
        __syncthreads();

        for (unsigned int stride = BLOCK_DIM / 2;  stride > WARP_SIZE; stride /= 2) {
            if (threadIdx.x < stride) {
                input_s[threadIdx.x] += input_s[threadIdx.x + stride];
            }
            __syncthreads();
        }

        // Reduction tree with warp shuffle
        float sum;
        if (threadIdx.x == 0) {
            sum = input_s[threadIdx.x] + input_s[threadIdx.x + WARP_SIZE];
        }
        for (unsigned int stride = WARP_SIZE / 2;  stride > 0; stride /= 2) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, stride);
        }
        
        if (threadIdx.x == 0) {
            partialSums[blockIdx.x] = sum;
        }
    }
```

## Warp Vote Functions

- 작업 내 스레드가 특정 조건을 만족하는지 투표할 수 있도록 함.
- 종류
    - `__all_sync(unsigned mask, predicate)`
        - 셔플 명령어에 참여하는 스레드의 mask와 predicate를 가져와서 참여하는 모든 스레드에 대해 0이 아닌지 체크함.
    - `__any_sync(unsigned mask, predicate)`
        - 참여하는 스레드 중 predicate가 0이 아닌 것이 적어도 하나가 있는지 체크함.
    - `__ballot_sync(unsigned mask, predicate)`
        - predicate가 0이 아닌 mask를 반환함. (32-bit integer)
    - `__activemask()`
        - 특정 시점에 활성화된 스레드를 알려줌.
        - 제어 발산이 있는 곳에서 warp 명령어를 수행할 때 유용함.

### Optimization with Warp Vote

- 최적화: 워프당 한 스레드만 다른 것들을 대신해서 전역 카운터를 증가시킴.
- 단계
    - 리더 스레드를 정함.
    - 워프에 큐에 넣고 싶은 스레드가 얼마나 많은지 찾음
    - 리더가 atomic operation을 수행해 큐의 공간을 예약함.
    - 자신의 결과를 다른 모든 스레드에게 broadcast함
    - 각 스레드는 자신의 결과에 대한 오프셋을 해당 위치에서 알아냄.

```c
    #define WARP_SIZE 32

    // enqueue kernel 값이 어떤 작업을 충족하면 enqueue 함.
    __global__ void enqueue_kernel(unsigned int* input, unsigned int N, unsigned int* queueSize) {
        // unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        // if (i < N) {
        //     unsigned int val = input[i];
        //     if(cond(val)) {
        //         unsigned int j = atomicAdd(queueSize, 1);
        //         queue[j] = val;
        //     }
        // }
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < N) {
            unsigned int val = input[i];
            if(cond(val)) {
                // Assign a leader thread (0이 이 지점에 도달 못할 수도 있음)
                // - __activemask()를 사용하여 활성화된 스레드를 확인함.
                // - 첫번째 활성화된 스레드를 리더로 지정함.
                unsigned int activeThreads = __activemask();
                unsigned int leader = __ffs(activeThreads) - 1; // activemask의 첫번째 비트를 찾는다. 
                // 워프 수준 아니고 그냥 내장 함수임, 인덱스가 1부터 시작함. 0이면 1로 설정된 bit가 없음.

                // Find how many threads need to add to the queue = how many threads are active
                // 또다른 내장 함수가 있음(population count)
                unsigned int numActive = __popc(activeThreads);

                // Have the leader perform the atomic operation
                unsigned int j;
                if (threadIdx.x % WARP_SIZE == leader) {
                    j = atomicAdd(queueSize, numActive);
                }

                // Broadcast the result to all threads
                j = __shfl_sync(activeThreads, j, leader);
                
                // Find offset of each active thread and store result
                // - 활성 상태인 이전 스레드의 수를 찾음
                unsigned int previousThreads = (1 << (threadIdx.x % WARP_SIZE)) - 1;
                unsigned int previousActiveThreads = activeThreads & previousThreads;
                unsigned int offset = __popc(previousActiveThreads);

                // Store the result
                queue[j + offset] = val;
            }
        }
    }
```
