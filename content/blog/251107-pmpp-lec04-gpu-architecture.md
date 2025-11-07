+++
title = "Pmpp Lec04 Gpu Architecture 요약"
date = "2025-11-07T22:44:48+09:00"

#
# description is optional
#
# description = "An optional description for SEO. If not provided, an automatically created summary will be used."

tags = ['pmpp', 'gpu']
categories = ['pmpp']
+++

source: [Lecture 04 - GPU Architecture](https://www.youtube.com/watch?v=pBQJAwogMoE&list=PLRRuQYjFhpmubuwx-w8X964ofVkW1T8O4&index=5)

## GPU Architecture

- GPU는 여러개의 Streaming Multiprocessor(SM)를 가지고 있음.
- SM은 여러 개의 코어로 구성되어 있고 코어들은 일종의 제어 장치(control)와 memory를 공유햠.
- 모든 SM은 동일한 Global Memory에 접근할 수 있음. (우리가 복사했던 GPU의 메모리에 해당)
- volta v100 gpu는 80 SMs with 64 cores each which total 5120 cores.

### thread를 어떻게 GPU에서 실행시키는가?

- 스레드가 블록 단위로 SM에 할당됨.
- 블록 세분성(block granularity): 스레드 블록 안의 모든 스레드는 같은 SM에 할당됨.
- SM은 여러 스레드 블록을 가질 수 있음.
- thread 실행하는데 자원이 필요하기 때문에 SM은 한번에 블록당 제한된 수의 스레드만 수용할 수 있음.
- 우리는 전체 GPU에서 한번에 제한된 수의 스레드 블록만 실행할 수 있음.
- 우리가 그리드를 실행하는데 동시에 실행할 수 있는 블록보다 더 많은 블록이 있다면 나머지 스레드 블록들은 어떤 큐에 대기하게 됨.

## Synchronization

- 같은 블록에 있는 스레드들은 같은 sm에 할당되기 때문에 다른 블록에 있는 쓰레드들과 다른 방식으로 협업할 수 있음.
  - Barrier synchronization: `__syncthreads()`
    - 같은 블록 내에 있는 스레드들이 코드의 특정 지점에서 서로를 기다린 후에 모두 진행할 수 있도록 함.
    - 서로 소통하는 방법 중 하나는 서로를 기다리는 것임.
  - Shared memory (discussed later)
    - 동일한 블록의 스레드가 액세스할 수 있는 SM에 빠른 공유 메모리가 있음.(다른 블록은 접근할 수 없음)
  - 다른 방법들도 있음. 

## Scheduling Consideration

- 같은 블록에 있는 스레드를 같은 SM에 할당하면 스레드간의 효율적인 협업이 더 쉬어짐.
- 블록의 모든 스레드들은 동시에 하나의 SM에 할당됨.
  - 블록은 모든 스레드가 실행되기에 충분한 리소스를 확보할 때까지 SM에 할당될 수 없다.
  - 그렇게 되지 않으면 sync과정에서 데드락이 발생할 수 있음.

## Transparent Scalability (투명한 확장성)

- 서로 다른 블록의 스레드들은 동기화 되지 않음.
  - CUDA프로그래밍 모델의 좋은 점은 서로 다른 블록의 스레드가 서로 협력할 수 없다는 것.
  - 이것은 스레드 블록을 서로 독립적으로 만듬
  - 블록들이 어떠한 순서로 실행되든 상관이 없음
  - 블록들은 병렬로 실행되거나 순차적으로 실행될 수 있음
- 투명한 확정성을 가능하게 함
  - 동일한 코드가 하드웨어 병렬 처리 수준이 다른 여러 장치에서 실행될 수 있다는것. 
    - SM이 적은 장치에서는 순차적으로 실행됨
    - 더 많은 SM을 가진 장치에서는 병렬로 실행됨
- 프로그래머는 블록 간에 동기화를 시도하는 코드를 작성해서는 안 됨!
    - 쿠다는 블록 내 스레드간 동기화를 제공하지만
    - 다른 블록의 스레드 간 동기화를 가능하게 하는 명시적인 작업을 제공하지 않음
      - 컨텍스트 스위칭을 지원하는건 너무많은 저장 공간과 시간을 필요로 함.

## Thread Scheduling

- 스레드는 SM에 실행되며 블록단위로 동시에 실행됨.
  - SM에는 실행을 관리하는 스케줄러가 있음.
- 블록이 할당되어 스케줄되면 워프(warp)라고 불리는 단위로 더 나뉨.

### Warps

- Warp는 SM에서 스케줄링의 기본단위임.
  - Warp의 사이즈는 device마다 다르게 설정됨. 현재까지는 항상 32개였음.
- 워프안의 스레드들은 SIMD 모델에 따라 함께 스케줄되고 실행됨.
  - Single Instruction, Multiple Data
  - 하나의 명령어가 워프 내의 모든 스레드에 의해 fetch되고 실행됨
  - 하지만 각 스레드는 다른 데이터를 처리함

### 왜 SIMD를 사용하는가?

- 장점
  - 여러 실행 장치 또는 코어에 걸쳐 동일한 명령어 fetch/dispatch를 공유함.
    - 제어 장치 비용을 많이 줄여 더 많은 실행 장치에 분산함.
- 단점
  - 서로다른 스레드가 서로 다른 실행 경로를 택할 수 있음 - 제어 분기(control divergence)
    - 워프는 각 유니크 실행 경로에 대해 한번씩 통과해야함. 그런데 모든 명령은 함께 유지되어야함.
    - 결과로 각 경로에서 스레드가 경로를 실행하는동안 나머지는 비활성화됨.
  - SIMD 효율성(SIMD efficiency): SIMD가 실행 되는 동안 활성화된 코어/스레드의 비율

## Latency Hiding (지연시간 숨기기)

- 워프가 긴 지연 작업시간이 필요하면, 실행 준비된 다른 워프를 선택하고 스케줄링한다.
  - 워프레벨에서 멀티스레딩이 일어나고 있음. 파이프라인이 지연되는 대신 항상 일을 하게함.
  - 컨텍스트 스위칭이 아님. (레지스터를 메모리에 넣고 빼고 하는 일이 없음)
    - SM에 할당 된 모든 스레드들은 모두 레지스터가 예약되어 있고, 메모리가 예약되어 있음
    - 하지만 이 모든 것이 코어에서 동시에 실행되는 것은 아님
- 긴 지연 작업을 숨기기 위해 충분한 작업이 가능하도록 많은 워프가 필요함
  - 실행 준비가 된 워프를 찾을 확률이 더 높기 때문에
- 이러한 이유로 SM은 일반적으로 코어 수보다 훨씬 더 많은 스레드를 지원함
  - SM은 충분한 register, 메모리, 스레드 슬롯이 있음
- V100에서 SM Resources
  - Cores per SM 64
  - Max threads per SM 2048
  - Registers per SM 64K
  - Shared memory per SM 96KB
  - Max threads per SM은 cores per SM에 비해 매우 큼.

## Occupancy (점유율)

- SM에서의 점유율(occupancy)는 최대치에 비해 SM에 할당된 warp/스레드의 비율을 의미함.
- 일반적으로 지연시간을 숨기기 위해 점유율을 최대화하고 싶음.
  - 어떤 경우에는 낮은 점유율이 바람직할 수 있음.
  
### Occupancy Constraints

- SM당 최대 블록 수, SM당 최대 스레드 수, 블록당 최대 스레드 수를 따져야 함.
- 블록크기를 어떻게 선택해야 할까? 최대 32로 제한됨.
    - V100 기준으로 1024 threads/block, 2048 threads/SM, 32 blocks/SM을 허용함
      - block당 256 threads를 사용할 경우
        - 최대 허용량 2048 threads / (256 threads/block) = 8 blocks로 32블록보다 작음
        - 따라서 occupancy를 최대화할 수 있음.
      - block당 32 threads를 사용할 경우
        - 최대 허용량 2048 threads / (32 threads/block) = 64 blocks로 32블록보다 큼
        - 따라서 occupancy를 제한될 것임.
      - block당 768 threads를 사용할 경우
        - 최대 허용량 2048 threads / (768 threads/block) = 2 blocks로 32블록보다 작음
        - 하지만 2048이 768로 나누어 떨어지지 않으므로 나머지 512를 사용할 수 없음.
        - 따라서 occupancy를 제한될 것임.

### Querying Available Resources

- 쿠다에 device의 사용가능한 resource를 query할 수 있음.
```c
    cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device);
```
- example:
```c
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max threads per multi processor: %d\n", prop.maxThreadsPerMultiProcessor);
```
