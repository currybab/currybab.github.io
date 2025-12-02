+++
title = "Pmpp Lec23 Potpourri"
date = "2025-12-02T22:09:43+09:00"

#
# description is optional
#
# description = "An optional description for SEO. If not provided, an automatically created summary will be used."

tags = ["pmpp", "cuda", "gpu"]
+++

Source: [Lecture 23 - Potpourri](https://www.youtube.com/watch?v=wCyNd662aic)

### Today

- Multi-GPU programming
- Interconnect
- Memory management
- Events
- Tensor cores
- Libraries
- Other programming interfaces
- Other hardware
- Comparison with CPU

## Multi-GPU Programming

- 같은 노드에 여러 GPU가 있을 수 있음.
- CPU가 여러 GPU에 연결 되어 있다면 CPU는 각 GPU에 무엇을 해야 할지 알려주고 GPU 전체 작업을 조율해야 함.
    - `cudaGetDeviceCount(...)`를 사용하여 GPU의 수를 확인할 수 있음.
    - `cudaSetDevice(...)`를 사용하여 GPU를 선택할 수 있음.
    - 보통 각 GPU마다 스레드를 실행하여 CPU에 여러 스레드를 사용함.
    - GPU 전반에 걸친 무언가도 할수 있음.
- 여러 노드에 걸친 여러 GPU를 연결 할 수도 있음.
    - 여러 GPU뿐만 아니라 전체 작업을 조율하고 싶을것 .
    - 보통 MPI라는 것을 사용함.
        - 네트워크를 통해 여러 노드를 연결하여 여러 GPU를 사용할 수 있음.
        - MPI rank를 사용하여 각 노드를 구분함.
    - 여러개의 GPU를 사용할때 중요한 고려 사항은 네트워크를 통한 통신과 GPU에서의 연산을 어떻게 overlap할것인가. 

### Interconnect (상호 연결)

- PCIe는 많은 시스템에서 CPU와 GPU를 연결 하는 수단임.
- NVLINK는 더 빠른 상호연결을 제공함. 여러 GPU를 서로 연결 하는데 사용되고 지원되는 CPU에도 연결이 가능함


## 통합 가상 주소 지성 (Unified Virtual Addressing)

- CPU 메모리와 GPU 메모리가 물리적으로 구분되어 있지만 같은 가상 주소를 사용하게 함.
- 장점: 데이터의 위치는 포인터의 값을 보고 알 수 있기 때문에, 복사할 때 방향을 지정할 필요가 없어짐. 
    - 그냥 `cudaMemcpyDefault`를 사용하면 됨.

## Zero-copy memory

- 디바이스 스레드가 호스트 메모리에 직접 접근함.
- GPU에서 실행되는 스레드가 CPU에 있는 메모리 주소 범위에 접근하려고 할때, 런타임 시스템이 필요에 따라 해당 데이터를 CPU에서 GPU로 복사하여 스레드에 제공함.
- 따라서 명시적으로 `cudaMemcpy`를 사용할 필요가 없음.
- 제로카피를 사용하려면 메모리가 고정되어야 함. 
    - `cudaHostAlloc(&ptr, size, cudaHostAllocMapped)`를 사용
    - `cudaHostGetDevicePointer(...)`를 사용하여 GPU에서 이 데이터에 접근하는데 사용할 수 있는 해당 포인터를 얻음
        - UVA를 시스템이 지원하면 불필요함.

## Events

- 우리는 타이밍 정보를 수집하기 위해 모든 복사 및 커널 호출 후에 동기화해왔음.
    - 현실에서 이 접근 방식은 실행을 방해하고 모든 것을 느리게 만듬.
- 동기화 없이 타이밍 정보를 수집할 수 있는 방법은 이벤트를 사용하는 것.
    - 스트림에서 시작되거나 끝날때마다 타임스탬프를 수집하기 위해 스트림에 이벤트를 추가하면 됨.
    - `cudaEventCreate(&event)`
    - `cudaEventRecord(event, stream=0)`
    - `cudaEventSynchronize(event)` :  과거의 특정 작업만 동기화하기 위해서 사용할 수 도 있음.
    - `cudaEventElapsedTime(&result, startEvent, endEvent)`
    - `cudaEventDestroy(event)`
- 프로파일러도 사용할 수 있음.

## Tensor Cores

- 텐서코어는 볼타 아키텍쳐부터 소개된 프로그래밍 가능한 행렬 곱 연산 유닛임.
- 각 SM에는 작은 행렬을 수행하는 데 사용할 수 있는 텐서 코어가 있음.
    - DNN 워크로드를 가속하는데 중요함.
    - 텐서코어를 활용하면 쿠다코어보다 더 행렬 곱을 빠르게 수행할 수 있음.
- 볼타 V100에서는
    - 640개의 텐서코어가 있음(SM마다 8개)
    - 각 코어는 4x4 행렬곱을 수행함. `D = A*B + C`

## Libraries

- CUDA의 일부이거나 CUDA와 함께 널리 사용되는 많은 라이브러리가 있음.
- Thrust: reduction, scan, filter, sort
- cuBLAS: 기본 선형 대수학 연산 (basic dense linear algebra)
    - NVBLAS: 여러 GPU에서 밀집 대수 연산을 수행할 수 있는 다중 GPU 라이브러리
- cuSPARSE: sparse-dense linear algebra
- cuSOLVER: 인수분해및 풀이 루틴에 유용
    - cuBLAS와 cuSPARSE 상에서 구현됨.
- cuDNN: deep neural networks
    - Caffe, MxNet, TensorFlow, PyTorch와 같은 DNN 라이브러리에서 사용됨.
- nvGraph: 그래프 처리
- cuFFT: fast fourier transform
- NPP: 모든 종류의 신호, 이미지, 비디오 처리에 유용.

## Comparison with CPU

- 강의에서 cpu code를 사용할 때 싱글 스레드, 벡터화 되지 않은 코드만을 사용함.
    - GPU의 성능을 과장할 수 있음.
    - 이러한 관행은 지양됨.
- GPU와 CPU를 공정하게 비교 싶다면 병렬화되고 벡터화된 CPU 코드를 사용하는 것이 좋음.
