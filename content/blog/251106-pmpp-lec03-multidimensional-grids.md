+++
title = "Pmpp Lec03 Multidimensional Grids and Data 요약"
date = "2025-11-06T21:39:17+09:00"

#
# description is optional
#
# description = "An optional description for SEO. If not provided, an automatically created summary will be used."

tags = ['pmpp', 'gpu']
categories = ['pmpp']
+++

source: [Lecture 03 - Multidimensional Grids and Data](https://www.youtube.com/watch?v=c8dehGOB8mQ&list=PLRRuQYjFhpmubuwx-w8X964ofVkW1T8O4&index=3)

### Today

- 다차원 그리드 만들기
- 다차원 데이터를 저장하고 접근하는법
- 예시 어플리캐이션
  - 컬리이미지 -> 흑백이미지화
  - 이미지 블러 
  - 행렬-행렬 곱셈
  
## RGB to Grayscale

- 이미지의 각 픽셀을 변환하도록 하나의 쓰레드를 할당.
- 쓰레드도 다차원 인덱스를 가지는게 편함.
  - CUDA는 3차원까지 지원함.
  - 다차원 데이터 처리를 단순화할 수 있어서 좋음.
- 각 픽셀의 색상이 하나의 쓰레드를 할당하는 법
  - 이 경우에 덧셈을 수행하기 위해 스레드가 서로 통신해야하는데 이정도 적은 작업량으로는 통신 오버헤드를 감당할 가치가 없음.
  - 1024개의 다른 채널이 있다면 그래야 할 수도 있음.
- multidimensional grid 설정하기
```c
    dim3 numThreadsPerBlock(32, 32); // 값이 안들어가면 1이 기본값
    dim3 numBlocks((width + 32 - 1) / 32, (height + 32 - 1) / 32); // numThreadsPerBlock.x 등을 사용해서 general하게 만들 수 있음
    rgb2gray_kernel<<< numBlocks, numThreadsPerBlock >>>(red_d, green_d, blue_d, gray_d, width, height);
```

### multidimensional indexing in grayscale

```c
    int row = blockIdx.y * blockDim.y + threadIdx.y; // y축은 높이
    int col = blockIdx.x * blockDim.x + threadIdx.x; // x축은 너비
    if (row < height && col < width) { // boundary check
        unsigned int idx = row * width + col; // 1차원 배열에 접근
        gray[idx] = red[idx] * 3 / 10 + green[idx] * 6 / 10 + blue[idx] * 1 / 10;
    }
```

## blur

- 이미지를 흐리게 하는 한가지 방법은 출력 픽셀을 해당 입력 픽셀의 주변 픽셀의 평균으로 설정하는 것
- 모든 출력 픽셀에 대해 스레드를 할당
  - 각 스레드는 해당 이벽 픽셀을 찾는 역할을 하고 주변 픽셀을 보고 평균을 계산함.
- 간단하고 좋은 방법이지만 더 나은 방법이 있음(나중에 배움)
```c
    __global__ void blur_kernel(unsigned char* image, unsigned char* blurred, unsigned int width, unsigned int height) {
        int outRow = blockIdx.y * blockDim.y + threadIdx.y;
        int outCol = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (outRow < height && outCol < width) {
            unsigned int average = 0;
            for (int inRow = outRow - BLUR_SIZE; inRow <= outRow + BLUR_SIZE; ++inRow) {
                for (int inCol = outCol - BLUR_SIZE; inCol <= outCol + BLUR_SIZE; ++inCol) {
                    if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                        average += image[inRow * width + inCol];
                    }
                }
            }
            blurred[outRow * width + outCol] = (unsigned char) average / ((BLUR_SIZE * 2 + 1) * (BLUR_SIZE * 2 + 1));
        }
    }
```

### boundary condition

- Rule of thumb: 모든 메모리 접근에는 인덱스를 배열 차원과 비교하는 해당 가드가 있어야 함.
- parallel programming의 절반은 index 계산과 boundary condition이라고 하심.

## Matrix-Matrix Multiplication
```
    C = A * B
```
- 행렬 C의 모든 요소는 A의 해당 행과 B의 해당 열의 내적임.
- parallelism을 극대화하는 방법은 C의 모든 출력 요소에 스레드를 할당하는 것.
```c
    __global__ void mm_kernel(float* A, float* B, float* C, unsigned int N) {
        unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
        unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (row < N && col < N) {
            float sum = 0.0f;
            for (unsigned int i = 0; i < N; ++i) {
                sum += A[row * N + i] * B[i * N + col];
            }
            C[row * N + col] = sum;
        }
    }
```
- 이보다 훨씬 더 빠른 기술들을 과정 내내 보게 될것임.
- 일반 행렬 곱셈이 3개의 루프를 가지고 있는데에 비해 한번만 진행함.
  - index가 루프 하나씩을 책임지고 있다고 보면 됨.
  - 남은 하나는 없애기 어려운데 여기에 루프 캐리 의존성(loop carry dependency)이 있기 때문임.
    - 이 루프를 통해 덧셈 연산을 하고 있음.
    - 이 루프를 없앨수 있냐하면 없앨수는 있음. 3차원 그리드를 선언하고 이 루프를 병렬화 할 수 있다고 함.
