+++
title = "Pmpp Lec17 Sparse Matrix Computation (ELL and JDS)"
date = "2025-11-28T20:57:32+09:00"

#
# description is optional
#
# description = "An optional description for SEO. If not provided, an automatically created summary will be used."

tags = ["pmpp","gpu","cuda"]
+++

Source: [Lecture 17 - Sparse Matrix Computation (ELL and JDS)](https://www.youtube.com/watch?v=bDbUoRrT6Js)

### Today

- Parallel patterns: sparse matrix computation
    - Case study: sparse matrix-vector multiplication (SpMV)
    - Storage formats:
        - ELL (Ellpack Format)
        - JDS (Jagged Diagonal Storage Format)

## ELLPACK Format (ELL)

- CSR처럼 행별로 nonzero 값들을 묶음.
- 각 행을 모두 같은 사이즈를 같도록 padding함.
    - 각 행이 고정된 양의 메모리가 할당되도록 함.
- 그리고 nonzero 값들을 column major order로 저장함.

![ell concept](https://img.buidl.day/blog/ell-1.png) 
![ell array](https://img.buidl.day/blog/ell-2.png)

### SpMV/ELL

- CSR에서 사용한 방식과 동일한 방식 이용.
- 각 입력행을 순차적으로 반복하도록 하고 해당 출력 요소를 업데이트 함.
    - 값과 열을 접근하는 모양을 보면 coalesced가 됨.

```c
    struct ELLMatrix {
        unsigned int numRows;
        unsigned int numCols;
        unsigned int maxNNZPerRow;
        unsigned int* nnzPerRow;
        unsigned int* colIdxs;
        float *values;
    }
    __global__ void spmv_ell_kernel(ELLMatrix ellMatrix, const float* inVector, float* outVector) {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < ellMatrix.numRows) {
            float sum = 0.0f;
            for (int iter = 0; iter < ellMatrix.nnzPerRow[row]; iter++) {
                unsigned int i = iter * ellMatrix.numRows + row;
                unsigned int col = ellMatrix.colIdxs[i];
                float value = ellMatrix.values[i];
                sum += value * inVector[col];
            }
            outVector[row] = sum;
        }
    }
    // csr버전에 비해 더 나은 coalesced 접근으로 성능이 증가함 (0.386mx -> 0.295ms)
    // 복사시간이 길어짐(패딩 때문에 저장공간이 더 큼)
```

### ELL Tradeoffs

- 장점:
    - Flexibility: row가 채워지지 않았다면 새 요소를 추가할 수 있음
    - Accessiblity: 
        - 행이 주어지면, 모든 nonzero 요소에 접근할 수 있음(numRows에 대한 나머지 값으로)
        - nonzero 값이 주어지면, 행과 열을 알 수 있음.
    - SpMV/ELL 메모리 접근이 coalesced 함.
- 단점:
    - Space efficiency: padding으로 저장공간이 더 큼
    - Accessibility: 열이 주어지면 nonzero 요소를 찾기 힘듬.
    - SpMV/ELL 연산시 control divergence가 발생함.

## Hybrid ELL + COO

- 어떤 행이 전부 0이 아닌 값을 들고 있을 경우에 큰 패딩으로 문제가 발생함.
- 대부분의 행이 그 숫자보다 작은 어떤 숫자를 ELL 표현의 너비로 사용하고 그 숫자를 초과하는 행에 대해서는 COO 표현을 사용함.

![hybrid ell + coo](https://img.buidl.day/blog/ell-coo-hybrid.png)

### ELL + COO Tradeoffs

- ELL과 유사함, COO사용시 추가적 이득이 있음.
    - Space efficiency: 더 적은 패딩
    - Flexibilty: 어떤 행에도 새 element를 넣을 수 있음.

## Jagged Diagonal Storage (JDS)

- 제어 분기를 최적화하는 커널
- 행의 0이 아닌 값들을 모두 함께 묶음

![jds row collect](https://img.buidl.day/blog/jds-1.png)

- 행의 사이즈 내림차순으로 정렬하고, 원래 행의 인덱스를 따로 저장함. (control divergence 해결)

![jds row sort](https://img.buidl.day/blog/jds-2.png)

- coalescing 접근을 최적화한다는 것은 값이 쓰레드 작업에 대해 연속적으로 저장되기를 원한다는것을 말함.
- column major 방향으로 nonzero 값들을 저장함. (ELL 처럼)

![jds column major](https://img.buidl.day/blog/jds-3.png)

- 쓰레드가 돌아갈때 각 반복이 어떤 index에서 시작하는지 알 수 있도록 별도로 저장해야 함.

![jds storage](https://img.buidl.day/blog/jds-4.png)

### SpMV/JDS kernel

- 한 스레드가 각 행의 모든 nonzero 요소를 순차적으로 접근할 수 있도록 함.
- 해당하는 output 요소를 업데이트 함.

### JDS tradeoffs

- CSR 만큼은 아니지만 padding이 없기 때문에 ell 보다 공간 효율적임.
- 요소를 추가하고 제거하는 것은 매우 어려움. flexibility가 없음.
- 행을 주면 행의 모든 0이 아닌 값을 찾기 쉬움.
- 0이 아닌 값이 주어졌을때 row를 찾기 힘듬, column을 주면 해당 열을 찾기 힘듬.
- 메모리 접근이 coalesced 함. 쓰레드가 끝에서부터 drop out 되기 때문에 control divergence도 최소화함.
- 소개한 모든 커널 중에서 가장 성능이 좋음.
