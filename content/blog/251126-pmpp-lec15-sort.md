+++
title = "Pmpp Lec15 Sort"
date = "2025-11-26T16:26:03+09:00"

#
# description is optional
#
# description = "An optional description for SEO. If not provided, an automatically created summary will be used."

tags = ["gpu", "cuda", "pmpp"]
+++

Source: [Lecture 15 - Sort](https://www.youtube.com/watch?v=XTfH6Ll9KaA)

### Today

- Parallel patterns: sort
  - Radix sort
  - Merge sort

## Radix sort (기수 정렬)

- 기수 정렬은 기수(radix)를 기반으로 정렬할 키를 버킷에 배분하는 정렬 알고리즘.
  - 입력 키는 위치 기수법의 특정 기수로 표현 됨.
- 키를 버킷에 분배하는 작업은 각 자릿수에 대해 반복됨.
  - 한 가지 중요한점은 각 버킷 내에서 이전 반복의 순서를 유지해야 한다는 것.
- 보통 기수 정렬을 할때 2의 거듭제곱인 기수를 사용하는 것을 선호
  - 이진 수 처리를 단순화 하기 때문에.
  
### Radix sort Iteration (1bit)

- 입력 배열의 각 요소의 destination index를 어떻게 찾는가?
- 0 값의 위치:  
  `왼쪽에 있는 0의 갯수`  
  `= 왼쪽에 있는 요소의 갯수 - 왼쪽에 있는 1의 갯수`  
  `= element index - 왼쪽에 있는 1의 갯수`

- 1 값의 위치:  
  `전체 0의 갯수 + 왼쪽에 있는 1의 갯수`  
  `= (전체 요소의 갯수 - 전체 1의 갯수) + 왼쪽에 있는 1의 갯수`  
  `= input size - 전체 1의 갯수 + 왼쪽에 있는 1의 갯수`
- 찾아야 할것은: 각 요소의 왼쪽에 있는 1의 갯수임. -> exclusive scan을 활용함.

![finding destination index](https://img.buidl.day/blog/sort-finding-destination-index-2.png)

### Parallelizing Radix Sort

- 입력 배열의 각 요소에 스레드 하나를 할당해서 자신이 담당하는 요소에 0 또는 1을 추출함.
- 배타적 스캔을 수행하고 나서 index를 계산함.
- 스레드가 담당하는 요소의 목적지에 복사함.

### Optimizing Memory Coalescing
- 저장이 coalesced되지 않음 (poor memory coalescing)
- 전역 메모리에 0과 1을 바로 쓰는 대신에 shared 메모리에서 0과 1을 분리해서 정렬한 후 복사하면 연속된 0과 1을 쓸 수 있음.

![optimizing memory coalescing](https://img.buidl.day/blog/sort-optimizing-memory-coalescing.png)

## Choice of Radix Value

- 지금까지 1 bit 기수를 사용했었음.
  - N bit 기수를 사용할 경우 N번의 iteration이 필요함.
- 더큰 기수를 사용할 수 있음.
  - 장점: iteration이 적어짐. (더 빨리 끝날 수도 있음.)
  - 단점: 더 많은 버킷을 가짐. 
    - 공유 메모리에서 전역 메모리로 써야 할 버킷이 더 많음. 따라서 더 많은 개별 메모리를 가짐
    - 결과적으로 coalesced memory access가 더 어려워짐.
- 성능에 맞춰서 잘 선택해야함.

## 2-bit Radix Sort Iteration

- 2 bit radix 사용시 N/2번의 iteration이 필요함.
- 처음에 스레드 블록 내에서 2 bit radix를 사용하여 sort함.(sort locally)
  - 가장 좋은 방법은 1bit step을 연속적으로 내부에서 쓰는 것임.

![2-bit radix sort iteration](https://img.buidl.day/blog/radix-sort-2bit-iteration.png)

## Thread Coarsening

- 더 큰 기수를 선택함으로써 iteration이 줄어들지만, coalesced memory access가 더 어려워짐.
- 더 많은 블록에 걸쳐 병렬화하는데 드는 비용은 블록당 버킷이 줄어들고, 따라서 coalescing할 기회가 더 줄어듬
- 한 블록에 더 많은 요소를 처리하게 되면 블록당 더 많은 버킷을 처리하게 되고 coalescing 측면에서 좋아짐.

![radix sort with thread coarsening](https://img.buidl.day/blog/radix-sort-with-thread-coarsening.png)

## Merge Sort

- radix sort는 모든 종류의 키에 대해서 적용할 수는 없음.
- 병렬화에도 적합한 일종의 비교 기반 알고리즘이 필요함.
- 초기 단계는 merge 연산들 간의 병렬성에 더 의존하고, 후기 단계는 merge 연산 내부의 병렬성에 더 의존함.

![merge sort](https://img.buidl.day/blog/merge-sort.png)
