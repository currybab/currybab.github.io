+++
title = "CuTe DSL 개념 정리"
date = "2025-12-13T22:50:47+09:00"
math = true
#
# description is optional
#
# description = "An optional description for SEO. If not provided, an automatically created summary will be used."

tags = ["cutedsl", "cuda", "gpu"]
+++

## Tensor

cute에서 텐서는 Engine과 Layout으로 구성됨. $ T = E \circ L $

- Engine: pointer 같은 존재이며 다음 연산을 지원함.
    - offset 연산 : $ e+d \rightarrow e $
    - dereference 연산 : $ *e \rightarrow v $
- Layout
    - 레이아웃을 따라 좌표 c를 매핑하고 엔진을 따라 오프셋함, 결과를 역참조하여 텐서의 값을 획득함.
- $ T(c) = (E \circ L)(c) = *(E + L(c)) $
- DLPACK 프로토콜을 지원하여, 이를 지원하는 torch, jax, numpy와 쉽게 연동 가능. `from cutlass.cute.runtime import from_dlpack`

### Tensor Evaluation

1. full evaluation: 특정 요소를 접근하는데 사용함.
2. partial evaluation (slicing)
- 불완전 좌표 $ c = c^{\prime} \oplus c ^{\ast} $ (여기서 $c^{\ast}$는 지정되지 않은 부분을 나타냄)로 평가할 때, 결과는 제공된 좌표를 표현하기 위해 엔진 오프셋이 적용된 원본 텐서의 슬라이스인 새 텐서임.
  $$ T(c) = (E ∘ L)(c) = (E + L(c^\prime)) ∘ L(c^\ast) = T^\prime(c^\ast) $$
- 슬라이싱은 텐서의 차원을 효과적으로 줄여 추가로 평가하거나 조작할 수 있는 서브 텐서를 생성

### 메모리 뷰로서의 텐서

- generic: 다른 모든 메모리 공간을 참조할 수 있는 기본 메모리 공간
- global memory (gmem): 모든 블록의 모든 스레드에서 접근할 수 있지만, 지연 시간이 더 깁음
- shared memory (smem): 한 블록 내의 모든 스레드에서 접근할 수 있으며, 전역 메모리보다 훨씬 낮은 지연 시간을 가짐
- register memory (rmem): 스레드 전용 메모리로, 지연 시간이 가장 짧지만 용량이 제한적
- tensor memory (tmem): NVIDIA Blackwell 아키텍처에서 텐서 연산을 위해 도입된 특수 메모리

## 좌표 텐서 (Coordinates Tensor)

- 좌표 텐서 $ T: Z^n \rightarrow Z^m $는 좌표 공간 간의 매핑을 설정하는 수학적 구조
- 스칼라 값에 좌표를 매핑하는 표준 텐서와 달리, 좌표 텐서는 좌표를 다른 좌표에 매핑하여 텐서 연산 및 변환을 위한 기본적인 구성 요소를 형성함.

### 항등 텐서 (Identity Tensor)

- 항등 텐서 $ I $는 항등 매핑 함수를 구현하는 좌표 텐서의 특수한 경우입니다.
- 정의: 주어진 형상 $ S = (s_1, s_2, \cdots, s_n) $에 대해 항등 텐서 $ I $는 다음을 만족함.
$$ I(c) = c, \ \ \forall c \in \prod_{i=1}^n [0, s_i) $$
- 속성
    - Bijective Mapping: 항등 텐서는 좌표 간의 일대일 대응을 설정함.
    - Layout Invariance(레이아웃 불변성): 기본 메모리 레이아웃과 관계없이 논리적 구조는 일정하게 유지
    - Coordinate Preservation(좌표 보존): 모든 좌표 c에 대해 $ I(c) = c $ 임.
- CuTe는 사전식 순서를 통해 1차원 인덱스와 N차원 좌표 간의 동형 사상을 설정함. 형상 $ S = (s_1, s_2, \cdots, s_n) $를 가진 항등 텐서의 좌표 $ c = (c_1, c_2, \cdots, c_n) $에 대해 선형 인덱스 공식은 아래와 같음.

$$ \text{idx} = c_1 + \sum_{i=2}^{n} \left(c_i \prod_{j=1}^{i-1} s_j\right) $$

`cute.make_identity_tensor(shape)` 양방향 매핑은 선형 인덱스에서 N차원 좌표로의 효율적인 변환을 가능하게 하여 텐서 연산 및 메모리 접근 패턴을 용이하게 함.
- 필요성
    - GPU 스레드 매핑: 각 스레드가 자신의 1D 인덱스만 알 때 Identity tensor로 N-D 좌표 얻어 N-D 좌표로 실제 데이터 접근
    - 타일링 연산
    - 복잡한 레이아웃 변환

## TensorSSA

- CuTe DSL에서 정적 단일 할당 (static single assignment) 형식의 텐서 값을 나타내는 클래스. (시뮬레이션 된) 레지스터에 있는 텐서라고 생각할 수 있음.
- 사용하는 이유
    - TensorSSA는 기본 MLIR 텐서 값을 Python에서 더 쉽게 조작할 수 있는 객체로 캡슐화함.
    - Python 연산자를 오버로드하여 사용자가 텐서 계산을 보다 Pythonic한 방식으로 표현할 수 있도록 함. 
    - 이러한 요소별 연산은 최적화된 벡터화 명령으로 변환됨.
    - 이는 CuTe DSL의 일부로, 사용자가 설명한 계산 논리와 하위 수준 MLIR IR, 특히 레지스터 수준 데이터를 표현하고 조작하는 것 사이의 다리 역할.

### SSA: Static Single Assignment
```
일반 코드:                    SSA 형태:
─────────────────────────────────────────────
x = 1                        x₁ = 1
x = x + 2                    x₂ = x₁ + 2
x = x * 3                    x₃ = x₂ * 3

↓ 각 변수가 딱 한 번만 할당됨 (불변)
SSA의 장점: 컴파일러가 최적화하기 쉬움 (데이터 흐름 추적이 명확)
```

### MLIR(Multi-Level Intermediate Representation)

```
Python 코드
    │
    ▼
┌─────────────────────────────────────────────┐
│                   MLIR                       │
│  (Multi-Level Intermediate Representation)  │
│                                             │
│   고수준 IR  →  중간 IR  →  저수준 IR        │
│   (텐서)       (루프)      (GPU 명령)        │
└─────────────────────────────────────────────┘
    │
    ▼
PTX / CUDA 코드
```
- MLIR은 구글이 만든 다단계 중간 표현으로, 코드를 점진적으로 낮은 수준으로 변환함. 

### TensorSSA의 역할

```python
# TensorSSA 없이 (저수준 MLIR 직접 조작)
op1 = mlir.arith.addf(tensor_a, tensor_b)
op2 = mlir.arith.mulf(op1, tensor_c)
result = mlir.math.exp(op2)

# TensorSSA 사용 (Pythonic!)
result = cute.exp((a + b) * c)
```
```
┌────────────────────────────────────────────────────────┐
│  Python 표현식        →    MLIR SSA 연산들             │
│                                                        │
│  (a + b) * c          →    %1 = arith.addf %a, %b     │
│                            %2 = arith.mulf %1, %c     │
└────────────────────────────────────────────────────────┘
```

### TensorSSA 사용 시나리오

1. 메모리에서 로드하고 메모리에 저장: `load()`, `store()`
2. 레지스터 수준 텐서 연산 - 커널 로직을 작성할 때, 레지스터에 로드된 데이터에 대해 다양한 계산, 변환, 슬라이싱 등이 수행.
3. 산술 연산
    - 이항 연산
        - 이항 연산의 경우, LHS 피연산자는 TensorSSA 이고 RHS 피연산자는 TensorSSA 또는 Numeric 가 될 수 있음. 
        - RHS가 Numeric 인 경우, TensorSSA 로 브로드캐스트.
    - 단항 연산
        - `cute.math.sqrt`, `cute.math.exp`, `cute.math.log`, `cute.math.sin`, `cute.math.cos` 등
    - 축소 연산
        - TensorSSA 의 reduce 메서드는 초기 값으로 시작하여 지정된 축소 연산( ReductionOp.ADD , ReductionOp.MUL , ReductionOp.MAX , ReductionOp.MIN )을 적용하고, reduction_profile 에 지정된 차원을 따라 이 축소를 수행함.
        - 결과는 일반적으로 차원이 축소된 새로운 TensorSSA 이거나, 모든 축에 걸쳐 축소되는 경우 스칼라 값.
        - `a_vec.reduce(cute.ReductionOp.ADD, 0.0, reduction_profile=0)`: 전체 축소 (스칼라)
        - `a_vec.reduce(cute.ReductionOp.ADD, 0.0, reduction_profile=(None, 1))`: 열 방향 축소 (행 유지)
4. Broadcast
    - TensorSSA 은 NumPy의 브로드캐스팅 규칙에 따라 브로드캐스팅 연산을 지원함. 
    - 브로드캐스팅을 사용하면 특정 조건이 충족될 때 서로 다른 모양의 배열에 대해 연산을 수행할 수 있음.
    - 규칙
        1. 소스 형상은 대상 형상의 랭크와 일치하도록 1로 채워짐. (차원수 맞춤)
        2. 소스 형상의 각 모드 크기는 1이거나 대상 형상과 같아야 함. (크기 맞춤)
        3. 브로드캐스팅 후, 모든 모드는 타겟 형상과 일치해야 함. (결과 형상 맞춤)

## Layout Algebra

### Layout

- Shape와 Stride의 쌍으로 정의
- 좌표 공간을 인덱스 공간으로 매핑하고, 정적(컴파일 타임) 및 동적(런타임) 값 모두를 지원
- CuTe는 레이아웃을 결합하고 조작하기 위한 강력한 연산 집합인 레이아웃 대수(Layout Algebra)를 제공
    - 레이아웃 구성: 레이아웃의 함수적 구성
    - 레이아웃 "나누기": 레이아웃을 두 개의 구성 요소 레이아웃으로 분할
    - 레이아웃 "곱하기": 다른 레이아웃에 따라 레이아웃을 재현

### Layout Algebra Operations로 가능한 것들

- 효율적인 데이터 타일링 및 분할
- 스레드 및 데이터 레이아웃을 모두 나타내는 표준 유형으로 분리
- 텐서 코어 프로그램에 필수적인 스레드 및 데이터의 계층적 텐서에 대한 기본 설명 및 조작
- 혼합 정적/동적 레이아웃 변환
- 레이아웃 대수와 텐서 연산의 완벽한 통합
- 복잡한 MMA 및 복사본을 정규 루프로 표현

### Coalesce (병합)

- `coalesce` 연산은 정수에 대한 함수로서 크기나 동작을 변경하지 않고, 가능한 경우 모드를 평면화하고 결합하여 레이아웃을 단순화
- 크기 보존: `cute.size(layout) == cute.size(result)`
- Flattened: `depth(result) <= 1`
- 기능 보존: 모든 i에 대해, `0 <= i < cute.size(layout), layout(i) == result(i)`

### Composition (결합)

- 레이아웃 A 의 Composition 과 레이아웃 B 가 결합하여 새로운 레이아웃 $R = A ◦ B$ 을 생성.
- B 의 모양은 R 의 모양과 호환되므로 B 의 모든 좌표를 R 의 좌표로도 사용 가능함. (`R shape == B shape`)
- B 의 도메인에 있는 모든 좌표 c 에 대해 $R(c) = A(B(c))$
- 레이아웃 결합은  레이아웃을 재구성하고 재정렬하는데 유용함.
- by-mode composition
    - 특정 모드 레이아웃을 독립적으로 조작하려는 경우(예: 행 및 열) 특히 유용
    - CuTe의 맥락에서 모드별 구성은 레이아웃 또는 레이아웃 튜플이 될 수 있는 Tiler을 통해 사용할 수 있음.
    - Tiler 튜플의 리프는 대상 레이아웃의 해당 모드가 구성되는 방식을 지정하여 하위 레이아웃을 독립적으로 처리할 수 있도록 함.

### Composition 공식

```
  A ∘ s:d 계산법:
  1. A / d  →  stride d만큼 건너뛰기 (나누기)
  2. % s    →  shape s만큼만 유지 (모듈로)
```

- left-distributive: `A ∘ (2,3):(1,8) = (A ∘ 2:1, A ∘ 3:8)`
- 예시
```
  A = (4, 2, 3):(2, 1, 8)일 때, A ∘ 4:2 계산

  step 1:"첫 번째 모드에서 2개를 나눠라"

  Shape:  4/2 = 2  →  (2, 2, 3)
  Stride: 2*2 = 4  →  (4, 1, 8)

  결과: (2, 2, 3):(4, 1, 8)

  step 2: "앞에서부터 4개만 유지"

  2 × 2 = 4  ← 딱 맞음!

  Shape:  (2, 2, 1)  →  (2, 2)
  Stride: (4, 1, _)  →  (4, 1)

  결과: (2, 2):(4, 1)
```

### Division (Splitting into Tiles)

- CuTe의 Division 연산은 레이아웃을 타일로 분할하는 데 사용되며, 이는 스레드 또는 메모리 계층 구조에 걸쳐 데이터를 분할하는 데 특히 유용
- 압축, 타일, 평면 분할은 모드를 더 편리한 형태로 재배열할 수 있는 logical_divide 의 종류
```
  연산                    결과 shape                          특징
  ----------------------------------------------------------------------
  logical_divide    ((TileM,RestM), (TileN,RestN))   원본 모드 구조 유지
  zipped_divide     ((TileM,TileN), (RestM,RestN))   타일 / 나머지 분리
  tiled_divide      ((TileM,TileN), RestM, RestN)    타일 묶고 나머지 분리
  flat_divide       (TileM, TileN, RestM, RestN)     전부 평탄화 
```

### `logical_divide`

- 두 레이아웃에 적용될 때, logical_divide 은 레이아웃을 두 가지 모드로 분할함. 
- 첫 번째 모드에는 타일러가 가리키는 요소가 포함되고, 두 번째 모드에는 나머지 요소가 포함.
- 레이아웃과 Tiler 튜플에 적용될 때, logical_divide 은 Tiler 의 리프와 대상 레이아웃의 해당 모드에 자체적으로 적용됨. 이는 하위 레이아웃이 Tiler 내의 레이아웃에 따라 독립적으로 분할됨을 의미함.

$$ logical_divide(A, B) := A ∘ (B, B*) \ \ \ where \ \ B* = complement(B, size(A))$$

- complement: A가 선택하지 않는 나머지를 채워서 전체를 커버하는 레이아웃
```
  질문: 4:2를 어떻게 배치하면 24개를 다 채울까?

  4:2의 cosize = 8 (0~6 사용, 다음은 8부터)

  24개를 채우려면:
  ├── stride 내 구멍: 2개 (0,1 / 2,3 / 4,5 / 6,7 중 짝수만 선택하니까)
  └── 반복 횟수: 24 / 8 = 3번


  방법 1: 구멍 먼저 채우기
  ────────────────────────
  4:2가    : 0     2     4     6
  구멍     :    1     3     5     7
          └─ offset +1 하면 됨

  = 2:1 (2개, stride 1로 오프셋)


  방법 2: 그 다음 반복
  ────────────────────────
  블록0: 0~7   (offset 0)
  블록1: 8~15  (offset 8)
  블록2: 16~23 (offset 16)

  = 3:8 (3개, stride 8로 오프셋)
```

### Product (타일 복제)

- CuTe의 Product 연산은 하나의 레이아웃을 다른 레이아웃에 따라 재현하는 데 사용
- 첫 번째 모드는 원본 레이아웃 A
- 두 번째 모드는 타일러가 가리키는 요소가 포함된 모드
- 이는 "반복" 패턴을 생성하기 위해 데이터 타일 전체에 스레드 레이아웃을 반복하는 데 특히 유용
- `logical_product` 공식
$$ A ⊗ B := (A, A* ∘ B) \ \ \ where \ \ A* = complement(A, size(A) × cosize(B))$$
- Blocked Product: A와 B의 모드를 블록과 같은 방식으로 결합하여, 곱셈 후 재연관을 통해 모드의 의미론적 의미를 보존
- Raked Product: A와 B의 모드를 엇갈리거나 "갈퀴형" 방식으로 결합하여 타일의 순환 분포를 만듬
- 나누기 연산과 유사하게, zipped, tiled, flat product는 모드를 더 편리한 형태로 재배열할 수 있는 `logical_product` 의 한 종류

### Divide vs Product 차이
```
    logical_divide(A, B)                              
    ─────────────────────                             
    "A를 B로 나눈다"                                  
                                                    
    A가 전체 레이아웃, B가 타일 패턴                   
    → 결과: ((타일 내부), (타일 배치))                
                                                    
    size(결과) = size(A)                              


    logical_product(A, B)                                     
    ──────────────────────                                     
    "A를 B만큼 복제한다"                                        
                                                              
    A가 타일, B가 복제 패턴                                     
    → 결과: ((타일), (복제 배치))                               
                                          
    size(결과) = size(A) × size(B)                   
```
