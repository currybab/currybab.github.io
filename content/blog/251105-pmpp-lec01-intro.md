+++
title = 'pmpp lecture 01 요약'
date = 2025-11-05T19:42:14+09:00
tags = ['pmpp', 'gpu']
categories = ['pmpp']
+++

## lecture 01 요약

### design approaches

- latency-oriented design: minimize the time it takes to perform a single task
- throughput-oriented design: maximize the number of tasks that can be performed in a given time frame. 
- cpu는 latency-oriented design을 사용.
  - 적은 수의 강한 ALU를 사용.
  - 큰 캐시를 사용함(메모리 접근 시간을 최소화하기 위해)
  - 제어 속도를 늘이기 위해 많은 것들을 함. (out of order execution 등) - 여러 위험요소를 감지하기 위해 추가 장치 필요.
  - 파이프라인 지연을 숨기기 위해 무엇을 하는가.
    - 소프트웨어 적으로는 고급 컴파일러 기술들을 활용
    - 하드웨어 적으로는 적당한 양의 멀티 스레딩을 사용.(1코어당 2쓰레드 정도)
- gpu는 throughput-oriented design을 사용.
  - 많은 수의 작은 ALU를 사용(파이프라인화 하여 처리량을 향상)
  - 더 작은 캐시를 가짐. - 더 많은 영역을 연산에 할애할 수 있음.
  - 간단한 제어 로직을 가짐. - 더 많은 영역을 연산에 할애할 수 있음.
  - 높은 지연시간을 숨기기 위해 많은 양의 스레드를 허용함.

source: [Lecture 01 - Introduction](https://www.youtube.com/watch?v=4pkbXmE4POc&list=PLRRuQYjFhpmubuwx-w8X964ofVkW1T8O4&index=1&pp=iAQB)
