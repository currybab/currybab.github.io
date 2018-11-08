---
layout: post
title: EOS CDT UPDATE 하기
categories: EOS
tags: blockchain study EOS
---

오늘? eosio.cdt 1.4.0 버전이 update 되었다.
그래서 업데이트 하려고 소스 폴더에 가서 `git pull`을 했더니 이것이 몬가??
왜 커밋해야될 파일이 생기지...???
해결 방법은 간단했다...!

```
git submodule update --init --recursive
```
git을 사용하기는 하지만 거의 회사 프로젝트와 개인플젝에서 아주 간단하게만 사용하고 있으니 사용법을 잘 모르겠다라...
암튼 이렇게 하고 나니 recursive하게 import된 서브 모듈까지 업데이트하고 깔끔하게 완료가 됬다...
그렇게 build를 하고 install을 하면 되는데......업데이트 사항을 봤더니 왠걸 homebrew를 통한 설치법이 나왔더라(1.3.0 부터 있었더라...)

그래서 오히려 uninstall을 하고....

```
brew tap eosio/eosio.cdt
brew install eosio.cdt
```
아 깔끔..<https://github.com/EOSIO/eosio.cdt>에 가면 리눅스 계열도 apt나 yum을 이용해 받는 방법이 적혀있다.

c++을 평소에 잘하지 않는지라 차마 여러모로 힘든거 같기도하당...
EOS battle은 1.2.0의 문법을 사용하는데 1.3.0에서 시작된 대격변으로 인해 문법이 너무 달라져서 안하기로 했다.
사실 너무 리액트 쪽 내용이 많기도 했다.
암튼 안할거다 ㅠㅠ

내가 어느정도 eos 개발을 시작해서 깨우치고 나면 그래두 1.2.0 -> 1.3.0의 대격변 상황이라도 정리해볼까한다..