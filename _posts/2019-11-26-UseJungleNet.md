---
layout: post
title: cleos로 정글넷에 컨트랙트 올리기
categories: EOS
tags: blockchain study EOS
---

이전까지는 로컬에서만 작업했는데, 어쨌든 실제와 가까운 환경에서 사용하려면 퍼블릭 테스트넷에서 작동시킬 필요가 있다.
따지고 들어가면, 테스트넷의 트랜잭션보다는 메인넷의 트랜잭션이 훨씬 많기 때문에 이것도 아닐 수도 있다.

어쨌든 정글넷 2가 이제 막 런칭을 하였고, 어쩌다보니 나도 짜고 있는 컨트랙트를 테스트넷에 올려야할 계기가 생겨서 컨트랙트가 매우 초창기 단계이지만, 올려보고, 액션을 날리고, 디비에 잘 저장되었는지를 확인해보고자 한다.

### 계정 생성 및 정글 테스트넷 연결
<https://monitor.jungletestnet.io/>에 들어가면 계정생성, faucet 등을 해결할 수 있다. 임시로 내 컨트랙트 이름을 totagamelist라 만들었고 faucet을 통해 100 EOS를 전달 받았다. 상단 메뉴의 account info를 누르면 다음과 같이 100EOS를 전달 받았음을 확인할 수 있다.

![account info image](https://user-images.githubusercontent.com/7679722/48989264-d0ffff00-f16c-11e8-92b1-533070fa1a47.png)

이렇게 대략적으로 계정 생성을 마무리 했다.

이제 cleos를 로컬에 연결하는 작업을 해보자. 나같은 경우에는 docker를 통해 eosio/eos @latest를 설치한 상태이다. 기존 블로그와는 조금 다르게 `docker run --rm --name eosioJungle -d -v ~/codingcoding/eosContracts:/eosContracts eosio/eos /bin/bash -c 'keosd --http-server-address=localhost:8888'`를 실행하여 docker container를 만들어 주었다. 여기에서 중요한 것은 지갑을 관리하는 파일인 keosd를 localhost:8888 포트를 사용해 열었다는 것과 내 로컬 컨트랙트 파일의 경로인 ~/codingcoding/eosContracts를 docker 내부에선  /eosContracts에 저장해 주었다는 것 정도?

그리고 나서 alias cleos를 변경하였는데 `alias cleos='docker exec -it eosioJungle /opt/eosio/bin/cleos --wallet-url http://localhost:8888 -u https://jungle2.cryptolions.io:443'`와 같이 변경하였다. `https://jungle2.cryptolions.io:443`이 우리가 통신할 bp의 url이라고 보면 되겠다. 정글넷 홈페이지의 Api Endpoint라는 목록을 누르면 다음과 같이 목록이 뜬다

![API Endpoint List](https://user-images.githubusercontent.com/7679722/48989523-ffcaa500-f16d-11e8-926d-5243e292c28f.png)

그냥 저중에서 뭔가 제일 신뢰가는 것 같이 보이는 것을 선택했다.(느낌상으로다가.....)

`cleos get accounts $publicKey`를 실행하면 내가 만든 계정을 확인할 수 있을 것이다.(잘 연결이 된것이 맞다면..!)


### Contract 배포하기 및 동작 확인
이제 계정 생성도 마쳤고, 정글넷에 연결하는것도 되었으니 컨트랙트를 세팅할 시간이다. 방법은 로컬에서 진행하는 것과 당연히 같다!!
```
cleos set contract totagamelist /eosContracts/tota/ -p totagamelist@active
```
너무 내 기준에서 작성한것이라... totagamelist 대신 여러분의 컨트랙트 계정명, /eosContracts/tota/ 대신 docker에 있는 내 컨트랙트의 폴더를 넣어주면 될것이다.

컨트랙트를 정글넷에 올리는데 성공했으면 push action과 get table을 이용해 db를 확인해보자. 이경우 순전히 컨트랙트의 action과 table을 구성하기 나름이므로... 각자 자신의 컨트랙트를 보고 하길 바란다... 나의 예제는 다음과 같다...
```
cleos push action totagamelist insertgame '["totagamelist", "testgame2", 0, 4, 5, 6, "totaproxyno1", "totaproxyno2"]' -p totagamelist@active
cleos get table totagamelist totagamelist games
cleos get table totagamelist totagamelist games --lower 2
```

<https://developers.eos.io/eosio-cleos/reference#cleos-get-table>에 가면 더 다양한 검색방법을 찾을 수 있다.

### DEBUG
1. 최초에 지갑이 생성 되어 있지 않을 것이기 때문에 `cleos wallet create --to-console`을 통해 지갑을 생성해준다. 여기서 나오는 비밀번호는 이 컨테이너에서 해당 지갑의 비밀번호이기 때문에 신경써서 저장해준다. 이 후 `cleos wallet import`명령어를 통해 내가 사용할 private key를 등록해 준다.

2. 컨트랙트를 setting하는 과정에서 램이 부족하다고 나는 뜨더라.... 아마 같은 확률일 것이다. 최초에 ram이 많지 않기 때문에... `cleos system buyram totagamelist totagamelist '5.0000 EOS'` 이런 식으로 입력하면 쉽게 5이오스를 해당 계정에 살 수 있다.
