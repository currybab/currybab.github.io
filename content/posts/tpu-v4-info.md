+++
title = 'TRC TPU v4 팁?'
date = 2024-03-18T00:31:04+09:00
tags = ['tpu', 'trc']
categories = ['tpu']
+++

## TPU v4 팁 (아닐 수도)

최근 구글 [TPU Research Cloud](https://sites.research.google/trc/about/)를 통해 TPU를 지원 받아 훈련을 진행 중이다.
물론 구글에 방대한 문서로 설명을 해주고 있긴하나 너무 많아서 뭔가 버튼을 눌렀을때 두려움이 있기 때문에 기본 설정과 관련된 글을 좀 적어보고자한다.

### preemptible vs on-demand

제공 받을 수 있는 tpu의 일종의 지불 형태라고 볼 수 있다. on-demand는 우리가 일반적으로 클라우드에서 기대하는 필요시에 리소스를 차지하는 방식이고 preemptible은 선점형 방식으로 저렴한 대신에 해당 리전에 사용량이 많으면 반환되는 형태라고 파악하고 있다. on-demand는 편안하게 선택하거나 명령어를 치면 되지만 preemptible 같은 경우에는 cli로 할때는 --preemptible을 추가하고 웹에서 할때는 선점형에 대한 체크를 꼭 해야 한다.

### v4 vs v3 vs v2

당연히 구조에서 차이가 난다. 하지만 내가 느끼는 가장 현실적인 차이는 우선 tpu v4는 us-central-2b에만 존재하고 또한 us-central-2b에는 tpu v4 옵션 밖에 없는 듯하다. 잘은 모르지만 pricing calculator에 별도의 해당 리전이 없는 것으로 보아 tpu v4는 개인에게는 trc 프로그램을 통해서만 제공하는게 아닐까 추측만 한다.

이글을 적는 시점에는 gcp tpu에서 preemptible로만 v2와 v3가 주어져서 딱히 v2와 v3를 켜보지는 않았다. 하지만 이전에 캐글이나 코랩에서 각각 v3-8과 v2-8을 써볼 수 있었는데 이때에는 라이브러리 지원상황 같은 것들이 차이가 낫던 것으로 기억한다.

### tpu v4

기본적으로 tpu v4는 accelerator type이 v4-8부터 가능하다. v4-8에서 8은 tensor core의 갯수로 tpu v4 칩 하나에는 2개의 tensor core로 이루어져 있기 때문에 v4-8은 4개의 칩이 연결된 하나의 tpu pod이다. 또한 tpu pod 하나당 메모리가 HBM2 32GiB라고 한다. v4-8을 켰을 경우에는 하나의 tpu pod만이 실행되기 때문에 ssh로 접속해서 작업하기는 편했다.

v4-16을 켜게 되면 총 8개의 칩을 켜게 된다. 하나의 pod에는 4개의 칩으로 이루어져 있기 때문에 이 경우 두 개의 tpu pod이 실행된다. 각각의 pod이 개별 작업자로써 동작하기 때문에 함께 분산 처리를 할 수 있도록 같은 명령을 보내주기 위해서 `gcloud compute tpus tpu-vm ssh user@tpu-test --worker=all --command "echo 'hihi'"` 와 같이 모든 worker에게 명령어를 전달할 수 있도록 해야 한다.

이론적으로 32개의 칩을 배정 받았으면 v4-64가 켜질 수 있는 듯한데 region에 남아있는 resource가 부족해서 나는 아직까지 켜보지는 못했다.

### 간단한 후기

개인적으로 소유하고 있는 GPU가 딱히 없어서 요렇게 tpu를 통해서 비교적 대규모 학습을 할 수 있는 기회가 아주 귀한 것 같다. 앞으로도 꾸준하게 부지런히 빌려써보고 싶다. 또 개인적으로 잘 학습한 모델들을 공개하고 싶다.
