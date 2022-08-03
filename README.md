# tensorflow object detection api를 활용한 독서실 유무 파악기

## 데이터 확보 및 전처리
### 1. 데이터 확보
좌석에 앉지 않은 사진 50장, 좌석에 앉은 사진 50장을 직접 도선관에 가서 촬영하였다.

### 2. 데이터 전처리
 Learning Data Augmentation Strategies for Object Detection을 인용하여서 테스트에대한 정확도 증감을 보였던 brightness,Horizontal flip, zoom,equlize_histogram 4가지를 활용하여서 1000장으로 데이터를 증감시켰고 640으로 사이즈를 조정해 주었다.
 
### 3. 데이터 라벨링
데이터의 이미지에서의 빈자리유무에 대한 라벨링을 직접 해주었다.

## 학습
### 1. 학습전 준비
tensorflow object detection api는 모델을 직접 만드는것이 아닌 학습된 모델을 재학습하여서 사용하는것 이기에 클래스 명에 따른 id 값을 부여해 주고 TFrecord의 형태로 데이터셋의 데이터를 묶어 주었다.

### 2. 모델 선택
모델은 efficientnet01을 사용하였고 batch_size는 2로 epoch는 20000을 수행해 주었다. 
efficientnet은 인공지능에서 세가지의 정확도가 좋은 방식을 사용하여서 만든 모델로 첫 번째로 모델의 networ의 깊이를 깊게 만들었고, filter의 개수를 늘렸으며, input 이미지의 해상도를 올려서 학습 시킨 모델이다.

## 테스트
테스트는 구글에서 있던 사진들 몇장과 미리 테스트를 위해서 빼두었던 이미지를 가지고 진행했다.\



### 자세한 사항은 인공지능및 설계 최종본 ppt 와 인공지능및설계 프로젝트 보고서에 상세히 제시되어 있다.
