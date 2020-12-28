# 음성/음향 인식을 위한 딥러닝 기반 패턴인식기술 (with Tensorflow) 💻💬

<br/>

**통계/딥러닝 기반 인식기법 실습**
IDE download: [pycharm](https://www.jetbrains.com/pycharm/download/#section=windows)

<br/>

---

<br/>

### 📌 Dependencies

실험을 수행하는데 필요한 패키지를 정리한 파일입니다.

```console
pip install -r requirements.txt
```

<br/>

---

<br/>

### 0️⃣ 특징 추출 단계 수행

log-mel spectrum(.ls)과 MFCC(.mfc) 특징을 추출하여 저장

```console
python ./utils/feature_extractor.py
```

<br/>

### 1️⃣ 모델 훈련 (반드시 isTrain값을 1로 설정)

사용할 특징과 모델을 매개변수로 넣어 훈련 (default: mel / gmm)

```console
python train.py --isTrain 1 --data [특징타입] --model [모델]
```

- MFCC 특징을 사용하여 CNN모델을 훈련할 경우 아래와 같이 수행

```console
python train.py --isTrain 1 --data mfc --model cnn
```

<br/>

### 2️⃣ 모델 추론 (반드시 isTrain값을 0으로 설정)

사용된 특징과 모델을 매개변수로 넣어 테스트 (default: mel / gmm)

```console
python train.py --isTrain 0 --data [특징타입] --model [모델]
```

- MFCC 특징을 사용하여 훈련된 CNN모델의 성능을 평가할 경우 아래와 같이 수행

```console
python train.py --isTrain 0 --data mfc --model cnn
```

<br/>

### 🔄 모든 단계를 일괄적으로 수행

- 0~2단계를 선택하여 매개변수로 넣어 특징추출/모델훈련/모델추론 과정 수행
- 선택한 단계의 상위 단계 모두 수행 (ex. --step 0; 0~2단계 수행)

```console
python run.py --step [수행단계]
```

<br/>

---

<br/>

#### ✔ Implemented features
```
log-mel spectrum (mel)
MFCC (mfc)
```

<br/>

#### ✔ Implemented models
```
KMEANS (kmeans)
GMM (gmm)
HMM (hmm)
FC-DNN (dnn)
CNN (cnn)
LSTM (rnn)
```

<br/>

---

<br/>

### ⏬ Dataset

- [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html)는 8,732개 sample로 구성(10개 class, 4초이하)
- Download form을 작성 후 다운 받아, repository에 있는 ./data/wav/* 에 압축해제

<br/>

---

<br/>

### License
```
Copyright (c) 2020-IMPRESS.
