# 🎵 Music Genre Classification & Recommendation

> **Librosa**로 오디오 특성을 추출하고, **XGBoost**로 음악 장르를 분류하며,  
> **Cosine Similarity**로 유사한 음악을 추천하는 ML 프로젝트입니다.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Librosa](https://img.shields.io/badge/Librosa-0.10+-orange.svg)](https://librosa.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-green.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 프로젝트 개요

음성 데이터는 시간 도메인의 진폭(Amplitude) 배열로 표현됩니다.  
이 프로젝트에서는 **푸리에 변환(Fourier Transform)** 을 핵심으로 오디오 신호를 주파수 영역으로 변환하고, 다양한 오디오 특성(Feature)을 추출해 머신러닝 모델을 학습시킵니다.

| 단계 | 내용 |
|------|------|
| **1 오디오 이해** | Waveform, STFT, Spectrogram, Mel Spectrogram 시각화 |
| **2 특성 추출** | Tempo, ZCR, Harmonic/Percussive, Spectral Centroid/Rolloff, MFCCs, Chroma |
| **3 장르 분류** | XGBoost 분류기 학습 (정확도 ~88%) |
| **4 음악 추천** | Cosine Similarity 기반 유사 곡 TOP-N 추천 |

---

## 🗂 프로젝트 구조

```
music-genre-classification/
├── notebooks/
│   └── music_genre_classification.ipynb   # 전체 실습 노트북 (메인)
├── src/
│   ├── features.py      # 오디오 특성 추출 유틸리티
│   ├── classifier.py    # XGBoost 장르 분류 클래스
│   └── recommender.py   # Cosine Similarity 추천 클래스
├── Data/                # 데이터 폴더 (직접 다운로드 필요, 아래 참고)
├── requirements.txt
└── README.md
```

---

## 🚀 빠른 시작

### 1. 환경 설정

```bash
git clone https://github.com/YOUR_USERNAME/music-genre-classification.git
cd music-genre-classification
pip install -r requirements.txt
```

### 2. 데이터셋 다운로드

[Kaggle - GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) 에서 다운로드 후 프로젝트 루트에 압축 해제합니다.

```
Data/
├── genres_original/       # 30초 wav 파일 (10장르 × 100곡)
│   ├── blues/
│   ├── classical/
│   └── ...
├── features_3_sec.csv     # 3초 단위 특성 추출 CSV
└── features_30_sec.csv    # 30초 단위 특성 추출 CSV
```

> **Colab 사용 시** 노트북 첫 번째 셀에서 Kaggle API 연동으로 자동 다운로드할 수 있습니다.

### 3. 노트북 실행

```bash
jupyter notebook notebooks/music_genre_classification.ipynb
```

---

## 🎓 핵심 개념

### 오디오 데이터 표현

음성 파일은 **진폭(Amplitude)을 시간 순서로 나열한 숫자 배열**입니다.

```
y  = [ 0.002, -0.003,  0.001, ... ]   # 진폭 배열
sr = 22050                             # Sampling Rate (1초당 샘플 수)
음악 길이(초) = len(y) / sr
```

### 푸리에 변환 (Fourier Transform)

> "모든 파형은 여러 주기함수의 합으로 나타낼 수 있다"

시간(Time) 도메인 신호를 **주파수(Frequency) 도메인으로 변환**합니다.  
STFT(Short-Time Fourier Transform)는 짧은 구간별로 FFT를 수행해 시간 해상도를 유지합니다.

```
Window 파라미터:
  n_fft      : 한 번 FFT를 수행할 구간 길이 (기본 2048)
  hop_length : 분석 구간을 이동하는 간격 (기본 512)
```

### 주요 오디오 특성

| 특성 | 설명 |
|------|------|
| **Tempo (BPM)** | 음악의 빠르기, 분당 박자 수 |
| **Zero Crossing Rate** | 음파가 양↔음으로 바뀌는 비율 |
| **Harmonic / Percussive** | 음색(색깔) / 리듬 충격파 성분 |
| **Spectral Centroid** | 주파수의 가중평균 — 소리의 "무게 중심" |
| **Spectral Rolloff** | 총 에너지의 85%가 집중된 주파수 |
| **MFCCs** | 인간 청각을 반영한 소리 고유 특징 벡터 (20차원) |
| **Chroma Frequencies** | 12음계 특성, 화음 인식에 효과적 |

### 분류 모델 — XGBoost

```
XGBoost (Extreme Gradient Boosting)
  - 잔차(Residual)를 학습하는 트리를 반복 추가
  - 약한 학습기 → 강한 학습기로 앙상블
  - 데이터 분석 대회에서 꾸준히 상위권
```

### 추천 알고리즘 — Cosine Similarity

$$\text{similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|} = \cos\theta$$

- cos θ = **1** : 두 곡이 완전히 유사
- cos θ = **-1** : 두 곡이 완전히 다름

---

## 📊 실험 결과

### 장르 분류 성능

| 모델 | Accuracy |
|------|----------|
| XGBoost (n=1000, lr=0.05) | **~88%** |

### Confusion Matrix 주요 인사이트

- **Rock ↔ Country** 혼동이 가장 많음 (음악적 특성 유사)
- Classical, Metal은 구분이 명확 (스펙트럼 특성 뚜렷)

### 추천 예시

```python
# rock.00000.wav 입력 시 추천 결과
┌─────────────────────┬──────────────┬─────────┐
│ song                │ similarity   │ genre   │
├─────────────────────┼──────────────┼─────────┤
│ rock.00099.wav      │ 0.9987       │ rock    │
│ rock.00045.wav      │ 0.9984       │ rock    │
│ country.00067.wav   │ 0.9971       │ country │  ← Rock↔Country 유사성
│ rock.00023.wav      │ 0.9969       │ rock    │
│ rock.00088.wav      │ 0.9965       │ rock    │
└─────────────────────┴──────────────┴─────────┘
```

---

## 🛠 모듈 사용 예시

```python
# 장르 분류
from src.classifier import GenreClassifier

clf = GenreClassifier()
clf.load_data('Data/features_3_sec.csv')
clf.train()
clf.evaluate()
clf.confusion_matrix()
clf.save('models/genre_clf.pkl')

# 음악 추천
from src.recommender import MusicRecommender

rec = MusicRecommender()
rec.fit('Data/features_30_sec.csv')
print(rec.recommend('jazz.00010.wav', n=5))
```

---

## 📚 참고 자료

- [빵형의 개발도상국 — 음악 장르 분류](https://www.youtube.com/watch?v=IE6lue0qusQ)
- [Librosa Documentation](https://librosa.org/doc/latest/)
- [GTZAN Dataset on Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

---

## 📝 License

MIT License
