# 🚗 NextFrame Model Verification

> NextFrame 차량 번호판 검출 프로젝트에서 사용한 YOLO 모델을 대상으로
> 프로젝트 종료 후 별도의 테스트셋을 구성하여 독립 성능 검증 및 오류 분석을 수행했습니다.

---

## 🔗 Original Project

### NextFrame - 자동 번호판 인식 시스템

[기존 프로젝트 GitHub](https://github.com/JHKINI/nextframe_opencv)

기존 프로젝트에서는 YOLO 기반 번호판 탐지와 EasyOCR을 활용하여
차량 영상에서 번호판을 검출하고 문자를 인식하는 전체 시스템을 구현했습니다.

본 저장소는 기존 프로젝트의 코드를 수정한 저장소가 아니라,
프로젝트 종료 후 모델의 성능을 별도로 검증하기 위해 수행한
개인 후속 분석을 정리한 저장소입니다.

---

## 📌 검증 목적

기존 프로젝트에서 사용한 번호판 검출 모델의 성능을
학습에 사용하지 않은 별도의 데이터로 다시 확인하고,

- 독립 테스트셋에서의 검출 성능 확인
- Precision / Recall / mAP 측정
- 검출 오류 후보 분석
- Bounding Box 위치 정밀도 확인
- 추가적인 개선 방향 도출

을 목적으로 검증을 수행했습니다.

---

## 📊 Test Dataset

### Dataset Source

[Kaggle - Automatic License Plate Recognition (ALPR) Dataset](https://www.kaggle.com/datasets/mgmitesh/automatic-license-plate-recognition-alpr-dataset)

기존 프로젝트와 별도로 Kaggle의 ALPR 데이터셋에서
테스트에 사용할 이미지를 직접 선별하여 독립 테스트셋을 구성했습니다.

### Image Selection

전체 데이터 중 다음 기준을 적용하여 테스트 이미지를 선별했습니다.

- 차량이 명확하게 식별되는 이미지
- 차량이 심하게 잘리지 않은 이미지
- 번호판이 정상적으로 포함된 이미지
- 번호판을 확인할 수 있는 이미지
- 심하게 흐리거나 화질이 지나치게 낮은 이미지 제외

최종적으로 **489장의 이미지**를 테스트셋으로 구성했습니다.

| 항목 | 결과 |
|---|---:|
| Test Images | 489 |
| Labels | 489 |
| Ground Truth Objects | 505 |
| Corrupt Images | 0 |

---

## ⚙️ Evaluation Environment

- Python 3.12.13
- Ultralytics 8.4.67
- PyTorch 2.11.0+cu128
- NVIDIA RTX 3080
- CUDA
- YOLO

---

## 🧪 Independent Evaluation

기존 프로젝트에서 사용한 `best.pt` 모델을 별도의 테스트셋에 적용하여
번호판 객체 검출 성능을 평가했습니다.

### Evaluation Result

| Metric | Result |
|---|---:|
| Precision | **98.8%** |
| Recall | **96.2%** |
| mAP@50 | **99.0%** |
| mAP@50-95 | **64.0%** |

### Result Interpretation

mAP@50에서는 높은 검출 성능을 확인했습니다.

반면 mAP@50-95에서는 상대적으로 낮은 결과가 나타났으며,
이를 통해 번호판의 존재 여부를 검출하는 것과 비교하여
Bounding Box의 위치 및 크기를 보다 정밀하게 맞추는 부분에
개선 여지가 있음을 확인했습니다.

---

## 🔍 Error Analysis

정량적인 성능 평가 이후 테스트 이미지에 대한
별도의 오류 분석을 수행했습니다.

### Analysis Process

```text
Independent Test Dataset
          ↓
YOLO Inference
          ↓
Prediction / Ground Truth 비교
          ↓
IoU 기반 매칭
          ↓
오류 후보 이미지 추출
          ↓
이미지 직접 확인
          ↓
오류 패턴 분석

확인한 오류 패턴

분석 과정에서 다음과 같은 상황에서 추가적인 개선 가능성을 확인했습니다.

작은 크기의 번호판
원거리 차량
다양한 촬영 각도
여러 차량 또는 여러 번호판이 포함된 장면
조명 및 반사 조건
Bounding Box 위치 및 크기 정밀도

단순히 수치만 확인하는 것이 아니라
오류 후보 이미지를 직접 확인하여
실제 검출 상황을 함께 분석했습니다.

📁 Repository Structure
nextframe-model-verification/
│
├── README.md
│
├── verification/
│   ├── remove_orphan_labels.py
│   ├── test.yaml
│   ├── error_analysis.py
│   └── error_analysis.csv
│
├── results/
│   ├── metrics/
│   │   ├── BoxF1_curve.png
│   │   ├── BoxP_curve.png
│   │   ├── BoxPR_curve.png
│   │   ├── BoxR_curve.png
│   │   ├── confusion_matrix.png
│   │   ├── confusion_matrix_normalized.png
│   │   └── val_batch*_labels/pred.jpg
│   │
│   └── error_analysis/
│       ├── 01_FN_놓친번호판/
│       ├── 02_FP_잘못검출/
│       └── 03_위치정밀도_낮은검출/  ← 대표 10장
│
└── portfolio/
        └── 차량_번호판_검출_독립검증.pptx
🎯 Verification Result

이번 검증을 통해 기존 모델의 성능을
별도의 테스트 데이터에서 다시 확인했습니다.

특히,

독립 테스트셋 구성 → 정량적 성능 평가 → 오류 후보 추출 → 이미지 직접 분석

의 과정을 통해 모델 성능을 단순히 제시하는 것에서 그치지 않고
검출 성능의 강점과 개선이 필요한 부분을 함께 확인했습니다.

📎 Related Project

Original Team Project

NextFrame - 자동 번호판 인식 시스템

https://github.com/JHKINI/nextframe_opencv
