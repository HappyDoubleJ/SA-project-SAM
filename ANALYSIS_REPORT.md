# SAM 기반 피부 병변 분할 파이프라인 분석 보고서

## 1. 개요

본 보고서는 `sam_masking.py` 코드의 전체 구조와 각 구성요소의 설계 근거, 파라미터 선택 이유, 그리고 현재 성능 문제의 원인 분석을 담고 있습니다.

---

## 2. 시스템 아키텍처

### 2.1 전체 구조

```
┌─────────────────────────────────────────────────────────────┐
│                    SAM Masking Pipeline                      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │  SAMSegmenter   │  │  SAM2Segmenter  │  │MedSAM2Segm.  │ │
│  │  (ViT-H 기반)   │  │  (Hiera-L 기반) │  │ (Hiera-T)    │ │
│  └────────┬────────┘  └────────┬────────┘  └──────┬───────┘ │
│           │                    │                   │         │
│           └────────────────────┼───────────────────┘         │
│                                ▼                             │
│                   ┌─────────────────────┐                    │
│                   │ LesionFeatureDetector│                   │
│                   │   (특징 기반 탐지)   │                    │
│                   └─────────────────────┘                    │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 분할 전략

| 전략 | 설명 | 장점 | 단점 |
|------|------|------|------|
| Center-Focused | 이미지 중앙에 병변이 있다고 가정 | 단순하고 빠름, 임상 사진에 적합 | 병변이 중앙에 없으면 실패 |
| Lesion-Feature-Based | 색상/질감 분석으로 병변 위치 추정 | 이론적으로 더 정확한 위치 탐지 | 노이즈에 민감, 파라미터 튜닝 필요 |

---

## 3. 모델별 상세 분석

### 3.1 SAM (Segment Anything Model)

#### 3.1.1 모델 구성
```python
CHECKPOINT_URL = "sam_vit_h_4b8939.pth"  # ViT-Huge 모델
```

**모델 선택 근거:**
- **ViT-H (Huge)**: 가장 큰 SAM 모델로, 최고의 분할 품질 제공
- 파라미터 수: ~636M
- 일반 객체 분할에서 SOTA 성능

#### 3.1.2 SamAutomaticMaskGenerator 파라미터
```python
SamAutomaticMaskGenerator(
    model=self.model,
    points_per_side=32,           # 그리드당 포인트 수
    pred_iou_thresh=0.86,         # 예측 IoU 임계값
    stability_score_thresh=0.92,  # 안정성 점수 임계값
    crop_n_layers=1,              # 크롭 레이어 수
    crop_n_points_downscale_factor=2,  # 크롭 시 다운스케일
    min_mask_region_area=100,     # 최소 마스크 영역
)
```

| 파라미터 | 값 | 선택 근거 |
|----------|-----|-----------|
| `points_per_side` | 32 | 32x32=1024개 포인트로 충분한 커버리지 확보 |
| `pred_iou_thresh` | 0.86 | SAM 논문 기본값, 높은 품질 마스크만 유지 |
| `stability_score_thresh` | 0.92 | 안정적인 마스크만 선택 (0.9~0.95 권장) |
| `min_mask_region_area` | 100 | 너무 작은 노이즈 마스크 제거 |

### 3.2 SAM2 (Segment Anything Model 2)

#### 3.2.1 모델 구성
```python
CHECKPOINT_URL = "sam2.1_hiera_large.pt"  # Hiera-Large 모델
CONFIG_NAME = "sam2.1_hiera_l.yaml"
```

**모델 선택 근거:**
- **Hiera-L (Large)**: 비디오 분할도 지원하는 차세대 SAM
- MAE 기반 사전학습으로 더 나은 표현 학습
- 메모리 효율적인 Hierarchical Vision Transformer 사용

#### 3.2.2 Hydra 설정 처리
```python
# Hydra 초기화가 필요한 이유:
# SAM2는 OmegaConf/Hydra를 사용하여 설정을 관리
# build_sam2 함수 호출 전 Hydra 컨텍스트 필요

from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

GlobalHydra.instance().clear()  # 기존 인스턴스 정리
with initialize_config_dir(config_dir=str(config_dir), version_base=None):
    self.model = build_sam2(config_name, checkpoint_path, device=self.device)
```

### 3.3 MedSAM2 (Medical SAM2)

#### 3.3.1 모델 구성
```python
CHECKPOINT_URL = "MedSAM2_latest.pt"  # 의료 영상 특화 모델
CONFIG_NAME = "sam2.1_hiera_t.yaml"   # Tiny 모델 (경량화)
```

**모델 선택 근거:**
- **Hiera-T (Tiny)**: 의료 영상에 최적화된 경량 모델
- 다양한 의료 영상 데이터셋으로 파인튜닝
- CT, MRI, X-ray, 피부 영상 등 다양한 모달리티 지원

#### 3.3.2 MedSAM2 특수 처리
```python
# MedSAM2는 단일 마스크 출력이 기본
masks, scores, _ = self.predictor.predict(
    point_coords=np.array([[point[0], point[1]]], dtype=np.float32),
    point_labels=np.array([1], dtype=np.int32),
    multimask_output=False,  # 단일 마스크 (MedSAM2 스타일)
)
```

---

## 4. LesionFeatureDetector 상세 분석

### 4.1 색상 이상 탐지 (detect_color_anomalies)

#### 4.1.1 LAB 색공간 변환
```python
lab = color.rgb2lab(image)
l_channel = lab[:, :, 0]  # 밝기 (0-100)
a_channel = lab[:, :, 1]  # 녹색-빨간색 (-128 ~ +127)
b_channel = lab[:, :, 2]  # 파란색-노란색 (-128 ~ +127)
```

**LAB 선택 근거:**
- 인간 시각에 근접한 색공간
- 피부색 분석에 적합 (a* 채널이 홍반 탐지에 유용)
- 조명 변화에 상대적으로 강건

#### 4.1.2 로컬 평균 차이 계산
```python
kernel_size = max(image.shape[0], image.shape[1]) // 8
```

| 이미지 크기 | kernel_size | 의미 |
|------------|-------------|------|
| 400x400 | 50 | 이미지의 1/8 영역 |
| 800x800 | 100 | 더 넓은 컨텍스트 |
| 최소값 | 15 | 너무 작으면 노이즈에 민감 |

**문제점:** 고정 비율은 다양한 병변 크기에 적응하지 못함

#### 4.1.3 색상 이상 점수 계산
```python
color_anomaly = (l_diff / 50.0 + a_diff / 30.0 + b_diff / 30.0) / 3.0
```

| 가중치 | 값 | 근거 |
|--------|-----|------|
| L* 정규화 | 50.0 | L* 범위(0-100)의 절반 |
| a* 정규화 | 30.0 | 일반적인 피부색 a* 변동 범위 |
| b* 정규화 | 30.0 | 일반적인 피부색 b* 변동 범위 |

#### 4.1.4 추가 특징
```python
# 채도 기반 탐지 (HSV)
s_normalized = s_channel.astype(float) / 255.0
high_saturation = s_normalized > 0.3  # 30% 이상 채도

# 홍반 탐지 (LAB a* 채널)
redness = (a_channel > 10).astype(float)  # a* > 10 = 붉은색

# 최종 조합
combined = color_anomaly + 0.3 * high_saturation + 0.2 * redness
```

| 특징 | 가중치 | 근거 |
|------|--------|------|
| color_anomaly | 1.0 | 기본 색상 이상 |
| high_saturation | 0.3 | 보조 지표 |
| redness | 0.2 | 염증성 병변에 유용 |

### 4.2 질감/입체감 탐지 (detect_texture_elevation)

#### 4.2.1 그래디언트 크기
```python
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)
```

**목적:** 경계/에지 탐지 → 융기된 병변의 윤곽 감지

**문제점:** 에지는 병변의 **경계**에서 최대이므로, 병변 **중심**을 찾는 데는 부적합

#### 4.2.2 라플라시안
```python
laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
```

**목적:** 2차 미분으로 급격한 밝기 변화 탐지

**문제점:** 노이즈에 매우 민감, 작은 질감 변화도 과대평가

#### 4.2.3 로컬 분산
```python
kernel_size = 11  # 고정 크기
local_mean = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
local_sq_mean = cv2.GaussianBlur(gray**2, (kernel_size, kernel_size), 0)
local_variance = local_sq_mean - local_mean**2
```

**목적:** 질감이 거친 영역(인설, 태선화) 탐지

**문제점:** kernel_size=11은 고정값으로, 이미지 크기에 적응하지 않음

#### 4.2.4 질감 점수 조합
```python
texture_score = 0.4 * gradient_norm + 0.3 * laplacian_norm + 0.3 * variance_norm
```

| 특징 | 가중치 | 근거 |
|------|--------|------|
| gradient | 0.4 | 경계 탐지에 가장 신뢰성 높음 |
| laplacian | 0.3 | 급격한 변화 보조 탐지 |
| variance | 0.3 | 질감 거칠기 탐지 |

### 4.3 후보 포인트 선택

```python
# 색상과 질감 조합
combined = 0.6 * color_map + 0.4 * texture_map

# 중앙 편향 적용
center_weight = 1.0 - 0.3 * (distance_from_center / max_dist)
combined_weighted = combined * center_weight
```

| 조합 | 가중치 | 근거 |
|------|--------|------|
| color_map | 0.6 | 색상이 더 신뢰성 높음 |
| texture_map | 0.4 | 질감은 보조 지표 |
| center_bias | 0.7~1.0 | 임상 사진은 대부분 병변 중심 |

### 4.4 바운딩 박스 생성

```python
threshold = 0.3  # 이진화 임계값
binary = combined > threshold

# 형태학적 연산
binary = morphology.binary_opening(binary, morphology.disk(3))  # 노이즈 제거
binary = morphology.binary_closing(binary, morphology.disk(5))  # 구멍 메우기
```

| 연산 | 구조 요소 | 목적 |
|------|-----------|------|
| opening | disk(3) | 작은 노이즈 점 제거 |
| closing | disk(5) | 병변 내부 작은 구멍 메우기 |

---

## 5. 분할 전략 비교

### 5.1 Center-Focused 전략

```python
def segment_center_focused(self, image):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # 방법 1: 중앙 포인트
    mask, score = self.segment_with_point(image, center)

    # 방법 2: 중앙 박스 (15% 마진)
    margin = 0.15
    box = (int(w * margin), int(h * margin),
           int(w * (1 - margin)), int(h * (1 - margin)))
    mask_box, score_box = self.segment_with_box(image, box)

    # 더 높은 점수 선택
    return best_result
```

**장점:**
- 단순하고 빠름
- 임상 피부 사진은 대부분 병변이 중앙에 위치
- SAM의 "주요 객체 분할" 특성과 잘 맞음

### 5.2 Lesion-Feature-Based 전략

```python
def segment_lesion_features(self, image):
    # 특징 기반 후보 포인트 추출
    candidate_points = self.feature_detector.get_lesion_candidate_points(image, n_points=3)
    feature_box = self.feature_detector.get_lesion_bounding_box(image)

    # 각 후보에 대해 분할 시도
    for point in candidate_points:
        mask, score = self.segment_with_point(image, point)
        # 최고 점수 선택

    # 바운딩 박스도 시도
    mask_box, score_box = self.segment_with_box(image, feature_box)
```

**현재 문제점:**
1. 에지 기반 특징이 병변 경계에서 최대 → 중심이 아닌 가장자리를 선택
2. 임계값(0.3)이 낮아 너무 많은 영역이 후보로 선택됨
3. 정상 피부의 색상 변이도 이상으로 탐지

---

## 6. 성능 저하 원인 분석

### 6.1 근본적 문제

| 문제 | 상세 설명 | 영향 |
|------|-----------|------|
| **에지 vs 중심** | Sobel/Laplacian은 경계에서 최대 응답 | 병변 가장자리 포인트 선택 |
| **로컬 평균 한계** | 병변이 주요 영역일 때 "이상"으로 감지 안됨 | 병변 누락 |
| **고정 파라미터** | 다양한 병변 크기/유형에 적응 불가 | 일반화 실패 |
| **피부톤 미고려** | Fitzpatrick 스케일별 기준색 없음 | 인종별 성능 차이 |

### 6.2 임상 사진 특성

```
┌─────────────────────────────────────┐
│     일반적인 피부과 임상 사진        │
├─────────────────────────────────────┤
│                                     │
│           ┌─────────┐               │
│           │         │               │
│           │  병변   │  ← 거의 항상  │
│           │  (중앙) │     중앙 배치 │
│           │         │               │
│           └─────────┘               │
│                                     │
│     정상 피부 (배경)                 │
└─────────────────────────────────────┘
```

**Center-Focused가 더 잘 작동하는 이유:**
1. 피부과 사진은 촬영 프로토콜상 병변을 중앙에 배치
2. SAM은 중앙 포인트에서 "주요 객체"를 분할하도록 학습됨
3. 추가적인 특징 분석이 오히려 노이즈를 유발

### 6.3 Feature Detection의 한계

```python
# 현재 접근: 에지 기반 특징
gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)  # 경계에서 최대

# 더 나은 접근: 영역 기반 특징
# - Superpixel segmentation (SLIC)
# - Region growing
# - Saliency detection
```

---

## 7. 개선 방안 제안

### 7.1 단기 개선 (파라미터 튜닝)

```python
# 색상 이상 탐지 개선
high_saturation = s_normalized > 0.5  # 0.3 → 0.5 (더 보수적)
redness = (a_channel > 15).astype(float)  # 10 → 15

# 바운딩 박스 임계값
threshold = 0.5  # 0.3 → 0.5 (더 선택적)

# 질감 가중치 조정
texture_score = 0.2 * gradient_norm + 0.3 * laplacian_norm + 0.5 * variance_norm
# gradient 낮추고 variance 높임 (에지보다 질감 중시)
```

### 7.2 중기 개선 (알고리즘 개선)

1. **Saliency Detection 도입**
```python
from skimage.segmentation import slic
from skimage.measure import regionprops

# SLIC superpixel로 영역 분할
segments = slic(image, n_segments=100, compactness=10)
# 영역별 색상 이상 계산
```

2. **적응형 임계값**
```python
# Otsu 기반 자동 임계값
from skimage.filters import threshold_otsu
thresh = threshold_otsu(combined_score)
```

3. **피부톤 정규화**
```python
# 피부 영역 자동 탐지 후 기준 색상으로 정규화
skin_mask = detect_skin_region(image)
skin_mean_lab = image[skin_mask].mean(axis=0)
normalized = image - skin_mean_lab
```

### 7.3 장기 개선 (모델 기반)

1. **전용 병변 탐지 모델 사용**
   - YOLO/Faster R-CNN 기반 피부 병변 탐지기
   - 탐지된 바운딩 박스를 SAM 프롬프트로 사용

2. **Self-Supervised Pre-detection**
   - 무감독 학습으로 병변 후보 영역 사전 탐지
   - Attention map 활용

---

## 8. 사용된 라이브러리 및 의존성

| 라이브러리 | 버전 | 용도 |
|-----------|------|------|
| torch | >= 2.0 | 딥러닝 프레임워크 |
| segment-anything | latest | SAM 모델 |
| sam2 | latest | SAM2 모델 |
| opencv-python | >= 4.0 | 이미지 처리 |
| scikit-image | >= 0.19 | 색공간 변환, 형태학 연산 |
| scipy | >= 1.7 | 수치 연산 |
| hydra-core | >= 1.3 | SAM2 설정 관리 |

---

## 9. 결론

### 9.1 현재 상태

- **Center-Focused**: 임상 피부 사진에서 안정적으로 작동
- **Lesion-Feature-Based**: 이론적 근거는 있으나 실제 성능 미달

### 9.2 권장 사항

1. **단기**: Center-Focused를 주 전략으로 사용
2. **중기**: Feature Detection 알고리즘 전면 개선
3. **장기**: 전용 병변 탐지 모델 통합

### 9.3 핵심 교훈

> "피부과 임상 사진은 이미 '전처리된' 데이터이다.
> 촬영자가 병변을 중앙에 배치했으므로,
> 복잡한 특징 분석보다 단순한 중앙 가정이 더 효과적이다."

---

## 부록: 피부 병변 형태학적 특징 (참고)

| 특징 | 설명 | 탐지 방법 |
|------|------|-----------|
| 융기 (Papule/Nodule) | 피부보다 솟아오름 | 그림자, 3D 재구성 |
| 색조 변화 (Erythema) | 붉은색/갈색/흰색 | LAB a* 채널 |
| 인설 (Scale) | 각질이 일어남 | 로컬 분산, 질감 |
| 경계 (Border) | 명확/불명확 | 에지 선명도 |
| 대칭성 (Symmetry) | 규칙적/불규칙 | 형태 분석 |

---

*보고서 작성일: 2024*
*작성: SAM Masking Pipeline Analysis*
