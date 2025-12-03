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

# Part 2: OpenAI LLM 진단 시스템 및 전체 파이프라인 분석

## 10. 전체 파이프라인 흐름도

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Main Pipeline (main.py)                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐                                                        │
│  │  입력 이미지 │  Derm1M_v2_pretrain_ontology_sampled_100_images/      │
│  │   디렉토리   │  (.jpg, .jpeg, .png, .bmp)                            │
│  └──────┬──────┘                                                        │
│         │                                                               │
│         ▼                                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │              모델 초기화 (Lazy Loading)                          │   │
│  │  ┌─────────┐  ┌──────────┐  ┌────────────┐                      │   │
│  │  │   SAM   │  │   SAM2   │  │  MedSAM2   │                      │   │
│  │  │ (ViT-H) │  │(Hiera-L) │  │ (Hiera-T)  │                      │   │
│  │  └─────────┘  └──────────┘  └────────────┘                      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│         │                                                               │
│         ▼                                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │           이미지별 분할 처리 (process_single_image)              │   │
│  │                                                                  │   │
│  │  각 모델 × 2 전략 = 최대 6개 분할 결과                           │   │
│  │  - SAM_center_focused, SAM_lesion_features                       │   │
│  │  - SAM2_center_focused, SAM2_lesion_features                     │   │
│  │  - MedSAM2_center_focused, MedSAM2_lesion_features               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│         │                                                               │
│         ▼                                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │              시각화 저장 (visualize.py)                          │   │
│  │  - 개별 오버레이 이미지                                          │   │
│  │  - 크롭된 병변 영역                                              │   │
│  │  - 전체 비교 그리드                                              │   │
│  │  - 마스크 차이 분석 (IoU, Dice)                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│         │                                                               │
│         ▼                                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │           OpenAI 진단 (openai_diagnosis.py)                      │   │
│  │                                                                  │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │   │
│  │  │ Original Only   │  │ With Overlay    │  │ With Cropped    │  │   │
│  │  │ (원본만)        │  │ (원본+마스크)   │  │ (원본+크롭)     │  │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘  │   │
│  │                              │                                   │   │
│  │                              ▼                                   │   │
│  │                    GPT-4o Vision API                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│         │                                                               │
│         ▼                                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                        출력 결과물                               │   │
│  │  outputs/                                                        │   │
│  │  ├── checkpoints/          (모델 체크포인트)                     │   │
│  │  ├── visualizations/       (시각화 이미지)                       │   │
│  │  ├── diagnosis/            (진단 JSON 파일)                      │   │
│  │  ├── segmentation_results.json                                   │   │
│  │  └── diagnosis_report.md                                         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 11. OpenAI 진단 시스템 상세 분석 (openai_diagnosis.py)

### 11.1 SkinDiseaseDiagnoser 클래스 구조

```python
class SkinDiseaseDiagnoser:
    """OpenAI-based skin disease diagnosis using GPT-4 Vision"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-4o"  # GPT-4o 비전 모델 사용
```

### 11.2 사용 모델 및 API 설정

| 설정 | 값 | 선택 근거 |
|------|-----|-----------|
| **모델** | `gpt-4o` | 최신 멀티모달 모델, 이미지 분석 능력 최상 |
| **max_tokens** | 2000 | 상세한 진단 결과를 위해 충분한 토큰 할당 |
| **temperature** | 0.3 | 낮은 값으로 일관성 있는 의료 분석 유도 |
| **detail** | "high" | 고해상도 이미지 분석 모드 |

### 11.3 프롬프트 설계 분석

#### 11.3.1 단일 이미지 진단 프롬프트 (DIAGNOSIS_PROMPT)

```python
DIAGNOSIS_PROMPT = """You are an expert dermatologist AI assistant.
Analyze the provided skin image(s) and provide a detailed diagnosis.

For each image analysis, provide:
1. **Observed Features**: Describe the visible skin lesion characteristics
   (color, shape, texture, borders, size estimation)
2. **Possible Conditions**: List the top 3-5 most likely skin conditions
3. **Confidence Level**: Rate your confidence (Low/Medium/High)
4. **Key Differentiating Factors**: What features led to your diagnosis
5. **Recommended Actions**: Suggest next steps
...
"""
```

**프롬프트 설계 근거:**

| 요소 | 설명 | 의도 |
|------|------|------|
| **역할 지정** | "expert dermatologist AI" | 전문가 수준의 분석 유도 |
| **구조화된 출력** | JSON 형식 강제 | 파싱 용이성, 일관된 결과 |
| **다중 조건 요청** | "top 3-5 conditions" | 불확실성 반영, 감별진단 |
| **신뢰도 레벨** | Low/Medium/High | 결과 해석 가이드 |
| **면책 조항** | "educational/research purposes" | 법적 보호 |

#### 11.3.2 비교 진단 프롬프트 (COMPARISON_PROMPT)

```python
COMPARISON_PROMPT = """You are provided with two views of the same skin lesion:
1. The original full image
2. A highlighted/segmented view showing the lesion area of interest

Analyze both images together and provide a comprehensive diagnosis.
The segmented image helps you focus on the specific area of concern.
...
"""
```

**추가 요소:**

| 필드 | 목적 |
|------|------|
| `segmentation_benefits` | SAM 분할이 진단에 도움이 되었는지 평가 |

### 11.4 진단 방법 비교

#### 11.4.1 방법 1: 원본 이미지만 사용

```python
def diagnose_original_only(self, image: np.ndarray) -> Dict:
    """단일 원본 이미지로 진단"""
    base64_image = self._encode_image(image)

    response = self.client.chat.completions.create(
        model=self.model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": self.DIAGNOSIS_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}",
                        "detail": "high"  # 고해상도 분석
                    }
                }
            ]
        }],
        max_tokens=2000,
        temperature=0.3
    )
```

**장점:**
- 비용 효율적 (이미지 1개)
- 빠른 응답
- 기준선(Baseline) 역할

**단점:**
- 병변 위치 불명확
- 배경 영역에 주의 분산 가능

#### 11.4.2 방법 2: 원본 + 마스크 오버레이

```python
def diagnose_with_mask(self, original, masked_overlay, cropped=None, use_cropped=False):
    """원본과 분할 결과를 함께 제공"""

    # 두 이미지를 순차적으로 제공
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": self.COMPARISON_PROMPT},
            {"type": "text", "text": "Image 1 - Original full image:"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_original}"}},
            {"type": "text", "text": "Image 2 - highlighted lesion area:"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_masked}"}}
        ]
    }]
```

**장점:**
- 관심 영역 명확히 지정
- 병변 경계 시각화
- 진단 초점 유도

**단점:**
- 비용 증가 (이미지 2개)
- 잘못된 분할 시 오진 유발 가능

#### 11.4.3 방법 3: 원본 + 크롭된 병변

```python
# use_cropped=True일 때
if use_cropped and cropped is not None:
    base64_masked = self._encode_image(cropped)
    second_image_desc = "cropped lesion region"
```

**장점:**
- 병변에 완전히 집중
- 세부 텍스처 분석 용이
- 확대된 뷰 제공

**단점:**
- 주변 맥락 손실
- 크롭 영역이 너무 작으면 정보 부족

### 11.5 이미지 인코딩 방식

```python
def _encode_image(self, image: np.ndarray) -> str:
    """numpy 배열을 base64 문자열로 변환"""
    pil_image = Image.fromarray(image)
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")  # PNG 형식 사용
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
```

| 포맷 | 선택 이유 |
|------|-----------|
| **PNG** | 손실 없는 압축, 의료 이미지에 적합 |
| **Base64** | API 전송에 필요한 텍스트 인코딩 |

### 11.6 JSON 응답 파싱

```python
# LLM 응답에서 JSON 추출
try:
    json_start = result_text.find('{')
    json_end = result_text.rfind('}') + 1
    if json_start != -1 and json_end > json_start:
        result = json.loads(result_text[json_start:json_end])
    else:
        result = {"raw_response": result_text}
except json.JSONDecodeError:
    result = {"raw_response": result_text}
```

**파싱 전략:**
1. 첫 번째 `{`와 마지막 `}` 사이 추출
2. JSON 파싱 시도
3. 실패 시 원본 텍스트 저장 (`raw_response`)

---

## 12. 메인 파이프라인 상세 분석 (main.py)

### 12.1 커맨드라인 인터페이스

```python
parser.add_argument("--data-dir", default="Derm1M_v2_pretrain_ontology_sampled_100_images")
parser.add_argument("--output-dir", default="outputs")
parser.add_argument("--max-images", type=int, default=None)
parser.add_argument("--no-diagnosis", action="store_true")
parser.add_argument("--no-sam", action="store_true")
parser.add_argument("--no-sam2", action="store_true")
parser.add_argument("--no-medsam2", action="store_true")
parser.add_argument("--no-visualizations", action="store_true")
```

**실행 예시:**
```bash
# 전체 파이프라인 실행
python main.py --data-dir ./skin_images --output-dir ./results

# SAM만 사용, 진단 제외
python main.py --no-sam2 --no-medsam2 --no-diagnosis

# 처음 10개 이미지만 테스트
python main.py --max-images 10
```

### 12.2 입력 데이터 구조

```
Derm1M_v2_pretrain_ontology_sampled_100_images/
├── 10730_4.png
├── 10883_2.png
├── 1587_2.png
├── 20486_2.png
├── 22269_2.png
└── ... (100개 이미지)
```

**지원 포맷:**
```python
extensions = ('.jpg', '.jpeg', '.png', '.bmp')
```

### 12.3 단일 이미지 처리 흐름

```python
def process_single_image(image_path, sam_segmenter, sam2_segmenter,
                         medsam2_segmenter, output_dir, save_visualizations):

    # 1. 이미지 로드
    image = load_image(str(image_path))  # RGB numpy array

    # 2. 각 모델로 분할 수행
    # SAM: 두 가지 전략 모두 실행
    sam_results = sam_segmenter.segment_both_strategies(image)
    # → {'center_focused': {...}, 'lesion_features': {...}}

    # 3. 오버레이 및 크롭 생성
    for strategy_name, seg_result in sam_results.items():
        seg_result['overlay'] = apply_mask_to_image(
            image, seg_result['mask'],
            color=(255, 0, 0),  # 빨간색 (SAM)
            alpha=0.4
        )
        seg_result['cropped'] = crop_masked_region(image, seg_result['mask'], padding=20)
```

**모델별 오버레이 색상:**

| 모델 | RGB 색상 | 시각적 구분 |
|------|----------|-------------|
| SAM | (255, 0, 0) | 빨간색 |
| SAM2 | (0, 255, 0) | 초록색 |
| MedSAM2 | (0, 0, 255) | 파란색 |

### 12.4 마스크 비교 메트릭

```python
# 모든 분할 결과 쌍에 대해 비교
for i, key1 in enumerate(seg_keys):
    for key2 in seg_keys[i+1:]:
        metrics = compute_mask_metrics(mask1, mask2)
        # → {'iou': 0.85, 'dice': 0.92, 'mask1_area': 1500, ...}
```

**계산 메트릭:**
```python
def compute_mask_metrics(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    iou = intersection / union                    # Intersection over Union
    dice = 2 * intersection / (mask1.sum() + mask2.sum())  # Dice coefficient

    return {
        'iou': iou,
        'dice': dice,
        'mask1_area': mask1.sum(),
        'mask2_area': mask2.sum(),
        'intersection': intersection,
        'union': union
    }
```

### 12.5 최고 분할 결과 선택 (진단용)

```python
# 진단에 사용할 분할 결과 선택
best_seg = None
best_score = 0
best_seg_name = None  # 어떤 모델/전략인지 추적

for seg_name, seg_data in r['segmentations'].items():
    if seg_data.get('score', 0) > best_score:
        best_score = seg_data['score']
        best_seg = seg_data
        best_seg_name = seg_name  # 예: "SAM2_center_focused"

# 가장 높은 confidence score를 가진 분할 결과 사용
diagnosis_data.append({
    'filename': r['filename'],
    'original': r['original'],
    'masked_overlay': best_seg.get('overlay'),
    'cropped': best_seg.get('cropped')
})
```

#### 12.5.1 진단에 사용되는 이미지의 출처

**현재 동작 방식:**
OpenAI 진단에 전송되는 마스크 오버레이와 크롭 이미지는 **모든 분할 결과 중 가장 높은 confidence score를 가진 것**이 자동 선택됩니다.

```
분할 결과 후보 (예시):
┌─────────────────────────────┬─────────┬────────────────┐
│ seg_name                    │ score   │ method         │
├─────────────────────────────┼─────────┼────────────────┤
│ SAM_center_focused          │ 0.987   │ center_point   │
│ SAM_lesion_features         │ 0.892   │ feature_point_0│
│ SAM2_center_focused         │ 0.991   │ center_box     │  ← 최고 점수
│ SAM2_lesion_features        │ 0.876   │ feature_box    │
│ MedSAM2_center_focused      │ 0.945   │ center_point   │
│ MedSAM2_lesion_features     │ 0.823   │ feature_point_1│
└─────────────────────────────┴─────────┴────────────────┘

→ "SAM2_center_focused" (score=0.991)가 진단에 사용됨
```

**선택 기준:**
| 기준 | 설명 |
|------|------|
| **선택 메트릭** | SAM/SAM2/MedSAM2 모델이 반환하는 confidence score |
| **범위** | 모든 모델 × 모든 전략 중에서 최고 선택 |
| **일반적 결과** | center_focused 방법이 더 높은 점수를 받는 경향 |

**주요 관찰:**
- **Center-focused 전략**이 대부분의 경우 더 높은 confidence score를 받음
- 이유: 임상 사진에서 병변이 중앙에 위치 → SAM이 명확한 객체 인식 가능
- **Lesion-features 전략**은 특징 탐지 노이즈로 인해 잘못된 포인트 선택 → 낮은 점수

**⚠️ 현재 제한사항:**
현재 코드는 어떤 모델/방법이 선택되었는지를 **진단 결과에 기록하지 않음**.
추적이 필요하면 아래와 같이 수정 필요:

```python
# 개선된 코드 (제안)
diagnosis_data.append({
    'filename': r['filename'],
    'original': r['original'],
    'masked_overlay': best_seg.get('overlay'),
    'cropped': best_seg.get('cropped'),
    'segmentation_source': best_seg_name,     # "SAM2_center_focused"
    'segmentation_score': best_score,          # 0.991
    'segmentation_method': best_seg.get('method')  # "center_box"
})
```

---

## 13. 시각화 모듈 분석 (visualize.py)

### 13.1 비교 그림 생성

```python
def create_comparison_figure(original, sam_mask, medsam_mask,
                             sam_overlay, medsam_overlay,
                             sam_score, medsam_score, ...):
    """
    2x3 그리드 비교 그림 생성

    Row 1: Original | SAM Overlay | MedSAM2 Overlay
    Row 2: Original | SAM Mask    | MedSAM2 Mask
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.2)
```

### 13.2 마스크 차이 시각화

```python
def visualize_mask_difference(mask1, mask2, labels=("SAM", "MedSAM2")):
    """
    두 마스크의 차이를 색상으로 시각화

    - 녹색: mask1만 포함하는 영역
    - 파란색: mask2만 포함하는 영역
    - 노란색: 두 마스크 모두 포함하는 영역
    """
    only_mask1 = mask1 & ~mask2  # 녹색
    only_mask2 = mask2 & ~mask1  # 파란색
    both = mask1 & mask2         # 노란색
```

---

## 14. 테스트 데이터 및 실험 설정

### 14.1 테스트 데이터셋

| 항목 | 값 |
|------|-----|
| **데이터셋 명** | Derm1M_v2_pretrain_ontology_sampled_100_images |
| **이미지 수** | 100장 |
| **형식** | PNG (주로) |
| **출처** | Derm1M v2 (피부과 이미지 대규모 데이터셋) |
| **내용** | 다양한 피부 질환 임상 사진 |

### 14.2 실험 매트릭스

```
┌────────────────────────────────────────────────────────────┐
│                    실험 구성 매트릭스                        │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  모델 (3종)           전략 (2종)           진단 (3종)       │
│  ┌─────────┐         ┌───────────────┐    ┌─────────────┐  │
│  │   SAM   │    ×    │Center-focused│  × │Original only│  │
│  │  SAM2   │         │Lesion-feature│    │With overlay │  │
│  │MedSAM2  │         └───────────────┘    │With cropped │  │
│  └─────────┘                              └─────────────┘  │
│                                                            │
│  = 3 × 2 = 6 분할 결과 per image                           │
│  = 3 진단 방법 per image (최고 분할 결과 사용)              │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 14.3 출력 파일 구조

```
outputs/
├── checkpoints/                      # 다운로드된 모델 체크포인트
│   ├── sam_vit_h_4b8939.pth         # SAM ViT-H (~2.5GB)
│   ├── sam2.1_hiera_large.pt        # SAM2 Hiera-L (~900MB)
│   └── MedSAM2_latest.pt            # MedSAM2 (~300MB)
│
├── visualizations/                   # 시각화 이미지
│   ├── 10730_4_SAM_center_focused_overlay.png
│   ├── 10730_4_SAM_center_focused_cropped.png
│   ├── 10730_4_SAM_lesion_features_overlay.png
│   ├── 10730_4_SAM2_center_focused_overlay.png
│   ├── ... (모든 조합)
│   └── 10730_4_full_comparison.png  # 전체 비교 그리드
│
├── diagnosis/                        # OpenAI 진단 결과
│   ├── 10730_4_diagnosis.json
│   ├── 10883_2_diagnosis.json
│   ├── ...
│   └── all_diagnosis_results.json   # 전체 결과 통합
│
├── segmentation_results.json         # 분할 결과 요약
└── diagnosis_report.md               # 진단 비교 보고서
```

### 14.4 JSON 출력 형식 예시

**segmentation_results.json:**
```json
[
  {
    "filename": "10730_4.png",
    "path": "/path/to/10730_4.png",
    "segmentations": {
      "SAM_center_focused": {
        "score": 0.987,
        "method": "center_point"
      },
      "SAM_lesion_features": {
        "score": 0.892,
        "method": "feature_point_0"
      },
      "SAM2_center_focused": {
        "score": 0.991,
        "method": "center_box"
      }
    },
    "mask_comparisons": {
      "SAM_center_focused_vs_SAM2_center_focused": {
        "iou": 0.856,
        "dice": 0.923
      }
    }
  }
]
```

**진단 JSON 예시:**
```json
{
  "original_only": {
    "observed_features": {
      "color": "Erythematous base with overlying white scale",
      "shape": "Irregular, roughly circular",
      "texture": "Scaly, raised plaques",
      "borders": "Well-demarcated",
      "size": "Approximately 3-4 cm diameter"
    },
    "possible_conditions": [
      {"name": "Psoriasis", "confidence": "High",
       "reasoning": "Classic silvery scale on erythematous base"},
      {"name": "Nummular eczema", "confidence": "Medium",
       "reasoning": "Coin-shaped lesion, but scale pattern differs"},
      {"name": "Tinea corporis", "confidence": "Low",
       "reasoning": "Would expect more central clearing"}
    ],
    "method": "original_only",
    "tokens_used": 1523
  },
  "with_overlay": {
    "segmentation_benefits": "The highlighted region clearly delineates
     the lesion boundary, confirming well-demarcated borders typical of psoriasis",
    ...
  }
}
```

---

## 15. 비용 및 성능 분석

### 15.1 API 비용 추정

| 진단 방법 | 이미지 수 | 토큰 (입력) | 토큰 (출력) | 예상 비용/이미지 |
|-----------|-----------|------------|------------|-----------------|
| Original only | 1장 | ~1000 | ~500 | ~$0.02 |
| With overlay | 2장 | ~2000 | ~600 | ~$0.04 |
| With cropped | 2장 | ~1800 | ~600 | ~$0.035 |
| **전체 (3방법)** | **5장** | **~4800** | **~1700** | **~$0.10** |

**100장 이미지 전체 처리 시:** 약 $10

### 15.2 처리 시간 추정

| 단계 | 시간/이미지 | 비고 |
|------|------------|------|
| SAM 분할 | ~3초 | GPU 기준 |
| SAM2 분할 | ~2초 | 더 효율적 |
| MedSAM2 분할 | ~1.5초 | Tiny 모델 |
| OpenAI API 호출 | ~5-10초 | 네트워크 지연 포함 |
| 시각화 저장 | ~1초 | I/O bound |
| **총계** | **~15-20초** | GPU + API |

---

## 16. 결론 및 향후 계획

### 16.1 현재 파이프라인 강점

1. **모듈화된 설계**: 각 컴포넌트 독립적 테스트 가능
2. **다중 모델 비교**: SAM/SAM2/MedSAM2 동시 평가
3. **다중 전략 비교**: Center vs Feature-based 전략 평가
4. **LLM 기반 검증**: GPT-4o로 분할 품질의 임상적 유용성 평가
5. **포괄적인 시각화**: 모든 결과를 시각적으로 비교

### 16.2 개선 필요 사항

| 영역 | 현재 문제 | 개선 방안 |
|------|-----------|-----------|
| Feature Detection | 성능 저조 | Saliency 기반 접근 |
| 진단 비용 | 비쌈 ($0.10/이미지) | 로컬 VLM 도입 (LLaVA-Med) |
| 처리 속도 | 느림 (~20초/이미지) | 배치 처리, 병렬화 |
| 평가 메트릭 | 정성적 | Ground Truth 도입, 정량 평가 |

### 16.3 향후 연구 방향

1. **Ground Truth 마스크 수집**: 전문가 라벨링으로 정량 평가
2. **다중 프롬프트 실험**: 다양한 LLM 프롬프트로 진단 품질 비교
3. **Fine-tuning**: MedSAM2를 피부 질환에 특화하여 추가 학습
4. **온라인 서비스화**: Gradio/Streamlit 웹 인터페이스 구축

---

*보고서 작성일: 2024*
*작성: SAM 피부 병변 분할 파이프라인 분석*
