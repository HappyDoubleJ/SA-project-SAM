# LLM-Guided SAM Pipeline 상세 문서

## 1. 개요

### 1.1 기존 파이프라인의 한계

```
기존 파이프라인:
┌─────────┐    ┌─────────────┐    ┌─────────┐
│ 이미지  │ →  │     SAM     │ →  │   LLM   │ → 진단
└─────────┘    │ (중앙 분할) │    │         │
               └─────────────┘    └─────────┘
                     ↑
              어디가 병변인지 모름
              → 중앙만 맹목적으로 분할
```

**발견된 문제점:**

| 문제 | 상세 설명 | 실제 영향 |
|------|-----------|-----------|
| **Score ≠ 품질** | SAM의 confidence score가 높아도 실제 병변을 분할했다는 보장 없음 | 배경 분할이 score 0.98을 받는 경우 발생 |
| **위치 불확실성** | Center-focused는 병변이 중앙에 없으면 실패 | 100개 중 ~40%에서 잘못된 영역 분할 |
| **다발성 병변** | 여러 병변이 있어도 하나만 분할 | 피부 질환의 분포 패턴 정보 손실 |
| **모델/전략 선택** | 어떤 모델이 좋을지 사전에 알 수 없음 | 최적이 아닌 결과가 선택됨 |

### 1.2 개선된 파이프라인 개념

```
LLM-Guided 파이프라인:
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│ 이미지  │ →  │  LLM 1  │ →  │   SAM   │ →  │  LLM 2  │ → 진단
└─────────┘    │(위치파악)│    │(정밀분할)│   │(최종진단)│
               └─────────┘    └─────────┘    └─────────┘
                    │              ↑
                    └──────────────┘
               "좌상단 30%, 50% 위치에
                붉은 반점 있음"
                → 해당 좌표로 SAM 프롬프트
```

**핵심 아이디어:**
> LLM은 "어디를 볼지" 알고, SAM은 "정밀하게 분할"할 수 있다.
> 둘을 결합하면 각자의 강점을 활용할 수 있다.

---

## 2. 이론적 근거

### 2.1 역할 분리 원칙

| 모델 | 강점 | 약점 |
|------|------|------|
| **LLM (GPT-4o)** | 의미적 이해, 위치 파악, 진단 추론 | 픽셀 단위 정밀 작업 불가 |
| **SAM** | 픽셀 단위 정밀 분할, 일관된 경계 | 의미적 이해 없음, 프롬프트 의존 |

**결합 시너지:**
```
LLM의 지능 + SAM의 정밀도 = 더 나은 결과

- LLM: "이 부분이 병변처럼 보인다" (개념적 위치)
- SAM: "해당 위치의 정확한 경계는 여기다" (픽셀 마스크)
```

### 2.2 왜 LLM에게 직접 분할을 시키지 않는가?

**LLM의 한계:**
1. LLM은 텍스트/토큰 기반 → 픽셀 좌표 출력 불가
2. "왼쪽 상단"이라고만 말할 수 있고, 정확한 경계 생성 불가
3. 출력은 항상 텍스트 형태

**SAM의 역할:**
1. LLM이 가리킨 대략적 위치를 입력받아
2. 이미지 특징을 분석하여
3. 정확한 픽셀 단위 마스크 생성

### 2.3 기대 효과

| 지표 | 기존 | 개선 후 예상 | 근거 |
|------|------|-------------|------|
| 병변 위치 정확도 | ~60% | ~85% | LLM이 병변 인식 후 위치 지정 |
| 다발성 병변 커버리지 | 1개 | 모든 병변 | LLM이 모든 병변 위치 반환 |
| 잘못된 분할률 | ~40% | ~15% | 의미적 검증 후 분할 |
| API 거부율 | ~70% (Original) | ~10% | 교육적 위치 파악 프롬프트 |

---

## 3. 파이프라인 상세 설계

### 3.1 Stage 1: 병변 위치 파악

**목적:** 진단 없이 순수하게 병변의 위치만 파악

**프롬프트 설계 의도:**
```python
LOCATION_PROMPT = """You are a medical image analysis assistant.
Your task is to LOCATE skin lesions in this image - do NOT diagnose.
...
"""
```

| 설계 요소 | 선택 | 근거 |
|-----------|------|------|
| 역할 | "medical image analysis assistant" | "dermatologist" 대신 사용 → 진단 거부 방지 |
| 지시 | "LOCATE" + "do NOT diagnose" | 명확한 역할 제한으로 API 거부 최소화 |
| 좌표계 | 퍼센트 (0-100) | 이미지 크기 독립적, 직관적 |
| 출력 형식 | JSON | 파싱 용이, 구조화된 데이터 |

**출력 예시:**
```json
{
    "lesion_count": 2,
    "lesions": [
        {
            "id": 1,
            "center_x_percent": 35,
            "center_y_percent": 45,
            "size": "medium",
            "visual_description": "erythematous patch with scaling"
        },
        {
            "id": 2,
            "center_x_percent": 70,
            "center_y_percent": 60,
            "size": "small",
            "visual_description": "hyperpigmented macule"
        }
    ],
    "image_quality": "good"
}
```

### 3.2 Stage 2: SAM 분할

**목적:** LLM이 지정한 좌표를 SAM 포인트 프롬프트로 변환하여 정밀 분할

**좌표 변환:**
```python
# 퍼센트 → 픽셀 좌표
x = int(x_percent / 100 * image_width)
y = int(y_percent / 100 * image_height)

# SAM 포인트 프롬프트로 분할
mask, score = segmenter.segment_with_point(image, (x, y))
```

**다발성 병변 처리:**
```python
for lesion in lesions:
    # 각 병변마다 개별 분할
    mask, score = segment_with_point(image, (lesion.x, lesion.y))
    results.append({"mask": mask, "lesion_id": lesion.id})

# 모든 마스크를 하나의 오버레이로 합성
overlay = create_multi_lesion_overlay(image, results)
```

**오버레이 색상 체계:**
| 병변 번호 | 색상 | RGB |
|-----------|------|-----|
| 1 | 빨강 | (255, 0, 0) |
| 2 | 초록 | (0, 255, 0) |
| 3 | 파랑 | (0, 0, 255) |
| 4 | 노랑 | (255, 255, 0) |
| ... | ... | ... |

### 3.3 Stage 3: 최종 진단

**목적:** 원본 + 분할 오버레이를 함께 보여주고 최종 진단

**프롬프트 설계:**
```python
DIAGNOSIS_PROMPT = """You are an expert dermatologist AI assistant
for EDUCATIONAL purposes.

You are provided with:
1. Original skin image
2. Segmented view highlighting the lesion area(s) identified by AI

The red/colored overlay shows the automatically detected lesion boundaries.
...
"""
```

**컨텍스트 연결:**
Stage 1에서 파악한 위치 정보를 Stage 3 프롬프트에 포함:
```python
lesion_context = f"Previously identified {lesion_count} lesion(s):"
for lesion in lesions:
    lesion_context += f"\n- Lesion #{id}: {visual_description}"

full_prompt = DIAGNOSIS_PROMPT + lesion_context
```

**이 방식의 장점:**
- LLM이 자신이 찾은 병변에 대해 일관된 분석 가능
- 분할 품질을 직접 평가하도록 요청 (`segmentation_quality` 필드)

---

## 4. 코드 구조 설명

### 4.1 클래스 구조

```python
class LLMGuidedSegmenter:
    """
    LLM이 병변 위치를 가이드하고, SAM이 정밀 분할하는 파이프라인
    """

    def __init__(self, api_key, model="gpt-4o"):
        """OpenAI 클라이언트 초기화"""

    def locate_lesions(self, image) -> Dict:
        """Stage 1: 병변 위치 파악"""

    def segment_with_locations(self, image, locations, segmenter) -> List[Dict]:
        """Stage 2: SAM 분할"""

    def create_multi_lesion_overlay(self, image, segmentation_results) -> np.ndarray:
        """다발성 병변 오버레이 생성"""

    def diagnose_with_segmentation(self, original, overlay, location_info) -> Dict:
        """Stage 3: 최종 진단"""

    def run_full_pipeline(self, image, segmenter) -> Dict:
        """전체 파이프라인 실행"""
```

### 4.2 주요 메서드 상세

#### `locate_lesions(image)`
```python
def locate_lesions(self, image: np.ndarray) -> Dict:
    """
    Stage 1: LLM이 이미지에서 병변 위치 파악

    입력: RGB numpy array
    출력: {
        "lesion_count": int,
        "lesions": [
            {
                "id": int,
                "center_x_percent": float,
                "center_y_percent": float,
                "size": str,
                "visual_description": str
            }
        ],
        "image_quality": str,
        "tokens_used": int
    }

    특징:
    - temperature=0.2로 일관된 위치 출력
    - 진단 없이 위치만 파악 → API 거부 최소화
    """
```

#### `segment_with_locations(image, locations, segmenter)`
```python
def segment_with_locations(self, image, locations, segmenter) -> List[Dict]:
    """
    Stage 2: LLM이 찾은 위치를 SAM 프롬프트로 사용

    입력:
    - image: RGB numpy array
    - locations: locate_lesions()의 결과
    - segmenter: SAM/SAM2/MedSAM2 인스턴스

    출력: [
        {
            "lesion_id": int,
            "location": {"x": int, "y": int, "x_percent": float, "y_percent": float},
            "mask": np.ndarray (H, W),
            "score": float,
            "method": "llm_guided"
        }
    ]

    특징:
    - 병변이 없으면 중앙 분할로 폴백
    - 각 병변마다 개별 분할 수행
    - 좌표 범위 검증 포함
    """
```

#### `run_full_pipeline(image, segmenter)`
```python
def run_full_pipeline(self, image, segmenter) -> Dict:
    """
    전체 파이프라인 실행

    출력: {
        "location_result": Stage 1 결과,
        "segmentation_results": Stage 2 결과 (mask 제외),
        "diagnosis_result": Stage 3 결과,
        "overlay_image": 오버레이 numpy array,
        "masks": [마스크 리스트],
        "total_tokens": 총 사용 토큰,
        "pipeline": "llm_guided"
    }
    """
```

---

## 5. 사용 방법

### 5.1 기본 사용 (Python API)

```python
from llm_guided_pipeline import LLMGuidedSegmenter
from sam_masking import SAMSegmenter, load_image, save_image

# 이미지 로드
image = load_image("skin_lesion.png")

# SAM 초기화
segmenter = SAMSegmenter(checkpoint_dir="outputs/checkpoints")

# LLM-Guided Pipeline 초기화
pipeline = LLMGuidedSegmenter()

# 전체 파이프라인 실행 (결과 자동 저장)
result = pipeline.run_full_pipeline(
    image=image,
    segmenter=segmenter,
    save_results=True,           # 결과 저장 활성화
    output_dir="outputs/llm_guided",  # 저장 디렉토리
    filename="skin_lesion.png"   # 원본 파일명
)

# 결과 확인
print(f"감지된 병변: {result['location_result']['lesion_count']}개")
print(f"진단: {result['diagnosis_result']['possible_conditions']}")
```

### 5.2 단계별 실행

```python
# Stage 1만 실행
location_result = pipeline.locate_lesions(image)
print(f"병변 위치: {location_result['lesions']}")

# Stage 2만 실행 (Stage 1 결과 필요)
seg_results = pipeline.segment_with_locations(image, location_result, segmenter)

# 오버레이 생성
overlay = pipeline.create_multi_lesion_overlay(image, seg_results)

# Stage 3만 실행
diagnosis = pipeline.diagnose_with_segmentation(image, overlay, location_result)
```

### 5.3 다른 SAM 모델 사용

```python
from sam_masking import SAM2Segmenter, MedSAM2Segmenter

# SAM2 사용
segmenter = SAM2Segmenter(checkpoint_dir="outputs/checkpoints")
result = pipeline.run_full_pipeline(image, segmenter)

# MedSAM2 사용 (의료 영상 특화)
segmenter = MedSAM2Segmenter(checkpoint_dir="outputs/checkpoints")
result = pipeline.run_full_pipeline(image, segmenter)
```

### 5.4 배치 처리 (여러 이미지)

```python
from llm_guided_pipeline import run_batch_pipeline

# 디렉토리 내 모든 이미지 처리
results = run_batch_pipeline(
    data_dir="Derm1M_v2_pretrain_ontology_sampled_100_images",
    output_dir="outputs/llm_guided",
    max_images=10,           # 최대 10개만 처리 (None이면 전체)
    segmenter_type="sam",    # "sam", "sam2", "medsam2"
    save_csv=True            # CSV 요약 저장
)
```

### 5.5 커맨드라인 실행 (CLI)

```bash
# 기본 실행 (샘플 데이터셋에서 5개 이미지 자동 처리)
python llm_guided_pipeline.py

# 전체 이미지 처리 (96개)
python llm_guided_pipeline.py --max-images 0

# 특정 개수만 처리
python llm_guided_pipeline.py --max-images 10

# 단일 이미지 처리
python llm_guided_pipeline.py --single-image <image_path>

# MedSAM2 모델 사용
python llm_guided_pipeline.py --segmenter medsam2 --max-images 10

# 커스텀 데이터 디렉토리 사용
python llm_guided_pipeline.py --data-dir /path/to/images --output-dir outputs/custom
```

**자동 데이터 탐지:**
- `--data-dir` 옵션을 생략하면 자동으로 `Derm1M_v2_pretrain_ontology_sampled_100_images` 폴더를 찾습니다
- 현재 디렉토리 → 스크립트 위치 순서로 탐색

**CLI 옵션:**

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--data-dir` | 이미지 디렉토리 경로 | 자동 탐지 |
| `--output-dir` | 결과 저장 디렉토리 | `outputs/llm_guided` |
| `--max-images` | 최대 처리 이미지 수 (0=전체) | 5 |
| `--segmenter` | 분할 모델 (sam/sam2/medsam2) | `sam` |
| `--single-image` | 단일 이미지 경로 (배치 대신) | None |
| `--no-csv` | CSV 저장 건너뛰기 | False |

**샘플 데이터셋 구조:**
```
Derm1M_v2_pretrain_ontology_sampled_100_images/
├── IIYI/       # 이미지 소스 1
├── edu/        # 교육 자료 이미지
├── note/       # 의료 노트 이미지
├── public/     # 공개 데이터셋
├── pubmed/     # PubMed 논문 이미지
└── youtube/    # YouTube 캡처
총 96개 피부 질환 이미지
```

---

## 6. 출력 파일 구조

### 6.1 디렉토리 구조

파이프라인 실행 후 생성되는 출력 디렉토리 구조:

```
outputs/llm_guided/
├── overlays/                    # 오버레이 이미지
│   ├── image1_llm_guided_overlay.png
│   ├── image2_llm_guided_overlay.png
│   └── ...
├── masks/                       # 개별 마스크 이미지
│   ├── image1_mask_0.png        # 첫 번째 병변 마스크
│   ├── image1_mask_1.png        # 두 번째 병변 마스크
│   └── ...
├── comparisons/                 # 원본-오버레이 비교 이미지
│   ├── image1_comparison.png
│   └── ...
├── json/                        # 개별 결과 JSON
│   ├── image1_result.json
│   └── ...
├── checkpoints/                 # 모델 체크포인트 (자동 다운로드)
├── all_results.json             # 전체 결과 통합 JSON
├── diagnosis_summary.csv        # CSV 요약 테이블
└── diagnosis_report.md          # 마크다운 리포트
```

### 6.2 개별 결과 JSON 구조

`json/<filename>_result.json`:

```json
{
    "filename": "skin_lesion.png",
    "location_result": {
        "lesion_count": 2,
        "lesions": [
            {
                "id": 1,
                "center_x_percent": 35.5,
                "center_y_percent": 48.2,
                "size": "medium",
                "visual_description": "erythematous patch"
            }
        ],
        "image_quality": "good",
        "tokens_used": 450
    },
    "segmentation_results": [
        {
            "lesion_id": 1,
            "location": {"x": 284, "y": 386, "x_percent": 35.5, "y_percent": 48.2},
            "score": 0.9823,
            "method": "llm_guided"
        }
    ],
    "diagnosis_result": {
        "observed_features": {
            "color": "pink to red",
            "shape": "oval",
            "texture": "scaly surface",
            "borders": "well-defined"
        },
        "possible_conditions": [
            {"name": "Psoriasis", "confidence": "High"},
            {"name": "Eczema", "confidence": "Medium"}
        ],
        "segmentation_quality": {
            "accuracy": "good",
            "comments": "Accurately captured lesion boundaries"
        },
        "tokens_used": 1200
    },
    "total_tokens": 1650,
    "pipeline": "llm_guided",
    "timestamp": "2024-12-03T14:30:25.123456"
}
```

### 6.3 CSV 요약 구조

`diagnosis_summary.csv`:

| filename | lesion_count | image_quality | top_diagnosis | confidence | segmentation_accuracy | color | shape | tokens_used |
|----------|--------------|---------------|---------------|------------|----------------------|-------|-------|-------------|
| img1.png | 2 | good | Psoriasis | High | good | pink | oval | 1650 |
| img2.png | 1 | moderate | Eczema | Medium | partial | red | irregular | 1420 |
| img3.png | 0 | poor | Unknown | N/A | N/A | N/A | N/A | 450 |

### 6.4 마크다운 리포트

`diagnosis_report.md` 내용:

```markdown
# LLM-Guided SAM Pipeline 진단 리포트

생성 시간: 2024-12-03 14:35:00

## 요약
- 총 분석 이미지: 100
- 성공: 98
- 실패: 2
- 총 토큰 사용량: 165,000
- 총 감지 병변 수: 142

## 진단 결과 분포
- Psoriasis: 25건
- Eczema: 18건
- Contact Dermatitis: 12건
...

## 개별 분석 결과
### image1.png
**병변 수**: 2
**진단 결과**:
- Psoriasis (High)
- Eczema (Medium)
...
```

---

## 7. 기존 파이프라인과의 비교

### 7.1 처리 흐름 비교

**기존:**
```
이미지 → SAM(중앙) → 최고 score 선택 → LLM 진단
         SAM(특징)
         SAM2(중앙)
         SAM2(특징)
         MedSAM2(중앙)
         MedSAM2(특징)
                    ↓
            6개 중 score 기준 1개 선택
                    ↓
            (잘못된 것이 선택될 수 있음)
```

**개선:**
```
이미지 → LLM(위치 파악) → SAM(해당 위치) → LLM(진단)
              ↓                 ↓
        "여기가 병변"      정확한 해당 위치 분할
```

### 7.2 API 호출 비교

| 방식 | API 호출 | 이미지 수 | 예상 비용 |
|------|----------|-----------|-----------|
| 기존 (3가지 진단) | 3회 | 5장 | ~$0.10 |
| **LLM-Guided** | **2회** | **3장** | **~$0.07** |

### 7.3 예상 정확도 비교

| 상황 | 기존 | LLM-Guided |
|------|------|------------|
| 병변이 중앙 | ✅ 좋음 | ✅ 좋음 |
| 병변이 중앙 아님 | ❌ 실패 가능 | ✅ LLM이 찾음 |
| 다발성 병변 | ⚠️ 하나만 | ✅ 모두 분할 |
| 잘못된 분할 선택 | ⚠️ Score 의존 | ✅ LLM 검증 |

---

## 8. 한계점 및 향후 개선

### 8.1 현재 한계

| 한계 | 설명 | 가능한 해결책 |
|------|------|---------------|
| LLM 위치 오류 | LLM이 병변 위치를 잘못 파악할 수 있음 | 여러 LLM 앙상블 |
| API 비용 | 여전히 이미지당 ~$0.07 | 로컬 VLM 도입 |
| 좌표 정밀도 | 퍼센트 단위로 한계 있음 | 박스 좌표 추가 |
| 작은 병변 | LLM이 작은 병변을 놓칠 수 있음 | 멀티스케일 분석 |

### 8.2 향후 개선 방향

1. **앙상블 위치 파악**
   ```python
   # 여러 프롬프트/모델로 위치 파악 후 평균
   locations_1 = locate_with_prompt_a(image)
   locations_2 = locate_with_prompt_b(image)
   final_locations = ensemble_locations([locations_1, locations_2])
   ```

2. **박스 프롬프트 추가**
   ```python
   # 포인트 대신 박스 프롬프트 사용
   {
       "lesion": {
           "box": {
               "x1_percent": 20, "y1_percent": 30,
               "x2_percent": 40, "y2_percent": 50
           }
       }
   }
   ```

3. **피드백 루프**
   ```python
   # Stage 3에서 분할 품질이 "poor"면 재시도
   if diagnosis["segmentation_quality"]["accuracy"] == "poor":
       # 프롬프트 조정 후 재분할
       retry_with_adjusted_location()
   ```

4. **로컬 VLM 통합**
   ```python
   # LLaVA-Med 등 로컬 모델로 비용 절감
   class LocalVLMGuidedSegmenter(LLMGuidedSegmenter):
       def __init__(self):
           self.model = load_local_vlm("llava-med")
   ```

---

## 9. 결론

### 9.1 핵심 혁신

> **"LLM의 지능으로 SAM을 가이드한다"**

기존 파이프라인에서 SAM은 "어디가 병변인지" 모른 채 분할했습니다.
개선된 파이프라인에서는 LLM이 먼저 병변 위치를 파악하고,
SAM은 그 위치를 정밀하게 분할하는 역할만 수행합니다.

### 9.2 기대 효과 요약

| 지표 | 개선 효과 |
|------|-----------|
| 분할 정확도 | +25~30% |
| 다발성 병변 처리 | 1개 → 모두 |
| API 비용 | -30% |
| 잘못된 분할 선택 | -60% |

### 9.3 권장 사용 시나리오

- ✅ 병변 위치가 불확실한 이미지
- ✅ 다발성 병변이 있는 이미지
- ✅ 정밀한 분할 품질이 필요한 경우
- ✅ 진단 정확도가 중요한 연구/교육 목적

---

*문서 작성일: 2024-12-03*
*LLM-Guided SAM Pipeline v1.1*
*업데이트: 결과 저장, 배치 처리, CLI 기능 추가*
