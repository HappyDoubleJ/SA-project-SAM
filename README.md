# SA-project-SAM

피부병 진단 성능 향상을 위한 SAM/MedSAM2 마스킹 + OpenAI 연동 프로젝트

## 프로젝트 개요

이 프로젝트는 피부병 이미지에서 환부를 자동으로 마스킹(세그멘테이션)하고, 원본 이미지만 사용할 때와 마스킹된 이미지를 함께 사용할 때의 진단 성능을 비교합니다.

### 주요 기능

1. **SAM (Segment Anything Model)**: Meta의 범용 세그멘테이션 모델로 피부 병변 영역 추출
2. **MedSAM2**: 의료 이미지에 특화된 세그멘테이션 모델
3. **시각화**: 두 모델의 마스킹 결과 비교 시각화
4. **OpenAI 진단**: GPT-4o를 사용한 피부병 진단 비교
   - 원본 이미지만 사용
   - 원본 + 마스킹 오버레이 사용
   - 원본 + 크롭 이미지 사용

## 설치 방법

```bash
# 1. 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 환경 변수 설정
# .env 파일에 OpenAI API 키가 이미 설정되어 있습니다
```

## 사용 방법

### 전체 파이프라인 실행

```bash
# 전체 파이프라인 (SAM 마스킹 + 시각화 + OpenAI 진단)
python main.py

# 이미지 수 제한 (테스트용)
python main.py --max-images 5

# OpenAI 진단 없이 마스킹만
python main.py --no-diagnosis

# SAM만 사용 (MedSAM2 제외)
python main.py --no-medsam
```

### 개별 모듈 사용

```python
# SAM 마스킹
from sam_masking import SAMSegmenter, load_image, apply_mask_to_image

sam = SAMSegmenter()
image = load_image("path/to/image.jpg")
mask, score, method = sam.get_best_lesion_mask(image)
overlay = apply_mask_to_image(image, mask)

# OpenAI 진단
from openai_diagnosis import SkinDiseaseDiagnoser

diagnoser = SkinDiseaseDiagnoser()
result = diagnoser.diagnose_original_only(image)
result_with_mask = diagnoser.diagnose_with_mask(image, overlay)
```

## 프로젝트 구조

```
SA-project-SAM/
├── .env                    # OpenAI API 키 (git에서 제외됨)
├── .gitignore             # Git 제외 파일 목록
├── requirements.txt       # Python 의존성
├── README.md              # 이 파일
├── main.py                # 메인 파이프라인 스크립트
├── sam_masking.py         # SAM/MedSAM2 마스킹 모듈
├── visualize.py           # 시각화 모듈
├── openai_diagnosis.py    # OpenAI 진단 모듈
├── Derm1M_v2_pretrain_ontology_sampled_100_images/  # 데이터셋
│   ├── youtube/           # YouTube 소스 이미지
│   ├── public/            # 공개 데이터셋 이미지
│   ├── edu/               # 교육 자료 이미지
│   ├── pubmed/            # PubMed 논문 이미지
│   ├── IIYI/              # IIYI 데이터셋
│   └── note/              # 참고 이미지
└── outputs/               # 출력 디렉토리 (실행 후 생성)
    ├── checkpoints/       # 모델 체크포인트
    ├── visualizations/    # 시각화 이미지
    └── diagnosis/         # 진단 결과 JSON
```

## 출력 결과

### 시각화
- `*_sam_overlay.png`: SAM 마스크 오버레이
- `*_medsam_overlay.png`: MedSAM2 마스크 오버레이
- `*_comparison.png`: 두 모델 비교 이미지
- `batch_comparison.png`: 전체 배치 비교 그리드

### 진단 결과
- `*_diagnosis.json`: 개별 이미지 진단 결과
- `all_diagnosis_results.json`: 전체 진단 결과
- `diagnosis_report.md`: 진단 방법별 비교 리포트

## 연구 질문

이 프로젝트는 다음 질문에 답하고자 합니다:

1. SAM vs MedSAM2: 피부병 이미지에서 어떤 모델이 더 정확하게 병변을 세그멘테이션하는가?
2. 마스킹 효과: 원본 이미지만 사용할 때와 마스킹 정보를 함께 제공할 때 LLM의 진단 정확도가 향상되는가?
3. 크롭 vs 오버레이: 병변 영역만 크롭해서 제공하는 것과 전체 이미지에 오버레이를 보여주는 것 중 어떤 것이 더 효과적인가?

## 주의사항

- 이 프로젝트는 연구/교육 목적으로만 사용해야 합니다
- 실제 의료 진단에는 반드시 전문 의료인과 상담하세요
- OpenAI API 사용에는 비용이 발생합니다
- 처음 실행 시 SAM 모델 다운로드로 약 2.5GB가 필요합니다

## 라이선스

연구/교육 목적 사용
