"""
LLM-Guided SAM Pipeline for Skin Disease Diagnosis

개선된 파이프라인:
1. LLM이 먼저 병변 위치를 파악 (진단 X, 위치만)
2. LLM이 알려준 위치를 SAM 프롬프트로 사용
3. SAM이 정밀 분할 수행
4. 분할 결과와 함께 LLM에게 최종 진단 요청

기존 방식의 문제점:
- SAM이 병변 위치를 모른 채 중앙만 분할
- Score가 높아도 실제 품질이 낮을 수 있음
- 다발성 병변 시 하나만 분할됨

개선된 방식의 장점:
- LLM이 병변 위치를 지능적으로 파악
- SAM은 정밀한 픽셀 단위 분할에 집중
- 다발성 병변도 각각 처리 가능
"""

import os
import json
import base64
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
from io import BytesIO
from PIL import Image
import numpy as np

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class LLMGuidedSegmenter:
    """
    LLM이 병변 위치를 가이드하고, SAM이 정밀 분할하는 파이프라인

    Flow:
    1. locate_lesions(): LLM이 이미지에서 병변 위치 파악
    2. segment_with_locations(): SAM이 해당 위치들을 정밀 분할
    3. diagnose_with_segmentation(): 분할 결과로 최종 진단
    """

    # Stage 1: 병변 위치 파악 프롬프트 (진단 아님)
    LOCATION_PROMPT = """You are a medical image analysis assistant.
Your task is to LOCATE skin lesions in this image - do NOT diagnose.

Analyze the image and identify all visible skin abnormalities.
For each lesion found, provide:
1. Location in the image (use percentage coordinates where 0,0 is top-left and 100,100 is bottom-right)
2. Approximate size relative to image (small/medium/large)
3. Brief visual description (color, shape - NO diagnosis)

Respond in JSON format:
{
    "lesion_count": <number>,
    "lesions": [
        {
            "id": 1,
            "center_x_percent": <0-100>,
            "center_y_percent": <0-100>,
            "size": "small|medium|large",
            "visual_description": "brief description of appearance"
        }
    ],
    "image_quality": "good|moderate|poor",
    "notes": "any relevant observations about the image"
}

If no clear lesions are visible, return lesion_count: 0.
Remember: ONLY locate and describe visually, do NOT diagnose."""

    # Stage 3: 최종 진단 프롬프트
    DIAGNOSIS_PROMPT = """You are an expert dermatologist AI assistant for EDUCATIONAL purposes.

You are provided with:
1. Original skin image
2. Segmented view highlighting the lesion area(s) identified by AI

The red/colored overlay shows the automatically detected lesion boundaries.

Analyze both images and provide:
1. **Observed Features**: Describe visible characteristics (color, shape, texture, borders, distribution)
2. **Morphological Analysis**: Classify the lesion type (macule, papule, plaque, vesicle, etc.)
3. **Differential Diagnosis**: List top 3-5 possible conditions with confidence levels
4. **Key Findings**: What features support or rule out each diagnosis
5. **Segmentation Quality**: Did the AI segmentation accurately capture the lesion?

IMPORTANT: This is for educational/research purposes only.

Respond in JSON format:
{
    "observed_features": {
        "color": "",
        "shape": "",
        "texture": "",
        "borders": "",
        "distribution": "",
        "size_estimate": ""
    },
    "morphology": "",
    "possible_conditions": [
        {"name": "", "confidence": "High|Medium|Low", "supporting_features": [], "against_features": []}
    ],
    "segmentation_quality": {
        "accuracy": "good|partial|poor",
        "comments": ""
    },
    "primary_impression": "",
    "recommended_actions": []
}"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        """초기화"""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.sam_segmenter = None  # Lazy loading

    def _encode_image(self, image: np.ndarray) -> str:
        """numpy 배열을 base64로 인코딩"""
        pil_image = Image.fromarray(image)
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _parse_json_response(self, text: str) -> Dict:
        """LLM 응답에서 JSON 추출"""
        try:
            # JSON 블록 찾기
            json_start = text.find('{')
            json_end = text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                return json.loads(text[json_start:json_end])
        except json.JSONDecodeError:
            pass
        return {"raw_response": text, "parse_error": True}

    def locate_lesions(self, image: np.ndarray) -> Dict:
        """
        Stage 1: LLM이 이미지에서 병변 위치 파악

        Args:
            image: RGB numpy array

        Returns:
            {
                "lesion_count": int,
                "lesions": [{"id", "center_x_percent", "center_y_percent", "size", "visual_description"}],
                "image_quality": str,
                "notes": str
            }
        """
        print("  [Stage 1] LLM이 병변 위치 파악 중...")

        base64_image = self._encode_image(image)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": self.LOCATION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high"
                        }
                    }
                ]
            }],
            max_tokens=1000,
            temperature=0.2  # 낮은 temperature로 일관성 확보
        )

        result = self._parse_json_response(response.choices[0].message.content)
        result["tokens_used"] = response.usage.total_tokens

        lesion_count = result.get("lesion_count", 0)
        print(f"    → {lesion_count}개 병변 위치 파악됨")

        return result

    def segment_with_locations(self, image: np.ndarray,
                                locations: Dict,
                                segmenter) -> List[Dict]:
        """
        Stage 2: LLM이 찾은 위치를 SAM 프롬프트로 사용하여 분할

        Args:
            image: RGB numpy array
            locations: locate_lesions()의 결과
            segmenter: SAMSegmenter, SAM2Segmenter, 또는 MedSAM2Segmenter 인스턴스

        Returns:
            List of segmentation results for each lesion
        """
        print("  [Stage 2] SAM이 지정된 위치 분할 중...")

        h, w = image.shape[:2]
        results = []

        lesions = locations.get("lesions", [])

        if not lesions:
            print("    → 병변이 감지되지 않음, 중앙 분할로 폴백")
            # 폴백: 중앙 분할
            center = (w // 2, h // 2)
            mask, score = segmenter.segment_with_point(image, center)
            results.append({
                "lesion_id": 0,
                "location": {"x": center[0], "y": center[1]},
                "mask": mask,
                "score": float(score),
                "method": "fallback_center"
            })
            return results

        for lesion in lesions:
            lesion_id = lesion.get("id", len(results) + 1)

            # 퍼센트 좌표를 픽셀 좌표로 변환
            x_percent = lesion.get("center_x_percent", 50)
            y_percent = lesion.get("center_y_percent", 50)

            x = int(x_percent / 100 * w)
            y = int(y_percent / 100 * h)

            # 좌표 범위 제한
            x = max(0, min(w - 1, x))
            y = max(0, min(h - 1, y))

            print(f"    → 병변 #{lesion_id}: ({x}, {y}) 분할 중...")

            try:
                mask, score = segmenter.segment_with_point(image, (x, y))

                results.append({
                    "lesion_id": lesion_id,
                    "location": {"x": x, "y": y, "x_percent": x_percent, "y_percent": y_percent},
                    "visual_description": lesion.get("visual_description", ""),
                    "size": lesion.get("size", "unknown"),
                    "mask": mask,
                    "score": float(score),
                    "method": "llm_guided"
                })
                print(f"       Score: {score:.3f}")

            except Exception as e:
                print(f"       분할 실패: {e}")
                results.append({
                    "lesion_id": lesion_id,
                    "location": {"x": x, "y": y},
                    "error": str(e)
                })

        return results

    def create_multi_lesion_overlay(self, image: np.ndarray,
                                     segmentation_results: List[Dict],
                                     alpha: float = 0.4) -> np.ndarray:
        """
        여러 병변의 분할 결과를 하나의 오버레이로 합성

        각 병변마다 다른 색상 사용
        """
        # 병변별 색상 (최대 10개)
        colors = [
            (255, 0, 0),    # 빨강
            (0, 255, 0),    # 초록
            (0, 0, 255),    # 파랑
            (255, 255, 0),  # 노랑
            (255, 0, 255),  # 마젠타
            (0, 255, 255),  # 시안
            (255, 128, 0),  # 주황
            (128, 0, 255),  # 보라
            (0, 255, 128),  # 민트
            (255, 128, 128) # 연분홍
        ]

        result = image.copy().astype(float)

        for i, seg_result in enumerate(segmentation_results):
            if "mask" not in seg_result or seg_result.get("error"):
                continue

            mask = seg_result["mask"].astype(bool)
            color = colors[i % len(colors)]

            for c in range(3):
                result[:, :, c] = np.where(
                    mask,
                    result[:, :, c] * (1 - alpha) + color[c] * alpha,
                    result[:, :, c]
                )

        return result.astype(np.uint8)

    def diagnose_with_segmentation(self, original: np.ndarray,
                                    overlay: np.ndarray,
                                    location_info: Dict) -> Dict:
        """
        Stage 3: 분할 결과와 함께 최종 진단 요청

        Args:
            original: 원본 이미지
            overlay: 분할 오버레이가 적용된 이미지
            location_info: Stage 1에서 얻은 위치 정보 (컨텍스트용)

        Returns:
            진단 결과 딕셔너리
        """
        print("  [Stage 3] LLM 최종 진단 중...")

        base64_original = self._encode_image(original)
        base64_overlay = self._encode_image(overlay)

        # 위치 정보를 컨텍스트로 추가
        lesion_context = ""
        if location_info.get("lesions"):
            lesion_context = f"\n\nPreviously identified {location_info['lesion_count']} lesion(s):"
            for lesion in location_info["lesions"]:
                lesion_context += f"\n- Lesion #{lesion['id']}: {lesion.get('visual_description', 'N/A')}"

        full_prompt = self.DIAGNOSIS_PROMPT + lesion_context

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": full_prompt},
                    {"type": "text", "text": "Image 1 - Original:"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_original}",
                            "detail": "high"
                        }
                    },
                    {"type": "text", "text": "Image 2 - AI Segmentation Overlay:"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_overlay}",
                            "detail": "high"
                        }
                    }
                ]
            }],
            max_tokens=2000,
            temperature=0.3
        )

        result = self._parse_json_response(response.choices[0].message.content)
        result["tokens_used"] = response.usage.total_tokens
        result["method"] = "llm_guided_pipeline"

        print(f"    → 진단 완료")

        return result

    def run_full_pipeline(self, image: np.ndarray,
                          segmenter,
                          save_intermediates: bool = False,
                          output_dir: Optional[str] = None) -> Dict:
        """
        전체 파이프라인 실행

        Args:
            image: RGB numpy array
            segmenter: SAM/SAM2/MedSAM2 segmenter 인스턴스
            save_intermediates: 중간 결과 저장 여부
            output_dir: 저장 디렉토리

        Returns:
            {
                "location_result": Stage 1 결과,
                "segmentation_results": Stage 2 결과,
                "diagnosis_result": Stage 3 결과,
                "overlay_image": 오버레이 이미지,
                "total_tokens": 총 사용 토큰
            }
        """
        print("\n=== LLM-Guided SAM Pipeline 시작 ===")

        # Stage 1: 병변 위치 파악
        location_result = self.locate_lesions(image)

        # Stage 2: SAM 분할
        segmentation_results = self.segment_with_locations(
            image, location_result, segmenter
        )

        # 오버레이 생성
        overlay = self.create_multi_lesion_overlay(image, segmentation_results)

        # Stage 3: 최종 진단
        diagnosis_result = self.diagnose_with_segmentation(
            image, overlay, location_result
        )

        # 결과 종합
        total_tokens = (
            location_result.get("tokens_used", 0) +
            diagnosis_result.get("tokens_used", 0)
        )

        result = {
            "location_result": location_result,
            "segmentation_results": [
                {k: v for k, v in seg.items() if k != "mask"}  # mask 제외 (저장용)
                for seg in segmentation_results
            ],
            "diagnosis_result": diagnosis_result,
            "overlay_image": overlay,
            "masks": [seg.get("mask") for seg in segmentation_results if "mask" in seg],
            "total_tokens": total_tokens,
            "pipeline": "llm_guided"
        }

        print(f"\n=== 파이프라인 완료 (총 {total_tokens} 토큰 사용) ===")

        return result


def compare_pipelines(image: np.ndarray,
                      segmenter,
                      guided_pipeline: LLMGuidedSegmenter,
                      legacy_diagnoser) -> Dict:
    """
    기존 파이프라인과 개선된 파이프라인 비교

    Args:
        image: 입력 이미지
        segmenter: SAM segmenter
        guided_pipeline: LLMGuidedSegmenter 인스턴스
        legacy_diagnoser: 기존 SkinDiseaseDiagnoser 인스턴스

    Returns:
        비교 결과
    """
    from sam_masking import apply_mask_to_image

    results = {
        "legacy": {},
        "llm_guided": {},
        "comparison": {}
    }

    print("\n" + "="*60)
    print("기존 파이프라인 실행 (Center-focused)")
    print("="*60)

    # 기존 방식: Center-focused
    legacy_seg = segmenter.segment_center_focused(image)
    legacy_overlay = apply_mask_to_image(image, legacy_seg["mask"])

    legacy_diagnosis = legacy_diagnoser.diagnose_with_mask(
        image, legacy_overlay
    )

    results["legacy"] = {
        "segmentation": {
            "method": legacy_seg["method"],
            "score": legacy_seg["score"]
        },
        "diagnosis": legacy_diagnosis
    }

    print("\n" + "="*60)
    print("개선된 파이프라인 실행 (LLM-Guided)")
    print("="*60)

    # 개선된 방식: LLM-Guided
    guided_result = guided_pipeline.run_full_pipeline(image, segmenter)

    results["llm_guided"] = {
        "location": guided_result["location_result"],
        "segmentation": guided_result["segmentation_results"],
        "diagnosis": guided_result["diagnosis_result"]
    }

    # 비교 분석
    results["comparison"] = {
        "legacy_tokens": legacy_diagnosis.get("tokens_used", 0),
        "guided_tokens": guided_result["total_tokens"],
        "lesions_found": guided_result["location_result"].get("lesion_count", 0),
        "legacy_top_diagnosis": _get_top_diagnosis(legacy_diagnosis),
        "guided_top_diagnosis": _get_top_diagnosis(guided_result["diagnosis_result"])
    }

    return results


def _get_top_diagnosis(diagnosis: Dict) -> str:
    """진단 결과에서 최상위 진단명 추출"""
    conditions = diagnosis.get("possible_conditions", [])
    if conditions and isinstance(conditions, list) and len(conditions) > 0:
        first = conditions[0]
        if isinstance(first, dict):
            return first.get("name", "Unknown")
    return "Unknown"


if __name__ == "__main__":
    # 테스트
    from sam_masking import SAMSegmenter, load_image
    import sys

    if len(sys.argv) < 2:
        print("Usage: python llm_guided_pipeline.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    image = load_image(image_path)

    print(f"이미지 로드: {image_path}")
    print(f"크기: {image.shape}")

    # SAM 초기화
    segmenter = SAMSegmenter(checkpoint_dir="outputs/checkpoints")

    # LLM-Guided Pipeline 초기화
    pipeline = LLMGuidedSegmenter()

    # 파이프라인 실행
    result = pipeline.run_full_pipeline(image, segmenter)

    # 결과 출력
    print("\n=== 결과 요약 ===")
    print(f"감지된 병변 수: {result['location_result'].get('lesion_count', 0)}")

    if result['diagnosis_result'].get('possible_conditions'):
        print("\n진단 결과:")
        for cond in result['diagnosis_result']['possible_conditions'][:3]:
            if isinstance(cond, dict):
                print(f"  - {cond.get('name', 'N/A')} ({cond.get('confidence', 'N/A')})")

    # 오버레이 저장
    from sam_masking import save_image
    output_path = image_path.replace('.', '_llm_guided.')
    save_image(result['overlay_image'], output_path)
    print(f"\n오버레이 저장: {output_path}")
