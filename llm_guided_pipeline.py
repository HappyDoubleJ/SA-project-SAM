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
import sys
import json
import base64
import argparse
import csv
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
from datetime import datetime
from io import BytesIO
from PIL import Image
import numpy as np
from tqdm import tqdm

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
                          save_results: bool = False,
                          output_dir: Optional[str] = None,
                          filename: Optional[str] = None) -> Dict:
        """
        전체 파이프라인 실행

        Args:
            image: RGB numpy array
            segmenter: SAM/SAM2/MedSAM2 segmenter 인스턴스
            save_results: 결과 저장 여부
            output_dir: 저장 디렉토리
            filename: 원본 파일명 (저장 시 사용)

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
            "filename": filename or "unknown",
            "location_result": location_result,
            "segmentation_results": [
                {k: v for k, v in seg.items() if k != "mask"}  # mask 제외 (저장용)
                for seg in segmentation_results
            ],
            "diagnosis_result": diagnosis_result,
            "overlay_image": overlay,
            "masks": [seg.get("mask") for seg in segmentation_results if "mask" in seg],
            "total_tokens": total_tokens,
            "pipeline": "llm_guided",
            "timestamp": datetime.now().isoformat()
        }

        # 결과 저장
        if save_results and output_dir:
            self._save_results(result, image, output_dir, filename)

        print(f"\n=== 파이프라인 완료 (총 {total_tokens} 토큰 사용) ===")

        return result

    def _save_results(self, result: Dict, original: np.ndarray,
                      output_dir: str, filename: Optional[str] = None):
        """
        결과를 파일로 저장

        Args:
            result: 파이프라인 실행 결과
            original: 원본 이미지
            output_dir: 저장 디렉토리
            filename: 원본 파일명
        """
        from sam_masking import save_image

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        stem = Path(filename).stem if filename else "image"

        # 1. 오버레이 이미지 저장
        overlay_dir = output_path / "overlays"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        save_image(result["overlay_image"], str(overlay_dir / f"{stem}_llm_guided_overlay.png"))

        # 2. 개별 마스크 저장
        masks_dir = output_path / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        for idx, mask in enumerate(result.get("masks", [])):
            if mask is not None:
                mask_img = (mask * 255).astype(np.uint8)
                save_image(mask_img, str(masks_dir / f"{stem}_mask_{idx}.png"))

        # 3. JSON 결과 저장
        json_dir = output_path / "json"
        json_dir.mkdir(parents=True, exist_ok=True)

        # mask와 overlay_image는 저장에서 제외
        json_result = {
            k: v for k, v in result.items()
            if k not in ["overlay_image", "masks"]
        }

        with open(json_dir / f"{stem}_result.json", 'w', encoding='utf-8') as f:
            json.dump(json_result, f, indent=2, ensure_ascii=False, default=str)

        # 4. 비교 이미지 저장 (원본 + 오버레이)
        comparison_dir = output_path / "comparisons"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        self._save_comparison_figure(
            original, result["overlay_image"],
            result.get("location_result", {}),
            str(comparison_dir / f"{stem}_comparison.png")
        )

        print(f"    → 결과 저장됨: {output_path}")

    def _save_comparison_figure(self, original: np.ndarray, overlay: np.ndarray,
                                 location_info: Dict, save_path: str):
        """원본과 오버레이 비교 이미지 생성"""
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            axes[0].imshow(original)
            axes[0].set_title("Original Image", fontsize=12, fontweight='bold')
            axes[0].axis('off')

            axes[1].imshow(overlay)
            lesion_count = location_info.get("lesion_count", 0)
            axes[1].set_title(f"LLM-Guided Segmentation ({lesion_count} lesions)",
                             fontsize=12, fontweight='bold')
            axes[1].axis('off')

            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
        except Exception as e:
            print(f"    비교 이미지 저장 실패: {e}")


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


def get_image_files(data_dir: str, extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp')) -> List[Path]:
    """디렉토리에서 이미지 파일 목록 가져오기"""
    data_path = Path(data_dir)
    image_files = []

    for ext in extensions:
        image_files.extend(data_path.rglob(f"*{ext}"))
        image_files.extend(data_path.rglob(f"*{ext.upper()}"))

    return sorted(image_files)


def run_batch_pipeline(
    data_dir: str,
    output_dir: str,
    max_images: Optional[int] = None,
    segmenter_type: str = "sam",
    save_csv: bool = True
) -> List[Dict]:
    """
    배치 처리 파이프라인 실행

    Args:
        data_dir: 이미지 디렉토리
        output_dir: 결과 저장 디렉토리
        max_images: 최대 처리 이미지 수
        segmenter_type: 사용할 분할 모델 (sam, sam2, medsam2)
        save_csv: CSV 요약 저장 여부

    Returns:
        전체 결과 리스트
    """
    from sam_masking import SAMSegmenter, SAM2Segmenter, MedSAM2Segmenter, load_image

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 이미지 파일 목록
    image_files = get_image_files(data_dir)
    if max_images:
        image_files = image_files[:max_images]

    print(f"\n{'='*60}")
    print(f"LLM-Guided SAM Batch Pipeline")
    print(f"{'='*60}")
    print(f"이미지 수: {len(image_files)}")
    print(f"분할 모델: {segmenter_type.upper()}")
    print(f"출력 디렉토리: {output_path}")
    print(f"{'='*60}\n")

    # Segmenter 초기화
    checkpoint_dir = str(output_path / "checkpoints")

    print(f"[{segmenter_type.upper()}] 모델 초기화 중...")
    if segmenter_type.lower() == "sam":
        segmenter = SAMSegmenter(checkpoint_dir=checkpoint_dir)
    elif segmenter_type.lower() == "sam2":
        segmenter = SAM2Segmenter(checkpoint_dir=checkpoint_dir)
    elif segmenter_type.lower() == "medsam2":
        segmenter = MedSAM2Segmenter(checkpoint_dir=checkpoint_dir)
    else:
        print(f"지원하지 않는 모델: {segmenter_type}")
        sys.exit(1)

    # LLM-Guided Pipeline 초기화
    pipeline = LLMGuidedSegmenter()

    # 결과 저장
    all_results = []
    csv_rows = []

    for idx, image_path in enumerate(tqdm(image_files, desc="Processing")):
        print(f"\n[{idx+1}/{len(image_files)}] {image_path.name}")

        try:
            image = load_image(str(image_path))

            result = pipeline.run_full_pipeline(
                image=image,
                segmenter=segmenter,
                save_results=True,
                output_dir=str(output_path),
                filename=image_path.name
            )

            all_results.append(result)

            # CSV 행 추가
            csv_row = _extract_csv_row(result, image_path.name)
            csv_rows.append(csv_row)

        except Exception as e:
            print(f"  오류 발생: {e}")
            all_results.append({
                "filename": image_path.name,
                "error": str(e)
            })
            csv_rows.append({
                "filename": image_path.name,
                "error": str(e),
                "lesion_count": 0,
                "top_diagnosis": "Error",
                "confidence": "N/A",
                "tokens_used": 0
            })

    # 전체 결과 JSON 저장
    summary_path = output_path / "all_results.json"
    json_results = [
        {k: v for k, v in r.items() if k not in ["overlay_image", "masks"]}
        for r in all_results
    ]
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n전체 결과 저장: {summary_path}")

    # CSV 저장
    if save_csv:
        csv_path = output_path / "diagnosis_summary.csv"
        _save_csv(csv_rows, csv_path)
        print(f"CSV 요약 저장: {csv_path}")

    # 마크다운 리포트 생성
    report_path = output_path / "diagnosis_report.md"
    _generate_report(all_results, report_path)
    print(f"리포트 저장: {report_path}")

    # 요약 출력
    _print_summary(all_results)

    return all_results


def _extract_csv_row(result: Dict, filename: str) -> Dict:
    """결과에서 CSV 행 추출"""
    row = {
        "filename": filename,
        "lesion_count": result.get("location_result", {}).get("lesion_count", 0),
        "image_quality": result.get("location_result", {}).get("image_quality", "N/A"),
        "tokens_used": result.get("total_tokens", 0),
    }

    # 진단 결과 추출
    diagnosis = result.get("diagnosis_result", {})
    conditions = diagnosis.get("possible_conditions", [])

    if conditions and isinstance(conditions, list) and len(conditions) > 0:
        top = conditions[0]
        if isinstance(top, dict):
            row["top_diagnosis"] = top.get("name", "Unknown")
            row["confidence"] = top.get("confidence", "N/A")
        else:
            row["top_diagnosis"] = str(top)
            row["confidence"] = "N/A"
    else:
        row["top_diagnosis"] = "Unknown"
        row["confidence"] = "N/A"

    # 세그멘테이션 품질
    seg_quality = diagnosis.get("segmentation_quality", {})
    row["segmentation_accuracy"] = seg_quality.get("accuracy", "N/A")

    # 주요 특징
    features = diagnosis.get("observed_features", {})
    row["color"] = features.get("color", "N/A")
    row["shape"] = features.get("shape", "N/A")

    return row


def _save_csv(rows: List[Dict], path: Path):
    """CSV 파일 저장"""
    if not rows:
        return

    fieldnames = [
        "filename", "lesion_count", "image_quality", "top_diagnosis",
        "confidence", "segmentation_accuracy", "color", "shape", "tokens_used"
    ]

    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def _generate_report(results: List[Dict], path: Path):
    """마크다운 리포트 생성"""
    lines = [
        "# LLM-Guided SAM Pipeline 진단 리포트",
        "",
        f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 요약",
        "",
        f"- 총 분석 이미지: {len(results)}",
    ]

    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    lines.append(f"- 성공: {len(successful)}")
    lines.append(f"- 실패: {len(failed)}")
    lines.append("")

    if successful:
        total_tokens = sum(r.get("total_tokens", 0) for r in successful)
        total_lesions = sum(r.get("location_result", {}).get("lesion_count", 0) for r in successful)
        lines.append(f"- 총 토큰 사용량: {total_tokens:,}")
        lines.append(f"- 총 감지 병변 수: {total_lesions}")
        lines.append("")

        # 진단 결과 통계
        lines.append("## 진단 결과 분포")
        lines.append("")

        diagnosis_counts = {}
        for r in successful:
            conditions = r.get("diagnosis_result", {}).get("possible_conditions", [])
            if conditions and isinstance(conditions, list):
                for cond in conditions[:1]:  # top-1만
                    if isinstance(cond, dict):
                        name = cond.get("name", "Unknown")
                        diagnosis_counts[name] = diagnosis_counts.get(name, 0) + 1

        for name, count in sorted(diagnosis_counts.items(), key=lambda x: -x[1])[:10]:
            lines.append(f"- {name}: {count}건")

        lines.append("")

    # 개별 결과
    lines.append("## 개별 분석 결과")
    lines.append("")

    for r in successful[:20]:  # 최대 20개
        filename = r.get("filename", "Unknown")
        lines.append(f"### {filename}")
        lines.append("")

        lesion_count = r.get("location_result", {}).get("lesion_count", 0)
        lines.append(f"**병변 수**: {lesion_count}")
        lines.append("")

        conditions = r.get("diagnosis_result", {}).get("possible_conditions", [])
        if conditions:
            lines.append("**진단 결과**:")
            for cond in conditions[:3]:
                if isinstance(cond, dict):
                    name = cond.get("name", "Unknown")
                    conf = cond.get("confidence", "N/A")
                    lines.append(f"- {name} ({conf})")
            lines.append("")

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def _print_summary(results: List[Dict]):
    """결과 요약 출력"""
    print("\n" + "="*60)
    print("파이프라인 완료!")
    print("="*60)

    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    print(f"총 처리: {len(results)}")
    print(f"성공: {len(successful)}")
    print(f"실패: {len(failed)}")

    if successful:
        total_tokens = sum(r.get("total_tokens", 0) for r in successful)
        avg_tokens = total_tokens / len(successful)
        print(f"총 토큰: {total_tokens:,} (평균: {avg_tokens:.0f})")

        total_lesions = sum(r.get("location_result", {}).get("lesion_count", 0) for r in successful)
        avg_lesions = total_lesions / len(successful)
        print(f"총 병변: {total_lesions} (평균: {avg_lesions:.1f})")


# 기본 데이터 디렉토리 (프로젝트 내 샘플 이미지)
DEFAULT_DATA_DIR = "Derm1M_v2_pretrain_ontology_sampled_100_images"
DEFAULT_OUTPUT_DIR = "outputs/llm_guided"


def find_data_directory() -> Optional[str]:
    """
    데이터 디렉토리 자동 탐지

    탐색 순서:
    1. 현재 디렉토리의 DEFAULT_DATA_DIR
    2. 스크립트 위치 기준 DEFAULT_DATA_DIR
    """
    # 현재 디렉토리에서 찾기
    if Path(DEFAULT_DATA_DIR).exists():
        return DEFAULT_DATA_DIR

    # 스크립트 위치 기준으로 찾기
    script_dir = Path(__file__).parent
    data_path = script_dir / DEFAULT_DATA_DIR
    if data_path.exists():
        return str(data_path)

    return None


def main():
    parser = argparse.ArgumentParser(
        description="LLM-Guided SAM Pipeline for Skin Disease Diagnosis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 기본 실행 (샘플 이미지 5개 처리)
  python llm_guided_pipeline.py

  # 전체 이미지 처리
  python llm_guided_pipeline.py --max-images 0

  # 특정 이미지 처리
  python llm_guided_pipeline.py --single-image path/to/image.png

  # MedSAM2 사용
  python llm_guided_pipeline.py --segmenter medsam2 --max-images 10
        """
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help=f"이미지 디렉토리 경로 (기본값: {DEFAULT_DATA_DIR})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="결과 저장 디렉토리"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=5,
        help="최대 처리 이미지 수 (기본값: 5, 0=전체)"
    )
    parser.add_argument(
        "--segmenter",
        type=str,
        choices=["sam", "sam2", "medsam2"],
        default="sam",
        help="사용할 분할 모델"
    )
    parser.add_argument(
        "--single-image",
        type=str,
        default=None,
        help="단일 이미지 처리 (배치 대신)"
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="CSV 저장 건너뛰기"
    )

    args = parser.parse_args()

    # max_images=0이면 전체 처리
    if args.max_images == 0:
        args.max_images = None

    # 단일 이미지 모드
    if args.single_image:
        from sam_masking import SAMSegmenter, SAM2Segmenter, MedSAM2Segmenter, load_image, save_image

        image_path = args.single_image
        if not Path(image_path).exists():
            print(f"파일을 찾을 수 없음: {image_path}")
            sys.exit(1)

        image = load_image(image_path)
        print(f"이미지 로드: {image_path}")
        print(f"크기: {image.shape}")

        # Segmenter 초기화
        checkpoint_dir = str(Path(args.output_dir) / "checkpoints")
        if args.segmenter == "sam":
            segmenter = SAMSegmenter(checkpoint_dir=checkpoint_dir)
        elif args.segmenter == "sam2":
            segmenter = SAM2Segmenter(checkpoint_dir=checkpoint_dir)
        else:
            segmenter = MedSAM2Segmenter(checkpoint_dir=checkpoint_dir)

        # 파이프라인 실행
        pipeline = LLMGuidedSegmenter()
        result = pipeline.run_full_pipeline(
            image=image,
            segmenter=segmenter,
            save_results=True,
            output_dir=args.output_dir,
            filename=Path(image_path).name
        )

        # 결과 출력
        print("\n=== 결과 요약 ===")
        print(f"감지된 병변 수: {result['location_result'].get('lesion_count', 0)}")

        if result['diagnosis_result'].get('possible_conditions'):
            print("\n진단 결과:")
            for cond in result['diagnosis_result']['possible_conditions'][:3]:
                if isinstance(cond, dict):
                    print(f"  - {cond.get('name', 'N/A')} ({cond.get('confidence', 'N/A')})")

    else:
        # 배치 모드
        # 데이터 디렉토리 자동 탐지
        data_dir = args.data_dir
        if data_dir is None:
            data_dir = find_data_directory()
            if data_dir is None:
                print(f"데이터 디렉토리를 찾을 수 없습니다.")
                print(f"다음 경로에 이미지 폴더가 있어야 합니다: {DEFAULT_DATA_DIR}")
                print(f"또는 --data-dir 옵션으로 경로를 지정하세요.")
                sys.exit(1)
            print(f"데이터 디렉토리 자동 탐지: {data_dir}")

        if not Path(data_dir).exists():
            print(f"디렉토리를 찾을 수 없음: {data_dir}")
            sys.exit(1)

        # 이미지 수 확인
        image_count = len(get_image_files(data_dir))
        print(f"\n발견된 이미지: {image_count}개")
        if args.max_images:
            print(f"처리할 이미지: {min(args.max_images, image_count)}개")
        else:
            print(f"처리할 이미지: 전체 ({image_count}개)")

        run_batch_pipeline(
            data_dir=data_dir,
            output_dir=args.output_dir,
            max_images=args.max_images,
            segmenter_type=args.segmenter,
            save_csv=not args.no_csv
        )


if __name__ == "__main__":
    main()
