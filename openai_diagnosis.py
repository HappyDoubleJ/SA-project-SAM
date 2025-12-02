"""
OpenAI Integration for Skin Disease Diagnosis
Compares diagnosis performance with original image vs original + masked image
"""

import os
import base64
import json
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from io import BytesIO
from PIL import Image
import numpy as np

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()


class SkinDiseaseDiagnoser:
    """OpenAI-based skin disease diagnosis using GPT-4 Vision"""

    DIAGNOSIS_PROMPT = """You are an expert dermatologist AI assistant. Analyze the provided skin image(s) and provide a detailed diagnosis.

For each image analysis, provide:
1. **Observed Features**: Describe the visible skin lesion characteristics (color, shape, texture, borders, size estimation)
2. **Possible Conditions**: List the top 3-5 most likely skin conditions based on the visual features
3. **Confidence Level**: Rate your confidence (Low/Medium/High) for each suggested condition
4. **Key Differentiating Factors**: What features led to your diagnosis
5. **Recommended Actions**: Suggest next steps (e.g., biopsy, dermatologist visit, monitoring)

IMPORTANT: This is for educational/research purposes only. Always recommend consulting a healthcare professional for actual medical diagnosis.

Please structure your response as JSON with the following format:
{
    "observed_features": {
        "color": "description",
        "shape": "description",
        "texture": "description",
        "borders": "description",
        "size": "description",
        "other": "any other notable features"
    },
    "possible_conditions": [
        {"name": "condition name", "confidence": "High/Medium/Low", "reasoning": "why this condition"}
    ],
    "key_factors": ["factor1", "factor2"],
    "recommended_actions": ["action1", "action2"],
    "overall_assessment": "brief summary"
}"""

    COMPARISON_PROMPT = """You are an expert dermatologist AI assistant. You are provided with two views of the same skin lesion:
1. The original full image
2. A highlighted/segmented view showing the lesion area of interest

Analyze both images together and provide a comprehensive diagnosis. The segmented image helps you focus on the specific area of concern.

For your analysis, provide:
1. **Observed Features**: Describe the visible skin lesion characteristics (color, shape, texture, borders, size estimation)
2. **Possible Conditions**: List the top 3-5 most likely skin conditions based on the visual features
3. **Confidence Level**: Rate your confidence (Low/Medium/High) for each suggested condition
4. **Segmentation Benefits**: Describe how the highlighted region helped in your analysis
5. **Key Differentiating Factors**: What features led to your diagnosis
6. **Recommended Actions**: Suggest next steps

IMPORTANT: This is for educational/research purposes only.

Please structure your response as JSON:
{
    "observed_features": {
        "color": "description",
        "shape": "description",
        "texture": "description",
        "borders": "description",
        "size": "description",
        "other": "any other notable features"
    },
    "possible_conditions": [
        {"name": "condition name", "confidence": "High/Medium/Low", "reasoning": "why this condition"}
    ],
    "segmentation_benefits": "how the mask helped",
    "key_factors": ["factor1", "factor2"],
    "recommended_actions": ["action1", "action2"],
    "overall_assessment": "brief summary"
}"""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize OpenAI client"""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY in .env file")

        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-4o"  # Using GPT-4o for vision capabilities

    def _encode_image(self, image: np.ndarray) -> str:
        """Convert numpy array to base64 string"""
        pil_image = Image.fromarray(image)
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _encode_image_file(self, image_path: str) -> str:
        """Encode image file to base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def diagnose_original_only(self, image: np.ndarray) -> Dict:
        """
        Diagnose using only the original image

        Args:
            image: RGB image as numpy array

        Returns:
            Diagnosis result dictionary
        """
        base64_image = self._encode_image(image)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.DIAGNOSIS_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=2000,
            temperature=0.3
        )

        result_text = response.choices[0].message.content

        # Try to parse JSON from response
        try:
            # Find JSON in response
            json_start = result_text.find('{')
            json_end = result_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                result = json.loads(result_text[json_start:json_end])
            else:
                result = {"raw_response": result_text}
        except json.JSONDecodeError:
            result = {"raw_response": result_text}

        result["method"] = "original_only"
        result["tokens_used"] = response.usage.total_tokens

        return result

    def diagnose_with_mask(self, original: np.ndarray,
                           masked_overlay: np.ndarray,
                           cropped: Optional[np.ndarray] = None,
                           use_cropped: bool = False) -> Dict:
        """
        Diagnose using original + masked/cropped images

        Args:
            original: Original RGB image
            masked_overlay: Image with mask overlay
            cropped: Cropped lesion region (optional)
            use_cropped: If True, use cropped instead of overlay

        Returns:
            Diagnosis result dictionary
        """
        base64_original = self._encode_image(original)

        if use_cropped and cropped is not None:
            base64_masked = self._encode_image(cropped)
            second_image_desc = "cropped lesion region"
        else:
            base64_masked = self._encode_image(masked_overlay)
            second_image_desc = "highlighted lesion area"

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.COMPARISON_PROMPT},
                        {
                            "type": "text",
                            "text": "Image 1 - Original full image:"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_original}",
                                "detail": "high"
                            }
                        },
                        {
                            "type": "text",
                            "text": f"Image 2 - {second_image_desc}:"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_masked}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=2000,
            temperature=0.3
        )

        result_text = response.choices[0].message.content

        try:
            json_start = result_text.find('{')
            json_end = result_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                result = json.loads(result_text[json_start:json_end])
            else:
                result = {"raw_response": result_text}
        except json.JSONDecodeError:
            result = {"raw_response": result_text}

        result["method"] = "with_cropped" if use_cropped else "with_mask_overlay"
        result["tokens_used"] = response.usage.total_tokens

        return result

    def compare_diagnosis_methods(self,
                                   original: np.ndarray,
                                   masked_overlay: np.ndarray,
                                   cropped: Optional[np.ndarray] = None) -> Dict:
        """
        Compare all diagnosis methods and return results

        Args:
            original: Original RGB image
            masked_overlay: Image with mask overlay
            cropped: Cropped lesion region

        Returns:
            Dictionary with all diagnosis results and comparison
        """
        results = {}

        print("  Diagnosing with original image only...")
        results["original_only"] = self.diagnose_original_only(original)

        print("  Diagnosing with original + mask overlay...")
        results["with_overlay"] = self.diagnose_with_mask(
            original, masked_overlay, cropped, use_cropped=False
        )

        if cropped is not None:
            print("  Diagnosing with original + cropped region...")
            results["with_cropped"] = self.diagnose_with_mask(
                original, masked_overlay, cropped, use_cropped=True
            )

        # Compute comparison metrics
        results["comparison"] = self._compare_results(results)

        return results

    def _compare_results(self, results: Dict) -> Dict:
        """Compare diagnosis results across methods"""
        comparison = {
            "total_tokens": sum(r.get("tokens_used", 0) for r in results.values() if isinstance(r, dict)),
            "methods_used": list(results.keys()),
        }

        # Compare top conditions
        conditions_by_method = {}
        for method, result in results.items():
            if isinstance(result, dict) and "possible_conditions" in result:
                conditions = result["possible_conditions"]
                if conditions:
                    conditions_by_method[method] = [c.get("name", "Unknown") for c in conditions[:3]]

        comparison["top_conditions_by_method"] = conditions_by_method

        # Find consensus conditions
        all_conditions = []
        for conditions in conditions_by_method.values():
            all_conditions.extend(conditions)

        if all_conditions:
            from collections import Counter
            condition_counts = Counter(all_conditions)
            comparison["consensus_conditions"] = [
                {"condition": cond, "count": count}
                for cond, count in condition_counts.most_common(5)
            ]

        return comparison


def run_batch_diagnosis(
    images_data: List[Dict],
    output_dir: str,
    diagnoser: Optional[SkinDiseaseDiagnoser] = None
) -> List[Dict]:
    """
    Run diagnosis on multiple images

    Args:
        images_data: List of dicts with 'filename', 'original', 'masked_overlay', 'cropped'
        output_dir: Directory to save results
        diagnoser: SkinDiseaseDiagnoser instance

    Returns:
        List of diagnosis results
    """
    if diagnoser is None:
        diagnoser = SkinDiseaseDiagnoser()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = []

    for idx, data in enumerate(images_data):
        print(f"\nProcessing {idx + 1}/{len(images_data)}: {data['filename']}")

        try:
            result = diagnoser.compare_diagnosis_methods(
                original=data['original'],
                masked_overlay=data['masked_overlay'],
                cropped=data.get('cropped')
            )
            result['filename'] = data['filename']
            all_results.append(result)

            # Save individual result
            result_file = output_path / f"{Path(data['filename']).stem}_diagnosis.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)

        except Exception as e:
            print(f"  Error: {e}")
            all_results.append({
                'filename': data['filename'],
                'error': str(e)
            })

    # Save all results
    all_results_file = output_path / "all_diagnosis_results.json"
    with open(all_results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nAll results saved to: {all_results_file}")

    return all_results


def generate_comparison_report(results: List[Dict], output_path: str):
    """
    Generate a summary report comparing diagnosis methods

    Args:
        results: List of diagnosis results
        output_path: Path to save the report
    """
    report_lines = [
        "# Skin Disease Diagnosis Comparison Report",
        "",
        "## Summary",
        f"Total images analyzed: {len(results)}",
        "",
        "## Method Comparison",
        ""
    ]

    # Aggregate statistics
    successful = [r for r in results if 'error' not in r]
    failed = [r for r in results if 'error' in r]

    report_lines.append(f"- Successful analyses: {len(successful)}")
    report_lines.append(f"- Failed analyses: {len(failed)}")
    report_lines.append("")

    if successful:
        # Token usage
        total_tokens = sum(
            r.get('comparison', {}).get('total_tokens', 0)
            for r in successful
        )
        report_lines.append(f"Total tokens used: {total_tokens}")
        report_lines.append("")

        # Condition frequency
        report_lines.append("## Most Common Conditions Identified")
        all_conditions = []
        for r in successful:
            for method_key in ['original_only', 'with_overlay', 'with_cropped']:
                if method_key in r and 'possible_conditions' in r[method_key]:
                    for cond in r[method_key]['possible_conditions']:
                        if 'name' in cond:
                            all_conditions.append(cond['name'])

        if all_conditions:
            from collections import Counter
            condition_counts = Counter(all_conditions)
            for cond, count in condition_counts.most_common(10):
                report_lines.append(f"- {cond}: {count} occurrences")

    report_lines.append("")
    report_lines.append("## Individual Results")
    report_lines.append("")

    for r in successful:
        report_lines.append(f"### {r.get('filename', 'Unknown')}")
        report_lines.append("")

        for method in ['original_only', 'with_overlay', 'with_cropped']:
            if method in r:
                report_lines.append(f"**{method}:**")
                if 'possible_conditions' in r[method]:
                    for cond in r[method]['possible_conditions'][:3]:
                        name = cond.get('name', 'Unknown')
                        conf = cond.get('confidence', 'N/A')
                        report_lines.append(f"  - {name} (Confidence: {conf})")
                report_lines.append("")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    # Test with environment
    load_dotenv()

    if os.getenv("OPENAI_API_KEY"):
        print("OpenAI API key found!")
        diagnoser = SkinDiseaseDiagnoser()
        print(f"Using model: {diagnoser.model}")
    else:
        print("Warning: OPENAI_API_KEY not found in environment")
