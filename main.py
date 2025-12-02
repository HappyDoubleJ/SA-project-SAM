"""
Main Pipeline for Skin Disease Detection using SAM/SAM2/MedSAM2 + OpenAI

This script:
1. Loads skin disease images from the dataset
2. Applies multiple segmentation models and strategies:
   - SAM (center-focused + lesion-feature-based)
   - SAM2 (center-focused + lesion-feature-based)
   - MedSAM2 (optional, if installed)
3. Creates visualizations comparing all methods
4. Sends images to OpenAI for diagnosis comparison
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from PIL import Image
from tqdm import tqdm
import json

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from sam_masking import (
    SAMSegmenter,
    SAM2Segmenter,
    MedSAM2Segmenter,
    LesionFeatureDetector,
    apply_mask_to_image,
    crop_masked_region,
    create_masked_only_image,
    load_image,
    save_image
)
from visualize import (
    create_comparison_figure,
    create_detailed_comparison,
    compute_mask_metrics,
)
from openai_diagnosis import (
    SkinDiseaseDiagnoser,
    run_batch_diagnosis,
    generate_comparison_report
)


def get_image_files(data_dir: str, extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp')) -> List[Path]:
    """Get all image files from directory recursively"""
    data_path = Path(data_dir)
    image_files = []

    for ext in extensions:
        image_files.extend(data_path.rglob(f"*{ext}"))
        image_files.extend(data_path.rglob(f"*{ext.upper()}"))

    return sorted(image_files)


def create_full_comparison_figure(
    original: np.ndarray,
    results: Dict,
    title: str,
    save_path: str,
    show: bool = False
):
    """Create a comprehensive comparison figure showing all segmentation results"""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    # Count number of results
    n_results = len(results)
    n_cols = min(4, n_results + 1)  # +1 for original
    n_rows = (n_results + n_cols) // n_cols + 1

    fig = plt.figure(figsize=(5 * n_cols, 5 * n_rows))

    # Original image
    ax = fig.add_subplot(n_rows, n_cols, 1)
    ax.imshow(original)
    ax.set_title("Original", fontsize=10, fontweight='bold')
    ax.axis('off')

    # Each segmentation result
    for idx, (name, data) in enumerate(results.items(), start=2):
        ax = fig.add_subplot(n_rows, n_cols, idx)

        if 'overlay' in data:
            ax.imshow(data['overlay'])
        elif 'mask' in data:
            overlay = apply_mask_to_image(original, data['mask'])
            ax.imshow(overlay)
        else:
            ax.imshow(original)

        score = data.get('score', 0)
        method = data.get('method', 'unknown')
        ax.set_title(f"{name}\n({method}, score: {score:.2f})", fontsize=9)
        ax.axis('off')

    plt.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')

    if show:
        plt.show()
    else:
        plt.close()


def process_single_image(
    image_path: Path,
    sam_segmenter: Optional[SAMSegmenter],
    sam2_segmenter: Optional[SAM2Segmenter],
    medsam2_segmenter: Optional[MedSAM2Segmenter],
    output_dir: Path,
    save_visualizations: bool = True
) -> Dict:
    """
    Process a single image with all available segmenters and strategies

    Returns:
        Dictionary with all processed data
    """
    result = {
        'filename': image_path.name,
        'path': str(image_path),
        'segmentations': {}
    }

    # Load image
    try:
        image = load_image(str(image_path))
        result['original'] = image
    except Exception as e:
        result['error'] = f"Failed to load image: {e}"
        return result

    # SAM segmentation (both strategies)
    if sam_segmenter is not None:
        print("  [SAM] Running segmentation...")
        try:
            sam_results = sam_segmenter.segment_both_strategies(image)

            for strategy_name, seg_result in sam_results.items():
                key = f"SAM_{strategy_name}"
                seg_result['overlay'] = apply_mask_to_image(
                    image, seg_result['mask'], color=(255, 0, 0), alpha=0.4
                )
                seg_result['cropped'] = crop_masked_region(image, seg_result['mask'], padding=20)
                result['segmentations'][key] = seg_result

        except Exception as e:
            print(f"    SAM error: {e}")
            result['sam_error'] = str(e)

    # SAM2 segmentation (both strategies)
    if sam2_segmenter is not None:
        print("  [SAM2] Running segmentation...")
        try:
            sam2_results = sam2_segmenter.segment_both_strategies(image)

            for strategy_name, seg_result in sam2_results.items():
                key = f"SAM2_{strategy_name}"
                seg_result['overlay'] = apply_mask_to_image(
                    image, seg_result['mask'], color=(0, 255, 0), alpha=0.4
                )
                seg_result['cropped'] = crop_masked_region(image, seg_result['mask'], padding=20)
                result['segmentations'][key] = seg_result

        except Exception as e:
            print(f"    SAM2 error: {e}")
            result['sam2_error'] = str(e)

    # MedSAM2 segmentation (both strategies)
    if medsam2_segmenter is not None:
        print("  [MedSAM2] Running segmentation...")
        try:
            medsam2_results = medsam2_segmenter.segment_both_strategies(image)

            for strategy_name, seg_result in medsam2_results.items():
                key = f"MedSAM2_{strategy_name}"
                seg_result['overlay'] = apply_mask_to_image(
                    image, seg_result['mask'], color=(0, 0, 255), alpha=0.4
                )
                seg_result['cropped'] = crop_masked_region(image, seg_result['mask'], padding=20)
                result['segmentations'][key] = seg_result

        except Exception as e:
            print(f"    MedSAM2 error: {e}")
            result['medsam2_error'] = str(e)

    # Save visualizations
    if save_visualizations and result['segmentations']:
        stem = image_path.stem
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)

        # Save individual overlays
        for seg_name, seg_data in result['segmentations'].items():
            if 'overlay' in seg_data:
                save_image(seg_data['overlay'], str(vis_dir / f"{stem}_{seg_name}_overlay.png"))
            if seg_data.get('cropped') is not None:
                save_image(seg_data['cropped'], str(vis_dir / f"{stem}_{seg_name}_cropped.png"))

        # Save comprehensive comparison
        try:
            create_full_comparison_figure(
                original=result['original'],
                results=result['segmentations'],
                title=f"Segmentation Comparison: {image_path.name}",
                save_path=str(vis_dir / f"{stem}_full_comparison.png"),
                show=False
            )
        except Exception as e:
            print(f"    Visualization error: {e}")

    # Compute mask comparison metrics between strategies
    seg_keys = list(result['segmentations'].keys())
    result['mask_comparisons'] = {}

    for i, key1 in enumerate(seg_keys):
        for key2 in seg_keys[i+1:]:
            mask1 = result['segmentations'][key1].get('mask')
            mask2 = result['segmentations'][key2].get('mask')
            if mask1 is not None and mask2 is not None:
                metrics = compute_mask_metrics(mask1, mask2)
                result['mask_comparisons'][f"{key1}_vs_{key2}"] = metrics

    return result


def run_pipeline(
    data_dir: str,
    output_dir: str,
    max_images: Optional[int] = None,
    run_diagnosis: bool = True,
    use_sam: bool = True,
    use_sam2: bool = True,
    use_medsam2: bool = True,
    save_visualizations: bool = True
):
    """
    Run the complete pipeline

    Args:
        data_dir: Directory containing skin disease images
        output_dir: Directory to save outputs
        max_images: Maximum number of images to process (None for all)
        run_diagnosis: Whether to run OpenAI diagnosis
        use_sam: Whether to use SAM
        use_sam2: Whether to use SAM2
        use_medsam2: Whether to use MedSAM2
        save_visualizations: Whether to save visualization images
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get image files
    image_files = get_image_files(data_dir)
    if max_images:
        image_files = image_files[:max_images]

    print(f"Found {len(image_files)} images to process")

    # Initialize segmenters
    checkpoint_dir = str(output_path / "checkpoints")

    sam_segmenter = None
    sam2_segmenter = None
    medsam2_segmenter = None

    if use_sam:
        print("\nInitializing SAM segmenter...")
        try:
            sam_segmenter = SAMSegmenter(checkpoint_dir=checkpoint_dir)
        except Exception as e:
            print(f"SAM initialization failed: {e}")

    if use_sam2:
        print("Initializing SAM2 segmenter...")
        try:
            sam2_segmenter = SAM2Segmenter(checkpoint_dir=checkpoint_dir)
        except Exception as e:
            print(f"SAM2 initialization failed: {e}")

    if use_medsam2:
        print("Initializing MedSAM2 segmenter...")
        try:
            medsam2_segmenter = MedSAM2Segmenter(checkpoint_dir=checkpoint_dir)
            if not medsam2_segmenter.is_available():
                print("MedSAM2 not available (checkpoint not found)")
                print("To use MedSAM2, please install from: https://github.com/bowang-lab/MedSAM2")
                medsam2_segmenter = None
        except Exception as e:
            print(f"MedSAM2 initialization failed: {e}")
            medsam2_segmenter = None

    if sam_segmenter is None and sam2_segmenter is None and medsam2_segmenter is None:
        print("ERROR: No segmentation models available!")
        sys.exit(1)

    # Process images
    print("\n" + "="*60)
    print("Processing images...")
    print("="*60)

    all_results = []
    for idx, image_path in enumerate(tqdm(image_files, desc="Processing")):
        print(f"\n[{idx+1}/{len(image_files)}] {image_path.name}")

        result = process_single_image(
            image_path=image_path,
            sam_segmenter=sam_segmenter,
            sam2_segmenter=sam2_segmenter,
            medsam2_segmenter=medsam2_segmenter,
            output_dir=output_path,
            save_visualizations=save_visualizations
        )
        all_results.append(result)

    # Save results summary
    summary_path = output_path / "segmentation_results.json"
    summary_data = []
    for r in all_results:
        item = {
            'filename': r['filename'],
            'path': r['path'],
            'segmentations': {}
        }
        for seg_name, seg_data in r.get('segmentations', {}).items():
            item['segmentations'][seg_name] = {
                'score': seg_data.get('score'),
                'method': seg_data.get('method'),
            }
        item['mask_comparisons'] = r.get('mask_comparisons', {})
        if 'error' in r:
            item['error'] = r['error']
        summary_data.append(item)

    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2, default=str)
    print(f"\nSegmentation results saved to: {summary_path}")

    # Run OpenAI diagnosis
    if run_diagnosis:
        print("\n" + "="*60)
        print("Running OpenAI diagnosis comparison...")
        print("="*60)

        try:
            diagnoser = SkinDiseaseDiagnoser()

            # Use the best segmentation result for diagnosis
            diagnosis_data = []
            for r in all_results:
                if 'error' not in r and 'original' in r and r.get('segmentations'):
                    # Find best segmentation by score
                    best_seg = None
                    best_score = 0
                    for seg_name, seg_data in r['segmentations'].items():
                        if seg_data.get('score', 0) > best_score:
                            best_score = seg_data['score']
                            best_seg = seg_data

                    if best_seg:
                        diagnosis_data.append({
                            'filename': r['filename'],
                            'original': r['original'],
                            'masked_overlay': best_seg.get('overlay', r['original']),
                            'cropped': best_seg.get('cropped')
                        })

            if diagnosis_data:
                diagnosis_results = run_batch_diagnosis(
                    images_data=diagnosis_data,
                    output_dir=str(output_path / "diagnosis"),
                    diagnoser=diagnoser
                )

                # Generate report
                generate_comparison_report(
                    results=diagnosis_results,
                    output_path=str(output_path / "diagnosis_report.md")
                )
        except Exception as e:
            print(f"Diagnosis error: {e}")
            print("Skipping diagnosis step...")

    # Print summary
    print("\n" + "="*60)
    print("Pipeline Complete!")
    print("="*60)
    print(f"Processed: {len(all_results)} images")
    print(f"Output directory: {output_path}")

    successful = [r for r in all_results if 'error' not in r]
    print(f"Successful: {len(successful)}")

    # Print average scores by model/strategy
    if successful:
        print("\nAverage Scores by Model/Strategy:")
        score_sums = {}
        score_counts = {}

        for r in successful:
            for seg_name, seg_data in r.get('segmentations', {}).items():
                score = seg_data.get('score', 0)
                if seg_name not in score_sums:
                    score_sums[seg_name] = 0
                    score_counts[seg_name] = 0
                score_sums[seg_name] += score
                score_counts[seg_name] += 1

        for name in sorted(score_sums.keys()):
            avg = score_sums[name] / score_counts[name]
            print(f"  {name}: {avg:.3f}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Skin Disease Detection Pipeline")

    parser.add_argument(
        "--data-dir",
        type=str,
        default="Derm1M_v2_pretrain_ontology_sampled_100_images",
        help="Directory containing skin disease images"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory to save outputs"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to process"
    )
    parser.add_argument(
        "--no-diagnosis",
        action="store_true",
        help="Skip OpenAI diagnosis step"
    )
    parser.add_argument(
        "--no-sam",
        action="store_true",
        help="Skip SAM"
    )
    parser.add_argument(
        "--no-sam2",
        action="store_true",
        help="Skip SAM2"
    )
    parser.add_argument(
        "--no-medsam2",
        action="store_true",
        help="Skip MedSAM2"
    )
    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Skip saving visualization images"
    )

    args = parser.parse_args()

    # Validate data directory
    if not Path(args.data_dir).exists():
        print(f"Error: Data directory not found: {args.data_dir}")
        sys.exit(1)

    # Run pipeline
    run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_images=args.max_images,
        run_diagnosis=not args.no_diagnosis,
        use_sam=not args.no_sam,
        use_sam2=not args.no_sam2,
        use_medsam2=not args.no_medsam2,
        save_visualizations=not args.no_visualizations
    )


if __name__ == "__main__":
    main()
