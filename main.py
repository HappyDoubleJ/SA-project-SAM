"""
Main Pipeline for Skin Disease Detection using SAM/MedSAM2 + OpenAI

This script:
1. Loads skin disease images from the dataset
2. Applies SAM and MedSAM2 segmentation to identify lesion areas
3. Creates visualizations comparing the two methods
4. Sends images to OpenAI for diagnosis comparison (original vs masked)
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
from PIL import Image
from tqdm import tqdm

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from sam_masking import (
    SAMSegmenter,
    MedSAM2Segmenter,
    apply_mask_to_image,
    crop_masked_region,
    create_masked_only_image,
    load_image,
    save_image
)
from visualize import (
    create_comparison_figure,
    create_detailed_comparison,
    create_batch_comparison_grid,
    compute_mask_metrics,
    visualize_mask_difference
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


def process_single_image(
    image_path: Path,
    sam_segmenter: SAMSegmenter,
    medsam_segmenter: Optional[MedSAM2Segmenter],
    output_dir: Path,
    save_visualizations: bool = True
) -> Dict:
    """
    Process a single image with both segmenters

    Returns:
        Dictionary with all processed data
    """
    result = {
        'filename': image_path.name,
        'path': str(image_path),
    }

    # Load image
    try:
        image = load_image(str(image_path))
        result['original'] = image
    except Exception as e:
        result['error'] = f"Failed to load image: {e}"
        return result

    # SAM segmentation
    print(f"  Running SAM segmentation...")
    try:
        sam_mask, sam_score, sam_method = sam_segmenter.get_best_lesion_mask(image)
        result['sam_mask'] = sam_mask
        result['sam_score'] = sam_score
        result['sam_method'] = sam_method

        # Create overlays and crops
        result['sam_overlay'] = apply_mask_to_image(image, sam_mask, color=(255, 0, 0), alpha=0.4)
        result['sam_cropped'] = crop_masked_region(image, sam_mask, padding=20)
        result['sam_masked_only'] = create_masked_only_image(image, sam_mask)
    except Exception as e:
        print(f"  SAM error: {e}")
        result['sam_error'] = str(e)
        # Create empty mask as fallback
        result['sam_mask'] = np.zeros(image.shape[:2], dtype=bool)
        result['sam_score'] = 0.0
        result['sam_overlay'] = image.copy()

    # MedSAM2 segmentation
    if medsam_segmenter is not None:
        print(f"  Running MedSAM2 segmentation...")
        try:
            medsam_mask, medsam_score = medsam_segmenter.segment_with_point(image)
            result['medsam_mask'] = medsam_mask
            result['medsam_score'] = medsam_score

            result['medsam_overlay'] = apply_mask_to_image(image, medsam_mask, color=(0, 0, 255), alpha=0.4)
            result['medsam_cropped'] = crop_masked_region(image, medsam_mask, padding=20)
            result['medsam_masked_only'] = create_masked_only_image(image, medsam_mask)
        except Exception as e:
            print(f"  MedSAM2 error: {e}")
            result['medsam_error'] = str(e)
            result['medsam_mask'] = np.zeros(image.shape[:2], dtype=bool)
            result['medsam_score'] = 0.0
            result['medsam_overlay'] = image.copy()
    else:
        # Use SAM result as fallback for MedSAM2
        result['medsam_mask'] = result.get('sam_mask', np.zeros(image.shape[:2], dtype=bool))
        result['medsam_score'] = result.get('sam_score', 0.0)
        result['medsam_overlay'] = result.get('sam_overlay', image.copy())
        result['medsam_cropped'] = result.get('sam_cropped')
        result['medsam_masked_only'] = result.get('sam_masked_only')

    # Compute mask comparison metrics
    if 'sam_mask' in result and 'medsam_mask' in result:
        result['mask_metrics'] = compute_mask_metrics(result['sam_mask'], result['medsam_mask'])

    # Save visualizations
    if save_visualizations:
        stem = image_path.stem
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)

        # Save individual images
        if 'sam_overlay' in result:
            save_image(result['sam_overlay'], str(vis_dir / f"{stem}_sam_overlay.png"))
        if 'medsam_overlay' in result:
            save_image(result['medsam_overlay'], str(vis_dir / f"{stem}_medsam_overlay.png"))
        if result.get('sam_cropped') is not None:
            save_image(result['sam_cropped'], str(vis_dir / f"{stem}_sam_cropped.png"))

        # Save comparison figure
        try:
            create_comparison_figure(
                original=result['original'],
                sam_mask=result['sam_mask'],
                medsam_mask=result['medsam_mask'],
                sam_overlay=result['sam_overlay'],
                medsam_overlay=result['medsam_overlay'],
                sam_score=result['sam_score'],
                medsam_score=result['medsam_score'],
                title=f"Comparison: {image_path.name}",
                save_path=str(vis_dir / f"{stem}_comparison.png"),
                show=False
            )
        except Exception as e:
            print(f"  Visualization error: {e}")

    return result


def run_pipeline(
    data_dir: str,
    output_dir: str,
    max_images: Optional[int] = None,
    run_diagnosis: bool = True,
    use_medsam: bool = True,
    save_visualizations: bool = True
):
    """
    Run the complete pipeline

    Args:
        data_dir: Directory containing skin disease images
        output_dir: Directory to save outputs
        max_images: Maximum number of images to process (None for all)
        run_diagnosis: Whether to run OpenAI diagnosis
        use_medsam: Whether to use MedSAM2 (requires additional setup)
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
    print("\nInitializing SAM segmenter...")
    sam_segmenter = SAMSegmenter(checkpoint_dir=str(output_path / "checkpoints"))

    medsam_segmenter = None
    if use_medsam:
        print("Initializing MedSAM2 segmenter...")
        try:
            medsam_segmenter = MedSAM2Segmenter(checkpoint_dir=str(output_path / "checkpoints"))
        except Exception as e:
            print(f"MedSAM2 initialization failed: {e}")
            print("Continuing with SAM only...")

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
            medsam_segmenter=medsam_segmenter,
            output_dir=output_path,
            save_visualizations=save_visualizations
        )
        all_results.append(result)

    # Create batch comparison grid
    if save_visualizations and len(all_results) > 0:
        print("\nCreating batch comparison grid...")
        grid_data = [
            {
                'filename': r['filename'],
                'original': r['original'],
                'sam_overlay': r['sam_overlay'],
                'medsam_overlay': r['medsam_overlay']
            }
            for r in all_results if 'original' in r
        ]
        if grid_data:
            create_batch_comparison_grid(
                results=grid_data[:12],  # Limit to 12 for readability
                output_path=str(output_path / "batch_comparison.png"),
                cols=3,
                show=False
            )

    # Run OpenAI diagnosis
    if run_diagnosis:
        print("\n" + "="*60)
        print("Running OpenAI diagnosis comparison...")
        print("="*60)

        try:
            diagnoser = SkinDiseaseDiagnoser()

            diagnosis_data = []
            for r in all_results:
                if 'error' not in r and 'original' in r:
                    diagnosis_data.append({
                        'filename': r['filename'],
                        'original': r['original'],
                        'masked_overlay': r['sam_overlay'],
                        'cropped': r.get('sam_cropped')
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

    if successful:
        avg_sam_score = np.mean([r['sam_score'] for r in successful if 'sam_score' in r])
        avg_medsam_score = np.mean([r['medsam_score'] for r in successful if 'medsam_score' in r])
        print(f"Average SAM confidence: {avg_sam_score:.3f}")
        print(f"Average MedSAM2 confidence: {avg_medsam_score:.3f}")

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
        "--no-medsam",
        action="store_true",
        help="Skip MedSAM2 (use SAM only)"
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
        use_medsam=not args.no_medsam,
        save_visualizations=not args.no_visualizations
    )


if __name__ == "__main__":
    main()
