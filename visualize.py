"""
Visualization module for comparing SAM and MedSAM2 masking results
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import cv2


def create_comparison_figure(
    original: np.ndarray,
    sam_mask: np.ndarray,
    medsam_mask: np.ndarray,
    sam_overlay: np.ndarray,
    medsam_overlay: np.ndarray,
    sam_score: float,
    medsam_score: float,
    title: str = "",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create a comparison figure showing original and masked results

    Args:
        original: Original RGB image
        sam_mask: Binary mask from SAM
        medsam_mask: Binary mask from MedSAM2
        sam_overlay: Image with SAM mask overlay
        medsam_overlay: Image with MedSAM2 mask overlay
        sam_score: Confidence score from SAM
        medsam_score: Confidence score from MedSAM2
        title: Figure title
        save_path: Path to save the figure
        show: Whether to display the figure

    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.2)

    # Row 1: Original, SAM overlay, MedSAM overlay
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(original)
    ax1.set_title("Original Image", fontsize=12, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(sam_overlay)
    ax2.set_title(f"SAM Mask Overlay\n(score: {sam_score:.3f})", fontsize=12, fontweight='bold')
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(medsam_overlay)
    ax3.set_title(f"MedSAM2 Mask Overlay\n(score: {medsam_score:.3f})", fontsize=12, fontweight='bold')
    ax3.axis('off')

    # Row 2: Masks only
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(original)
    ax4.set_title("Original (Reference)", fontsize=12)
    ax4.axis('off')

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(sam_mask, cmap='gray')
    ax5.set_title("SAM Binary Mask", fontsize=12)
    ax5.axis('off')

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.imshow(medsam_mask, cmap='gray')
    ax6.set_title("MedSAM2 Binary Mask", fontsize=12)
    ax6.axis('off')

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved comparison figure to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def create_detailed_comparison(
    original: np.ndarray,
    sam_result: Dict,
    medsam_result: Dict,
    title: str = "",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create detailed comparison with cropped regions

    Args:
        original: Original RGB image
        sam_result: Dictionary with 'mask', 'score', 'overlay', 'cropped' keys
        medsam_result: Dictionary with same keys
        title: Figure title
        save_path: Path to save
        show: Whether to display

    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.2)

    # Row 1: Original and overlays
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.imshow(original)
    ax1.set_title("Original Image", fontsize=14, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(sam_result['overlay'])
    ax2.set_title(f"SAM Overlay\n(score: {sam_result['score']:.3f})", fontsize=12)
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 3])
    ax3.imshow(medsam_result['overlay'])
    ax3.set_title(f"MedSAM2 Overlay\n(score: {medsam_result['score']:.3f})", fontsize=12)
    ax3.axis('off')

    # Row 2: Binary masks and masked only
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(sam_result['mask'], cmap='Reds')
    ax4.set_title("SAM Mask", fontsize=12)
    ax4.axis('off')

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(medsam_result['mask'], cmap='Blues')
    ax5.set_title("MedSAM2 Mask", fontsize=12)
    ax5.axis('off')

    ax6 = fig.add_subplot(gs[1, 2])
    if sam_result.get('masked_only') is not None:
        ax6.imshow(sam_result['masked_only'])
    ax6.set_title("SAM Masked Region", fontsize=12)
    ax6.axis('off')

    ax7 = fig.add_subplot(gs[1, 3])
    if medsam_result.get('masked_only') is not None:
        ax7.imshow(medsam_result['masked_only'])
    ax7.set_title("MedSAM2 Masked Region", fontsize=12)
    ax7.axis('off')

    # Row 3: Cropped regions
    ax8 = fig.add_subplot(gs[2, 0:2])
    if sam_result.get('cropped') is not None:
        ax8.imshow(sam_result['cropped'])
        ax8.set_title("SAM Cropped Lesion", fontsize=12, fontweight='bold')
    else:
        ax8.text(0.5, 0.5, "No valid crop", ha='center', va='center', fontsize=12)
        ax8.set_title("SAM Cropped Lesion", fontsize=12)
    ax8.axis('off')

    ax9 = fig.add_subplot(gs[2, 2:4])
    if medsam_result.get('cropped') is not None:
        ax9.imshow(medsam_result['cropped'])
        ax9.set_title("MedSAM2 Cropped Lesion", fontsize=12, fontweight='bold')
    else:
        ax9.text(0.5, 0.5, "No valid crop", ha='center', va='center', fontsize=12)
        ax9.set_title("MedSAM2 Cropped Lesion", fontsize=12)
    ax9.axis('off')

    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.99)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved detailed comparison to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def create_batch_comparison_grid(
    results: List[Dict],
    output_path: str,
    cols: int = 3,
    show: bool = False
):
    """
    Create a grid showing multiple image comparisons

    Args:
        results: List of dicts with 'filename', 'original', 'sam_overlay', 'medsam_overlay'
        output_path: Path to save the grid
        cols: Number of columns
        show: Whether to display
    """
    n_images = len(results)
    rows = (n_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols * 3, figsize=(cols * 12, rows * 4))

    for idx, result in enumerate(results):
        row = idx // cols
        col_base = (idx % cols) * 3

        if rows == 1:
            ax_orig = axes[col_base]
            ax_sam = axes[col_base + 1]
            ax_medsam = axes[col_base + 2]
        else:
            ax_orig = axes[row, col_base]
            ax_sam = axes[row, col_base + 1]
            ax_medsam = axes[row, col_base + 2]

        ax_orig.imshow(result['original'])
        ax_orig.set_title(f"{result['filename']}\nOriginal", fontsize=8)
        ax_orig.axis('off')

        ax_sam.imshow(result['sam_overlay'])
        ax_sam.set_title("SAM", fontsize=8)
        ax_sam.axis('off')

        ax_medsam.imshow(result['medsam_overlay'])
        ax_medsam.set_title("MedSAM2", fontsize=8)
        ax_medsam.axis('off')

    # Hide empty subplots
    for idx in range(n_images, rows * cols):
        row = idx // cols
        col_base = (idx % cols) * 3
        for offset in range(3):
            if rows == 1:
                axes[col_base + offset].axis('off')
            else:
                axes[row, col_base + offset].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved batch comparison grid to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def compute_mask_metrics(mask1: np.ndarray, mask2: np.ndarray) -> Dict[str, float]:
    """
    Compute comparison metrics between two masks

    Args:
        mask1: First binary mask
        mask2: Second binary mask

    Returns:
        Dictionary with IoU, Dice, and other metrics
    """
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    iou = intersection / union if union > 0 else 0
    dice = 2 * intersection / (mask1.sum() + mask2.sum()) if (mask1.sum() + mask2.sum()) > 0 else 0

    return {
        'iou': iou,
        'dice': dice,
        'mask1_area': mask1.sum(),
        'mask2_area': mask2.sum(),
        'intersection': intersection,
        'union': union
    }


def visualize_mask_difference(
    mask1: np.ndarray,
    mask2: np.ndarray,
    labels: Tuple[str, str] = ("SAM", "MedSAM2"),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Visualize the difference between two masks

    Args:
        mask1: First binary mask
        mask2: Second binary mask
        labels: Labels for the two masks
        save_path: Path to save
        show: Whether to display

    Returns:
        matplotlib Figure
    """
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    # Create difference visualization
    # Green: mask1 only, Blue: mask2 only, Yellow: both
    h, w = mask1.shape
    diff_image = np.zeros((h, w, 3), dtype=np.uint8)

    only_mask1 = mask1 & ~mask2
    only_mask2 = mask2 & ~mask1
    both = mask1 & mask2

    diff_image[only_mask1] = [0, 255, 0]    # Green
    diff_image[only_mask2] = [0, 0, 255]    # Blue
    diff_image[both] = [255, 255, 0]        # Yellow

    metrics = compute_mask_metrics(mask1, mask2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(mask1, cmap='Greens')
    axes[0].set_title(f"{labels[0]}\n(Area: {metrics['mask1_area']})", fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(mask2, cmap='Blues')
    axes[1].set_title(f"{labels[1]}\n(Area: {metrics['mask2_area']})", fontsize=12)
    axes[1].axis('off')

    axes[2].imshow(diff_image)
    axes[2].set_title(f"Difference\nIoU: {metrics['iou']:.3f}, Dice: {metrics['dice']:.3f}", fontsize=12)
    axes[2].axis('off')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label=f'{labels[0]} only'),
        Patch(facecolor='blue', label=f'{labels[1]} only'),
        Patch(facecolor='yellow', label='Both'),
    ]
    axes[2].legend(handles=legend_elements, loc='upper right', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved mask difference to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


if __name__ == "__main__":
    # Demo with synthetic data
    print("Visualization module loaded. Use functions to compare masks.")

    # Create dummy data for testing
    h, w = 256, 256

    # Dummy masks
    mask1 = np.zeros((h, w), dtype=bool)
    mask1[80:180, 80:180] = True

    mask2 = np.zeros((h, w), dtype=bool)
    mask2[90:190, 70:170] = True

    # Visualize difference
    visualize_mask_difference(mask1, mask2, save_path=None, show=True)
