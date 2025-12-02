"""
SAM and MedSAM2 Masking Pipeline for Skin Disease Detection
"""

import os
import numpy as np
import torch
import cv2
from PIL import Image
from typing import Optional, Tuple, List, Dict, Any
import urllib.request
from pathlib import Path


class SAMSegmenter:
    """Standard SAM (Segment Anything Model) for skin lesion segmentation"""

    CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    CHECKPOINT_NAME = "sam_vit_h_4b8939.pth"

    def __init__(self, checkpoint_dir: str = "checkpoints", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.model = None
        self.predictor = None
        self.mask_generator = None

    def download_checkpoint(self) -> str:
        """Download SAM checkpoint if not exists"""
        checkpoint_path = self.checkpoint_dir / self.CHECKPOINT_NAME

        if not checkpoint_path.exists():
            print(f"Downloading SAM checkpoint to {checkpoint_path}...")
            urllib.request.urlretrieve(self.CHECKPOINT_URL, checkpoint_path)
            print("Download complete!")

        return str(checkpoint_path)

    def load_model(self):
        """Load SAM model"""
        try:
            from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
        except ImportError:
            raise ImportError("Please install segment-anything: pip install git+https://github.com/facebookresearch/segment-anything.git")

        checkpoint_path = self.download_checkpoint()

        print(f"Loading SAM model on {self.device}...")
        self.model = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
        self.model.to(device=self.device)

        self.predictor = SamPredictor(self.model)
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.model,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,
        )
        print("SAM model loaded successfully!")

    def segment_auto(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Automatic segmentation - finds all possible masks

        Args:
            image: RGB image as numpy array (H, W, 3)

        Returns:
            List of mask dictionaries sorted by area (largest first)
        """
        if self.mask_generator is None:
            self.load_model()

        masks = self.mask_generator.generate(image)
        # Sort by area (largest first)
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        return masks

    def segment_with_point(self, image: np.ndarray,
                           point: Tuple[int, int] = None,
                           point_label: int = 1) -> Tuple[np.ndarray, float]:
        """
        Segment with a point prompt

        Args:
            image: RGB image as numpy array (H, W, 3)
            point: (x, y) point coordinate. If None, uses center of image
            point_label: 1 for foreground, 0 for background

        Returns:
            Tuple of (mask, confidence_score)
        """
        if self.predictor is None:
            self.load_model()

        self.predictor.set_image(image)

        if point is None:
            # Use center of image as default
            h, w = image.shape[:2]
            point = (w // 2, h // 2)

        input_point = np.array([[point[0], point[1]]])
        input_label = np.array([point_label])

        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        # Return the mask with highest score
        best_idx = np.argmax(scores)
        return masks[best_idx], scores[best_idx]

    def segment_with_box(self, image: np.ndarray,
                         box: Tuple[int, int, int, int] = None) -> Tuple[np.ndarray, float]:
        """
        Segment with a bounding box prompt

        Args:
            image: RGB image as numpy array (H, W, 3)
            box: (x1, y1, x2, y2) bounding box. If None, uses 10% margin from edges

        Returns:
            Tuple of (mask, confidence_score)
        """
        if self.predictor is None:
            self.load_model()

        self.predictor.set_image(image)

        if box is None:
            # Use 10% margin from edges as default box
            h, w = image.shape[:2]
            margin_x = int(w * 0.1)
            margin_y = int(h * 0.1)
            box = (margin_x, margin_y, w - margin_x, h - margin_y)

        input_box = np.array([box])

        masks, scores, logits = self.predictor.predict(
            box=input_box,
            multimask_output=True,
        )

        best_idx = np.argmax(scores)
        return masks[best_idx], scores[best_idx]

    def get_best_lesion_mask(self, image: np.ndarray,
                             use_center_bias: bool = True) -> Tuple[np.ndarray, float, str]:
        """
        Get the best mask for skin lesion using multiple strategies

        Args:
            image: RGB image as numpy array (H, W, 3)
            use_center_bias: Prefer masks near center (lesions usually centered)

        Returns:
            Tuple of (mask, confidence_score, method_used)
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        results = []

        # Try center point
        try:
            mask, score = self.segment_with_point(image, center)
            results.append((mask, score, "center_point"))
        except Exception as e:
            print(f"Center point failed: {e}")

        # Try bounding box
        try:
            mask, score = self.segment_with_box(image)
            results.append((mask, score, "bounding_box"))
        except Exception as e:
            print(f"Bounding box failed: {e}")

        # Try automatic and select best
        try:
            auto_masks = self.segment_auto(image)
            if auto_masks:
                # Filter masks that are too small or too large
                img_area = h * w
                valid_masks = [m for m in auto_masks
                              if 0.01 * img_area < m['area'] < 0.9 * img_area]

                if valid_masks and use_center_bias:
                    # Prefer masks whose centroid is near center
                    def center_distance(mask_dict):
                        mask = mask_dict['segmentation']
                        y_coords, x_coords = np.where(mask)
                        if len(x_coords) == 0:
                            return float('inf')
                        centroid = (np.mean(x_coords), np.mean(y_coords))
                        return np.sqrt((centroid[0] - center[0])**2 + (centroid[1] - center[1])**2)

                    valid_masks = sorted(valid_masks, key=center_distance)

                if valid_masks:
                    best_auto = valid_masks[0]
                    results.append((best_auto['segmentation'],
                                  best_auto['predicted_iou'],
                                  "auto_center_biased" if use_center_bias else "auto"))
        except Exception as e:
            print(f"Auto segmentation failed: {e}")

        if not results:
            # Return empty mask if all methods failed
            return np.zeros((h, w), dtype=bool), 0.0, "failed"

        # Return result with highest score
        best = max(results, key=lambda x: x[1])
        return best


class MedSAM2Segmenter:
    """MedSAM2 for medical image segmentation (specialized for medical images)"""

    def __init__(self, checkpoint_dir: str = "checkpoints", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.model = None
        self.predictor = None

    def load_model(self):
        """Load MedSAM2 model"""
        try:
            # Try to import MedSAM2
            # MedSAM2 uses SAM2 architecture
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            # Check for MedSAM2 checkpoint
            medsam2_checkpoint = self.checkpoint_dir / "medsam2_checkpoint.pth"

            if not medsam2_checkpoint.exists():
                print("MedSAM2 checkpoint not found.")
                print("Please download from: https://github.com/bowang-lab/MedSAM")
                print("Using SAM2 base model as fallback...")

                # Use SAM2 base model
                sam2_checkpoint = self.checkpoint_dir / "sam2_hiera_large.pt"
                sam2_config = "sam2_hiera_l.yaml"

                if not sam2_checkpoint.exists():
                    print(f"Downloading SAM2 checkpoint...")
                    url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
                    urllib.request.urlretrieve(url, sam2_checkpoint)

                self.model = build_sam2(sam2_config, str(sam2_checkpoint), device=self.device)
            else:
                # Load MedSAM2 specific model
                self.model = build_sam2("sam2_hiera_l.yaml", str(medsam2_checkpoint), device=self.device)

            self.predictor = SAM2ImagePredictor(self.model)
            print("MedSAM2/SAM2 model loaded successfully!")

        except ImportError:
            print("SAM2 not installed. Installing...")
            print("Please run: pip install git+https://github.com/facebookresearch/segment-anything-2.git")
            raise ImportError("SAM2 package required for MedSAM2")

    def segment_with_box(self, image: np.ndarray,
                         box: Tuple[int, int, int, int] = None) -> Tuple[np.ndarray, float]:
        """
        Segment with a bounding box prompt (recommended for MedSAM2)

        Args:
            image: RGB image as numpy array (H, W, 3)
            box: (x1, y1, x2, y2) bounding box

        Returns:
            Tuple of (mask, confidence_score)
        """
        if self.predictor is None:
            self.load_model()

        h, w = image.shape[:2]

        if box is None:
            margin_x = int(w * 0.1)
            margin_y = int(h * 0.1)
            box = (margin_x, margin_y, w - margin_x, h - margin_y)

        with torch.inference_mode():
            self.predictor.set_image(image)

            masks, scores, _ = self.predictor.predict(
                box=np.array(box),
                multimask_output=True,
            )

        best_idx = np.argmax(scores)
        return masks[best_idx], scores[best_idx]

    def segment_with_point(self, image: np.ndarray,
                           point: Tuple[int, int] = None) -> Tuple[np.ndarray, float]:
        """
        Segment with a point prompt

        Args:
            image: RGB image as numpy array (H, W, 3)
            point: (x, y) point coordinate

        Returns:
            Tuple of (mask, confidence_score)
        """
        if self.predictor is None:
            self.load_model()

        h, w = image.shape[:2]

        if point is None:
            point = (w // 2, h // 2)

        with torch.inference_mode():
            self.predictor.set_image(image)

            masks, scores, _ = self.predictor.predict(
                point_coords=np.array([[point[0], point[1]]]),
                point_labels=np.array([1]),
                multimask_output=True,
            )

        best_idx = np.argmax(scores)
        return masks[best_idx], scores[best_idx]


def apply_mask_to_image(image: np.ndarray, mask: np.ndarray,
                        color: Tuple[int, int, int] = (255, 0, 0),
                        alpha: float = 0.5) -> np.ndarray:
    """
    Apply colored mask overlay to image

    Args:
        image: RGB image as numpy array
        mask: Binary mask
        color: RGB color for mask overlay
        alpha: Transparency (0-1)

    Returns:
        Image with mask overlay
    """
    result = image.copy()
    mask_bool = mask.astype(bool)

    for c in range(3):
        result[:, :, c] = np.where(
            mask_bool,
            result[:, :, c] * (1 - alpha) + color[c] * alpha,
            result[:, :, c]
        )

    return result.astype(np.uint8)


def crop_masked_region(image: np.ndarray, mask: np.ndarray,
                       padding: int = 10) -> Optional[np.ndarray]:
    """
    Crop the masked region from image with padding

    Args:
        image: RGB image as numpy array
        mask: Binary mask
        padding: Pixels to add around the bounding box

    Returns:
        Cropped image region or None if mask is empty
    """
    if not mask.any():
        return None

    y_coords, x_coords = np.where(mask)

    x_min = max(0, x_coords.min() - padding)
    x_max = min(image.shape[1], x_coords.max() + padding)
    y_min = max(0, y_coords.min() - padding)
    y_max = min(image.shape[0], y_coords.max() + padding)

    return image[y_min:y_max, x_min:x_max]


def create_masked_only_image(image: np.ndarray, mask: np.ndarray,
                             background_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """
    Create image showing only the masked region, rest is background color

    Args:
        image: RGB image as numpy array
        mask: Binary mask
        background_color: RGB color for background

    Returns:
        Image with only masked region visible
    """
    result = np.full_like(image, background_color)
    mask_bool = mask.astype(bool)
    result[mask_bool] = image[mask_bool]
    return result


def load_image(image_path: str) -> np.ndarray:
    """Load image and convert to RGB numpy array"""
    image = Image.open(image_path).convert('RGB')
    return np.array(image)


def save_image(image: np.ndarray, path: str):
    """Save numpy array as image"""
    Image.fromarray(image).save(path)


if __name__ == "__main__":
    # Test with a sample image
    import sys

    if len(sys.argv) < 2:
        print("Usage: python sam_masking.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    image = load_image(image_path)

    print("Testing SAM segmentation...")
    sam = SAMSegmenter()
    mask, score, method = sam.get_best_lesion_mask(image)
    print(f"Best mask found using {method} with score {score:.3f}")

    # Apply and save
    result = apply_mask_to_image(image, mask)
    output_path = image_path.replace('.', '_sam_masked.')
    save_image(result, output_path)
    print(f"Saved to {output_path}")
