"""
SAM and MedSAM2 Masking Pipeline for Skin Disease Detection

Two segmentation strategies:
1. Center-focused: Assumes lesion is near image center
2. Lesion-feature-based: Detects based on color changes and texture/elevation
"""

import os
import numpy as np
import torch
import cv2
from PIL import Image
from typing import Optional, Tuple, List, Dict, Any
import urllib.request
from pathlib import Path
from scipy import ndimage
from skimage import color, filters, morphology


# =============================================================================
# Lesion Feature Detection (Color + Texture/Elevation)
# =============================================================================

class LesionFeatureDetector:
    """Detect potential lesion areas based on skin disease characteristics"""

    def __init__(self):
        pass

    def detect_color_anomalies(self, image: np.ndarray) -> np.ndarray:
        """
        Detect areas with abnormal color compared to surrounding skin

        Skin lesions often show:
        - Redness (erythema)
        - Brown/black pigmentation
        - White depigmentation
        - Yellow/purple discoloration
        """
        # Convert to LAB color space (better for skin color analysis)
        lab = color.rgb2lab(image)
        l_channel = lab[:, :, 0]  # Lightness
        a_channel = lab[:, :, 1]  # Green-Red
        b_channel = lab[:, :, 2]  # Blue-Yellow

        # Also use HSV for saturation analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h_channel = hsv[:, :, 0]
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]

        # Detect color deviations from local mean (potential lesion areas)
        kernel_size = max(image.shape[0], image.shape[1]) // 8
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel_size = max(kernel_size, 15)

        # Local mean subtraction to find color anomalies
        l_local_mean = cv2.GaussianBlur(l_channel, (kernel_size, kernel_size), 0)
        a_local_mean = cv2.GaussianBlur(a_channel, (kernel_size, kernel_size), 0)
        b_local_mean = cv2.GaussianBlur(b_channel, (kernel_size, kernel_size), 0)

        # Color deviation map
        l_diff = np.abs(l_channel - l_local_mean)
        a_diff = np.abs(a_channel - a_local_mean)
        b_diff = np.abs(b_channel - b_local_mean)

        # Combined color anomaly score
        color_anomaly = (l_diff / 50.0 + a_diff / 30.0 + b_diff / 30.0) / 3.0

        # Detect high saturation areas (often indicates lesions)
        s_normalized = s_channel.astype(float) / 255.0
        high_saturation = s_normalized > 0.3

        # Detect redness (high a* in LAB)
        redness = (a_channel > 10).astype(float)

        # Combine all features
        combined = color_anomaly + 0.3 * high_saturation.astype(float) + 0.2 * redness
        combined = np.clip(combined, 0, 1)

        # Normalize to 0-1
        if combined.max() > combined.min():
            combined = (combined - combined.min()) / (combined.max() - combined.min())

        return combined

    def detect_texture_elevation(self, image: np.ndarray) -> np.ndarray:
        """
        Detect areas with texture changes indicating elevation/depression

        Uses gradient and edge information to detect:
        - Raised lesions (papules, nodules)
        - Depressed lesions (atrophy, ulcers)
        - Textural changes (scaling, roughness)
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(float)
        else:
            gray = image.astype(float)

        # Compute gradient magnitude (indicates edges/elevation changes)
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)

        # Laplacian for detecting elevation changes
        laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_64F))

        # Local texture variance (indicates rough/scaly areas)
        kernel_size = 11
        local_mean = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        local_sq_mean = cv2.GaussianBlur(gray**2, (kernel_size, kernel_size), 0)
        local_variance = local_sq_mean - local_mean**2
        local_variance = np.clip(local_variance, 0, None)

        # Normalize each feature
        def normalize(x):
            if x.max() > x.min():
                return (x - x.min()) / (x.max() - x.min())
            return x

        gradient_norm = normalize(gradient_mag)
        laplacian_norm = normalize(laplacian)
        variance_norm = normalize(local_variance)

        # Combined texture/elevation score
        texture_score = 0.4 * gradient_norm + 0.3 * laplacian_norm + 0.3 * variance_norm

        return texture_score

    def get_lesion_candidate_points(self, image: np.ndarray,
                                     n_points: int = 5) -> List[Tuple[int, int]]:
        """
        Get candidate points for lesion location based on features

        Returns list of (x, y) points sorted by likelihood
        """
        color_map = self.detect_color_anomalies(image)
        texture_map = self.detect_texture_elevation(image)

        # Combined score (color is usually more reliable)
        combined = 0.6 * color_map + 0.4 * texture_map

        # Apply center bias (lesions often centered in clinical photos)
        h, w = combined.shape
        y_grid, x_grid = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        distance_from_center = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        center_weight = 1.0 - 0.3 * (distance_from_center / max_dist)

        combined_weighted = combined * center_weight

        # Find local maxima
        # Smooth first to avoid too many peaks
        smoothed = cv2.GaussianBlur(combined_weighted, (21, 21), 0)

        # Find top N points
        points = []
        temp = smoothed.copy()
        min_distance = min(h, w) // 10

        for _ in range(n_points):
            max_idx = np.unravel_index(np.argmax(temp), temp.shape)
            y, x = max_idx
            points.append((int(x), int(y)))

            # Suppress nearby area
            y_min = max(0, y - min_distance)
            y_max = min(h, y + min_distance)
            x_min = max(0, x - min_distance)
            x_max = min(w, x + min_distance)
            temp[y_min:y_max, x_min:x_max] = 0

        return points

    def get_lesion_bounding_box(self, image: np.ndarray,
                                 threshold: float = 0.3) -> Tuple[int, int, int, int]:
        """
        Get bounding box around detected lesion area

        Returns (x1, y1, x2, y2)
        """
        color_map = self.detect_color_anomalies(image)
        texture_map = self.detect_texture_elevation(image)
        combined = 0.6 * color_map + 0.4 * texture_map

        # Threshold to get binary mask
        binary = combined > threshold

        # Clean up with morphology
        binary = morphology.binary_opening(binary, morphology.disk(3))
        binary = morphology.binary_closing(binary, morphology.disk(5))

        # Find bounding box
        if not binary.any():
            # Fallback to center region
            h, w = image.shape[:2]
            margin = 0.1
            return (int(w * margin), int(h * margin),
                    int(w * (1 - margin)), int(h * (1 - margin)))

        y_indices, x_indices = np.where(binary)
        x1, x2 = x_indices.min(), x_indices.max()
        y1, y2 = y_indices.min(), y_indices.max()

        # Add padding
        h, w = image.shape[:2]
        padding = 20
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        return (x1, y1, x2, y2)


# =============================================================================
# SAM Segmenter
# =============================================================================

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
        self.feature_detector = LesionFeatureDetector()

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

    def segment_with_point(self, image: np.ndarray,
                           point: Tuple[int, int],
                           point_label: int = 1) -> Tuple[np.ndarray, float]:
        """Segment with a point prompt"""
        if self.predictor is None:
            self.load_model()

        self.predictor.set_image(image)

        input_point = np.array([[point[0], point[1]]])
        input_label = np.array([point_label])

        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        best_idx = np.argmax(scores)
        return masks[best_idx], scores[best_idx]

    def segment_with_box(self, image: np.ndarray,
                         box: Tuple[int, int, int, int]) -> Tuple[np.ndarray, float]:
        """Segment with a bounding box prompt"""
        if self.predictor is None:
            self.load_model()

        self.predictor.set_image(image)
        input_box = np.array([box])

        masks, scores, logits = self.predictor.predict(
            box=input_box,
            multimask_output=True,
        )

        best_idx = np.argmax(scores)
        return masks[best_idx], scores[best_idx]

    def segment_center_focused(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Strategy 1: Center-focused segmentation
        Assumes the lesion is near the center of the image
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        # Try center point
        mask, score = self.segment_with_point(image, center)

        # Also try with center bounding box
        margin = 0.15
        box = (int(w * margin), int(h * margin),
               int(w * (1 - margin)), int(h * (1 - margin)))
        mask_box, score_box = self.segment_with_box(image, box)

        # Return better result
        if score_box > score:
            return {
                'mask': mask_box,
                'score': float(score_box),
                'method': 'center_box',
                'prompt': box
            }
        else:
            return {
                'mask': mask,
                'score': float(score),
                'method': 'center_point',
                'prompt': center
            }

    def segment_lesion_features(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Strategy 2: Lesion-feature-based segmentation
        Detects lesion based on color and texture/elevation changes
        """
        # Get candidate points from feature analysis
        candidate_points = self.feature_detector.get_lesion_candidate_points(image, n_points=3)

        # Get bounding box from feature analysis
        feature_box = self.feature_detector.get_lesion_bounding_box(image)

        best_mask = None
        best_score = 0
        best_method = None
        best_prompt = None

        # Try each candidate point
        for i, point in enumerate(candidate_points):
            try:
                mask, score = self.segment_with_point(image, point)
                if score > best_score:
                    best_mask = mask
                    best_score = score
                    best_method = f'feature_point_{i}'
                    best_prompt = point
            except Exception:
                continue

        # Try feature-based bounding box
        try:
            mask_box, score_box = self.segment_with_box(image, feature_box)
            if score_box > best_score:
                best_mask = mask_box
                best_score = score_box
                best_method = 'feature_box'
                best_prompt = feature_box
        except Exception:
            pass

        if best_mask is None:
            # Fallback to center
            return self.segment_center_focused(image)

        return {
            'mask': best_mask,
            'score': float(best_score),
            'method': best_method,
            'prompt': best_prompt
        }

    def segment_both_strategies(self, image: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """
        Run both segmentation strategies and return both results
        """
        results = {}

        print("    - Center-focused segmentation...")
        results['center_focused'] = self.segment_center_focused(image)

        print("    - Lesion-feature-based segmentation...")
        results['lesion_features'] = self.segment_lesion_features(image)

        return results


# =============================================================================
# SAM2 Segmenter
# =============================================================================

class SAM2Segmenter:
    """SAM2 (Segment Anything Model 2) for segmentation"""

    CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
    CHECKPOINT_NAME = "sam2.1_hiera_large.pt"
    CONFIG_NAME = "sam2.1_hiera_l.yaml"

    def __init__(self, checkpoint_dir: str = "checkpoints", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.model = None
        self.predictor = None
        self.feature_detector = LesionFeatureDetector()

    def download_checkpoint(self) -> str:
        """Download SAM2 checkpoint if not exists"""
        checkpoint_path = self.checkpoint_dir / self.CHECKPOINT_NAME

        if not checkpoint_path.exists():
            print(f"Downloading SAM2 checkpoint to {checkpoint_path}...")
            urllib.request.urlretrieve(self.CHECKPOINT_URL, checkpoint_path)
            print("Download complete!")

        return str(checkpoint_path)

    def load_model(self):
        """Load SAM2 model"""
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError:
            raise ImportError("Please install SAM2: pip install git+https://github.com/facebookresearch/segment-anything-2.git")

        checkpoint_path = self.download_checkpoint()

        print(f"Loading SAM2 model on {self.device}...")
        self.model = build_sam2(self.CONFIG_NAME, checkpoint_path, device=self.device)
        self.predictor = SAM2ImagePredictor(self.model)
        print("SAM2 model loaded successfully!")

    def segment_with_point(self, image: np.ndarray,
                           point: Tuple[int, int]) -> Tuple[np.ndarray, float]:
        """Segment with a point prompt"""
        if self.predictor is None:
            self.load_model()

        with torch.inference_mode():
            self.predictor.set_image(image)

            masks, scores, _ = self.predictor.predict(
                point_coords=np.array([[point[0], point[1]]]),
                point_labels=np.array([1]),
                multimask_output=True,
            )

        best_idx = np.argmax(scores)
        return masks[best_idx], float(scores[best_idx])

    def segment_with_box(self, image: np.ndarray,
                         box: Tuple[int, int, int, int]) -> Tuple[np.ndarray, float]:
        """Segment with a bounding box prompt"""
        if self.predictor is None:
            self.load_model()

        with torch.inference_mode():
            self.predictor.set_image(image)

            masks, scores, _ = self.predictor.predict(
                box=np.array(box),
                multimask_output=True,
            )

        best_idx = np.argmax(scores)
        return masks[best_idx], float(scores[best_idx])

    def segment_center_focused(self, image: np.ndarray) -> Dict[str, Any]:
        """Strategy 1: Center-focused segmentation"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        mask, score = self.segment_with_point(image, center)

        margin = 0.15
        box = (int(w * margin), int(h * margin),
               int(w * (1 - margin)), int(h * (1 - margin)))
        mask_box, score_box = self.segment_with_box(image, box)

        if score_box > score:
            return {
                'mask': mask_box,
                'score': float(score_box),
                'method': 'center_box',
                'prompt': box
            }
        else:
            return {
                'mask': mask,
                'score': float(score),
                'method': 'center_point',
                'prompt': center
            }

    def segment_lesion_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Strategy 2: Lesion-feature-based segmentation"""
        candidate_points = self.feature_detector.get_lesion_candidate_points(image, n_points=3)
        feature_box = self.feature_detector.get_lesion_bounding_box(image)

        best_mask = None
        best_score = 0
        best_method = None
        best_prompt = None

        for i, point in enumerate(candidate_points):
            try:
                mask, score = self.segment_with_point(image, point)
                if score > best_score:
                    best_mask = mask
                    best_score = score
                    best_method = f'feature_point_{i}'
                    best_prompt = point
            except Exception:
                continue

        try:
            mask_box, score_box = self.segment_with_box(image, feature_box)
            if score_box > best_score:
                best_mask = mask_box
                best_score = score_box
                best_method = 'feature_box'
                best_prompt = feature_box
        except Exception:
            pass

        if best_mask is None:
            return self.segment_center_focused(image)

        return {
            'mask': best_mask,
            'score': float(best_score),
            'method': best_method,
            'prompt': best_prompt
        }

    def segment_both_strategies(self, image: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """Run both segmentation strategies"""
        results = {}

        print("    - Center-focused segmentation...")
        results['center_focused'] = self.segment_center_focused(image)

        print("    - Lesion-feature-based segmentation...")
        results['lesion_features'] = self.segment_lesion_features(image)

        return results


# =============================================================================
# MedSAM2 Segmenter
# =============================================================================

class MedSAM2Segmenter:
    """
    MedSAM2 for medical image segmentation
    Specialized for medical images, fine-tuned on medical datasets

    Reference: https://github.com/bowang-lab/MedSAM2
    """

    # HuggingFace checkpoint URL
    CHECKPOINT_URL = "https://huggingface.co/wanglab/MedSAM2/resolve/main/MedSAM2_latest.pt"
    CHECKPOINT_NAME = "MedSAM2_latest.pt"
    CONFIG_NAME = "sam2.1_hiera_t512.yaml"  # MedSAM2 uses tiny model config

    def __init__(self, checkpoint_dir: str = "checkpoints", medsam2_repo_path: str = None, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.model = None
        self.predictor = None
        self.feature_detector = LesionFeatureDetector()
        self._available = None

        # MedSAM2 repository path (for configs)
        self.medsam2_repo_path = Path(medsam2_repo_path) if medsam2_repo_path else None

        # Try to find MedSAM2 repo in common locations
        if self.medsam2_repo_path is None:
            possible_paths = [
                Path("MedSAM2"),
                self.checkpoint_dir.parent / "MedSAM2",
                Path("/content/MedSAM2"),  # Colab
                Path.home() / "MedSAM2",
            ]
            for p in possible_paths:
                if p.exists() and (p / "sam2").exists():
                    self.medsam2_repo_path = p
                    break

    def _find_checkpoint(self) -> Optional[Path]:
        """Find MedSAM2 checkpoint in various locations"""
        possible_paths = [
            self.checkpoint_dir / self.CHECKPOINT_NAME,
            self.checkpoint_dir / "MedSAM2_2411.pt",
        ]

        if self.medsam2_repo_path:
            possible_paths.extend([
                self.medsam2_repo_path / "checkpoints" / self.CHECKPOINT_NAME,
                self.medsam2_repo_path / "checkpoints" / "MedSAM2_2411.pt",
            ])

        for path in possible_paths:
            if path.exists():
                return path
        return None

    def _find_config(self) -> Optional[Path]:
        """Find MedSAM2 config file"""
        if self.medsam2_repo_path:
            possible_paths = [
                self.medsam2_repo_path / "configs" / self.CONFIG_NAME,
                self.medsam2_repo_path / "sam2" / "configs" / "sam2.1" / "sam2.1_hiera_t.yaml",
            ]
            for path in possible_paths:
                if path.exists():
                    return path
        return None

    def download_checkpoint(self) -> str:
        """Download MedSAM2 checkpoint from HuggingFace"""
        checkpoint_path = self.checkpoint_dir / self.CHECKPOINT_NAME

        if not checkpoint_path.exists():
            print(f"Downloading MedSAM2 checkpoint to {checkpoint_path}...")
            print("(This may take a few minutes, ~300MB)")
            try:
                urllib.request.urlretrieve(self.CHECKPOINT_URL, checkpoint_path)
                print("Download complete!")
            except Exception as e:
                print(f"Download failed: {e}")
                print("Please manually download from: https://huggingface.co/wanglab/MedSAM2")
                raise

        return str(checkpoint_path)

    def is_available(self) -> bool:
        """Check if MedSAM2 is available"""
        if self._available is not None:
            return self._available

        try:
            # Check if sam2 module is importable
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            # Check for checkpoint (or ability to download)
            checkpoint = self._find_checkpoint()
            if checkpoint is None:
                # We can download it
                self._available = True
            else:
                self._available = True

        except ImportError:
            print("SAM2 module not found. Please install: pip install git+https://github.com/facebookresearch/segment-anything-2.git")
            self._available = False
        except Exception as e:
            print(f"MedSAM2 availability check failed: {e}")
            self._available = False

        return self._available

    def load_model(self):
        """Load MedSAM2 model"""
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError:
            raise ImportError(
                "SAM2 not installed. Please install:\n"
                "pip install git+https://github.com/facebookresearch/segment-anything-2.git"
            )

        # Find or download checkpoint
        checkpoint_path = self._find_checkpoint()
        if checkpoint_path is None:
            checkpoint_path = Path(self.download_checkpoint())

        # Find config or use default
        config_path = self._find_config()

        print(f"Loading MedSAM2 model on {self.device}...")
        print(f"  Checkpoint: {checkpoint_path}")

        try:
            if config_path:
                print(f"  Config: {config_path}")
                self.model = build_sam2(str(config_path), str(checkpoint_path), device=self.device)
            else:
                # Use default SAM2 config (will work but may not be optimal)
                print("  Config: Using default sam2.1_hiera_t.yaml")
                self.model = build_sam2("sam2.1_hiera_t.yaml", str(checkpoint_path), device=self.device)

            self.predictor = SAM2ImagePredictor(self.model)
            print("MedSAM2 model loaded successfully!")

        except Exception as e:
            raise ImportError(
                f"Failed to load MedSAM2: {e}\n\n"
                "Please ensure MedSAM2 is properly installed:\n"
                "1. git clone https://github.com/bowang-lab/MedSAM2.git\n"
                "2. cd MedSAM2 && pip install -e .\n"
                "3. bash download.sh"
            )

    def segment_with_point(self, image: np.ndarray,
                           point: Tuple[int, int]) -> Tuple[np.ndarray, float]:
        """Segment with a point prompt"""
        if self.predictor is None:
            self.load_model()

        with torch.inference_mode():
            self.predictor.set_image(image)
            masks, scores, _ = self.predictor.predict(
                point_coords=np.array([[point[0], point[1]]], dtype=np.float32),
                point_labels=np.array([1], dtype=np.int32),
                multimask_output=False,  # MedSAM2 style: single mask
            )

        # Handle both single and multi-mask output
        if len(masks.shape) == 4:
            masks = masks.squeeze(0)
        if len(scores.shape) > 0:
            best_idx = np.argmax(scores)
            return masks[best_idx], float(scores[best_idx])
        return masks[0], float(scores)

    def segment_with_box(self, image: np.ndarray,
                         box: Tuple[int, int, int, int]) -> Tuple[np.ndarray, float]:
        """Segment with a bounding box prompt"""
        if self.predictor is None:
            self.load_model()

        with torch.inference_mode():
            self.predictor.set_image(image)
            masks, scores, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=np.array(box, dtype=np.float32)[None, :],  # Add batch dim
                multimask_output=False,
            )

        if len(masks.shape) == 4:
            masks = masks.squeeze(0)
        if len(scores.shape) > 0:
            best_idx = np.argmax(scores)
            return masks[best_idx], float(scores[best_idx])
        return masks[0], float(scores)

    def segment_center_focused(self, image: np.ndarray) -> Dict[str, Any]:
        """Strategy 1: Center-focused segmentation"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        mask, score = self.segment_with_point(image, center)

        margin = 0.15
        box = (int(w * margin), int(h * margin),
               int(w * (1 - margin)), int(h * (1 - margin)))
        mask_box, score_box = self.segment_with_box(image, box)

        if score_box > score:
            return {
                'mask': mask_box,
                'score': float(score_box),
                'method': 'center_box',
                'prompt': box
            }
        else:
            return {
                'mask': mask,
                'score': float(score),
                'method': 'center_point',
                'prompt': center
            }

    def segment_lesion_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Strategy 2: Lesion-feature-based segmentation"""
        candidate_points = self.feature_detector.get_lesion_candidate_points(image, n_points=3)
        feature_box = self.feature_detector.get_lesion_bounding_box(image)

        best_mask = None
        best_score = 0
        best_method = None
        best_prompt = None

        for i, point in enumerate(candidate_points):
            try:
                mask, score = self.segment_with_point(image, point)
                if score > best_score:
                    best_mask = mask
                    best_score = score
                    best_method = f'feature_point_{i}'
                    best_prompt = point
            except Exception:
                continue

        try:
            mask_box, score_box = self.segment_with_box(image, feature_box)
            if score_box > best_score:
                best_mask = mask_box
                best_score = score_box
                best_method = 'feature_box'
                best_prompt = feature_box
        except Exception:
            pass

        if best_mask is None:
            return self.segment_center_focused(image)

        return {
            'mask': best_mask,
            'score': float(best_score),
            'method': best_method,
            'prompt': best_prompt
        }

    def segment_both_strategies(self, image: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """Run both segmentation strategies"""
        results = {}

        print("    - Center-focused segmentation...")
        results['center_focused'] = self.segment_center_focused(image)

        print("    - Lesion-feature-based segmentation...")
        results['lesion_features'] = self.segment_lesion_features(image)

        return results


# =============================================================================
# Utility Functions
# =============================================================================

def apply_mask_to_image(image: np.ndarray, mask: np.ndarray,
                        color: Tuple[int, int, int] = (255, 0, 0),
                        alpha: float = 0.5) -> np.ndarray:
    """Apply colored mask overlay to image"""
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
    """Crop the masked region from image with padding"""
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
    """Create image showing only the masked region"""
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
    import sys

    if len(sys.argv) < 2:
        print("Usage: python sam_masking.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    image = load_image(image_path)

    print("Testing SAM segmentation with both strategies...")
    sam = SAMSegmenter()
    results = sam.segment_both_strategies(image)

    for strategy, result in results.items():
        print(f"\n{strategy}:")
        print(f"  Method: {result['method']}")
        print(f"  Score: {result['score']:.3f}")

        overlay = apply_mask_to_image(image, result['mask'])
        output_path = image_path.replace('.', f'_sam_{strategy}.')
        save_image(overlay, output_path)
        print(f"  Saved to: {output_path}")
