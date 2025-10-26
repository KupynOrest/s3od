"""
Consistent image resizing for FLUX feature extraction and segmentation training.
Ensures compatibility with FLUX (divisible by 32).
"""

import cv2
import numpy as np
from typing import Tuple
from PIL import Image


class FluxResizer:
    """
    Resizer that ensures images are compatible with FLUX requirements.
    - FLUX: Dimensions divisible by 32 (due to 2x2 packing on top of 16-stride VAE)
    """
    
    # Predefined optimal resolutions (all divisible by 32)
    OPTIMAL_RESOLUTIONS = [
        # Square and near-square
        (1024, 1024),  # 1:1 (32×32, 32×32)
        (896, 1152),   # ~0.78:1 (28×32, 36×32)
        (1152, 896),   # ~1.29:1 (36×32, 28×32)
        (768, 1344),   # ~0.57:1 (24×32, 42×32)
        (1344, 768),   # ~1.75:1 (42×32, 24×32)
        
        # Additional common ratios
        (832, 1216),   # ~0.68:1 (26×32, 38×32)
        (1216, 832),   # ~1.46:1 (38×32, 26×32)
        (704, 1408),   # 0.5:1 (22×32, 44×32)
        (1408, 704),   # 2:1 (44×32, 22×32)
        (960, 1088),   # ~0.88:1 (30×32, 34×32)
        (1088, 960),   # ~1.13:1 (34×32, 30×32)
    ]
    
    def __init__(self):
        """Initialize the resizer with precomputed aspect ratios."""
        self.resolution_aspects = [
            (h, w, w / h) for h, w in self.OPTIMAL_RESOLUTIONS
        ]
    
    def select_best_resolution(self, original_h: int, original_w: int) -> Tuple[int, int]:
        """
        Select the optimal resolution that best matches the original aspect ratio.
        
        Args:
            original_h: Original image height
            original_w: Original image width
            
        Returns:
            Tuple of (target_height, target_width)
        """
        original_aspect = original_w / original_h
        
        best_resolution = None
        min_aspect_diff = float('inf')
        
        for h, w, aspect in self.resolution_aspects:
            aspect_diff = abs(original_aspect - aspect)
            
            if aspect_diff < min_aspect_diff:
                min_aspect_diff = aspect_diff
                best_resolution = (h, w)
        
        return best_resolution
    
    def resize_image(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Resize numpy image to optimal FLUX-compatible resolution.
        
        Args:
            image: Input image as numpy array (H, W, C)
            
        Returns:
            Tuple of (resized_image, (target_height, target_width))
        """
        original_h, original_w = image.shape[:2]
        target_h, target_w = self.select_best_resolution(original_h, original_w)
        
        # Resize image directly (no padding needed)
        resized_image = cv2.resize(image, (target_w, target_h))
        
        return resized_image, (target_h, target_w)
    
    def resize_pil_image(self, image: Image.Image) -> Tuple[Image.Image, Tuple[int, int]]:
        """
        Resize PIL image to optimal FLUX-compatible resolution.
        
        Args:
            image: Input PIL image
            
        Returns:
            Tuple of (resized_image, (target_height, target_width))
        """
        original_w, original_h = image.size  # PIL uses (W, H)
        target_h, target_w = self.select_best_resolution(original_h, original_w)
        
        # Resize image directly
        resized_image = image.resize((target_w, target_h), Image.LANCZOS)
        
        return resized_image, (target_h, target_w)
    
    def resize_mask(self, mask: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize mask to target size using nearest neighbor interpolation.
        
        Args:
            mask: Input mask as numpy array (H, W) or (H, W, 1)
            target_size: (target_height, target_width)
            
        Returns:
            Resized mask
        """
        target_h, target_w = target_size
        
        if len(mask.shape) == 3 and mask.shape[2] == 1:
            mask = mask.squeeze(2)
        
        resized_mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        
        return resized_mask
    
    def get_compatible_resolutions(self) -> list:
        """Return list of all compatible resolutions."""
        return self.OPTIMAL_RESOLUTIONS.copy()
    
    @staticmethod
    def verify_compatibility(height: int, width: int) -> bool:
        """
        Verify that dimensions are compatible with FLUX.
        
        Args:
            height: Image height
            width: Image width
            
        Returns:
            True if compatible, False otherwise
        """
        return (height % 32 == 0) and (width % 32 == 0)
