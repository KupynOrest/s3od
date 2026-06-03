import torch
import torch.nn.functional as F
import numpy as np
import logging
import cv2

from data_generation.filter_dataset import BaseFilter, FilterResult, Sample, calculate_iou
from model_training.predictor import SODPredictor


class HorizontalFlipConsistencyFilter(BaseFilter):
    """Filter that validates mask consistency using horizontal flip test with a non-FLUX model."""
    
    def __init__(self, 
                 model_path: str,
                 name: str = "horizontal_flip_consistency",
                 threshold: float = 0.7,
                 consistency_threshold: float = 0.8,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__(name)
        
        self.threshold = threshold  # IoU threshold with generated mask
        self.consistency_threshold = consistency_threshold  # IoU threshold between orig and flipped
        self.device = device
        self.model = None
        self.model_path = model_path
        
        # Lazy loading - only load when first needed
        self._model_loaded = False
    
    def _load_model(self):
        """Lazy load the model to save memory."""
        if self._model_loaded:
            return
            
        self.model = SODPredictor(
            checkpoint_path=self.model_path,
            device=self.device
        )
        self._model_loaded = True
        logging.info(f"Loaded consistency model from {self.model_path}")
    
    def filter(self, sample: Sample) -> FilterResult:
        """
        Test consistency by:
        1. Predict mask on original image
        2. Horizontally flip image, predict mask, flip mask back
        3. Compare IoU with generated mask
        """
        
        # Load model if needed
        if not self._model_loaded:
            self._load_model()
        
        image = sample.load_image()  # RGB [H, W, 3]
        generated_mask = sample.load_mask()  # Grayscale [H, W]
        
        # Get predictions
        original_result = self.model.predict(image)
        original_pred = original_result.binary_mask
        
        # Horizontal flip test
        flipped_image = cv2.flip(image, 1)
        flipped_result = self.model.predict(flipped_image)
        flipped_pred = cv2.flip(flipped_result.binary_mask, 1)
        
        # Calculate IoUs
        iou_orig_generated = self.calculate_iou(original_pred, generated_mask)
        iou_flipped_generated = self.calculate_iou(flipped_pred, generated_mask)
        iou_orig_flipped = self.calculate_iou(original_pred, flipped_pred)
        
        # Score consistency (average performance with generated mask)
        avg_iou = (iou_orig_generated + iou_flipped_generated) / 2
        
        # Pass if both IoUs are above threshold and consistency is reasonable
        passes = (
            iou_orig_generated >= self.threshold and
            iou_flipped_generated >= self.threshold and
            iou_orig_flipped >= self.consistency_threshold
        )
        
        metadata = {
            'iou_orig_generated': iou_orig_generated,
            'iou_flipped_generated': iou_flipped_generated, 
            'iou_orig_flipped': iou_orig_flipped,
            'avg_iou': avg_iou
        }
        
        return FilterResult(
            passes=passes,
            score=avg_iou,
            metadata=metadata
        )
    
    def calculate_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Calculate IoU between two binary masks."""
        # Ensure binary masks
        mask1 = (mask1 > 0.5).astype(bool)
        mask2 = (mask2 > 0.5).astype(bool)
        
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        
        if union == 0:
            return 1.0  # Both masks are empty
        
        return float(intersection / union)            