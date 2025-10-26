import pytest
import numpy as np
from PIL import Image
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "s3od"))

from s3od import BackgroundRemoval, RemovalResult


def calculate_iou(pred_mask, gt_mask, threshold=0.5):
    """Calculate IoU between predicted and ground truth masks."""
    pred_binary = (pred_mask > threshold).astype(np.float32)
    gt_binary = (gt_mask > threshold).astype(np.float32)
    
    intersection = np.sum(pred_binary * gt_binary)
    union = np.sum(pred_binary) + np.sum(gt_binary) - intersection
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return intersection / union


class TestBackgroundRemoval:
    
    def test_result_dataclass(self):
        """Test RemovalResult dataclass structure."""
        predicted_mask = np.random.rand(480, 640)
        all_masks = np.random.rand(3, 480, 640)
        all_ious = np.array([0.8, 0.9, 0.7])
        rgba = Image.new('RGBA', (640, 480))
        
        result = RemovalResult(
            predicted_mask=predicted_mask,
            all_masks=all_masks,
            all_ious=all_ious,
            rgba_image=rgba
        )
        
        assert result.predicted_mask.shape == (480, 640)
        assert result.all_masks.shape == (3, 480, 640)
        assert len(result.all_ious) == 3
        assert isinstance(result.rgba_image, Image.Image)
    
    def test_preprocessing(self, random_image):
        """Test preprocessing pipeline."""
        # This will fail without a model, which is expected
        with pytest.raises(ValueError):
            model = BackgroundRemoval(model_id="nonexistent_model.pt")
    
    @pytest.mark.skipif(not Path("model.pt").exists(), 
                       reason="Model file not found")
    def test_model_inference_quality(self, test_image_with_mask):
        """Test model inference quality with fixture image."""
        model = BackgroundRemoval(model_id="model.pt")
        
        result = model.remove_background(test_image_with_mask['image'])
        
        # Calculate IoU between predicted and ground truth
        iou = calculate_iou(
            result.predicted_mask,
            test_image_with_mask['mask'],
            threshold=0.5
        )
        
        # Assert IoU is at least 0.9
        assert iou >= 0.9, f"IoU {iou:.3f} is below threshold 0.9"
        assert isinstance(result, RemovalResult)
        assert result.predicted_mask.shape == test_image_with_mask['mask'].shape
    
    @pytest.mark.skipif(not Path("model.pt").exists(),
                       reason="Model file not found")
    def test_remove_background_numpy_input(self, random_image):
        """Test background removal with numpy input."""
        model = BackgroundRemoval(model_id="model.pt")
        result = model.remove_background(random_image)
        
        assert isinstance(result, RemovalResult)
        assert result.predicted_mask.shape == random_image.shape[:2]
        assert isinstance(result.rgba_image, Image.Image)
    
    @pytest.mark.skipif(not Path("model.pt").exists(),
                       reason="Model file not found")
    def test_remove_background_pil_input(self, test_image_with_mask):
        """Test background removal with PIL input."""
        model = BackgroundRemoval(model_id="model.pt")
        result = model.remove_background(test_image_with_mask['image_pil'])
        
        assert isinstance(result, RemovalResult)
        assert result.predicted_mask.shape == test_image_with_mask['mask'].shape
    
    @pytest.mark.skipif(not Path("model.pt").exists(),
                       reason="Model file not found")
    def test_small_image(self, small_image):
        """Test with small image."""
        model = BackgroundRemoval(model_id="model.pt")
        result = model.remove_background(small_image)
        assert result.predicted_mask.shape == (100, 100)
    
    @pytest.mark.skipif(not Path("model.pt").exists(),
                       reason="Model file not found")
    def test_large_image(self, large_image):
        """Test with large image."""
        model = BackgroundRemoval(model_id="model.pt")
        result = model.remove_background(large_image)
        assert result.predicted_mask.shape == (2000, 2000)
    
    @pytest.mark.skipif(not Path("model.pt").exists(),
                       reason="Model file not found")
    def test_rectangular_images(self):
        """Test with rectangular images."""
        model = BackgroundRemoval(model_id="model.pt")
        
        wide_image = np.random.randint(0, 255, (400, 800, 3), dtype=np.uint8)
        tall_image = np.random.randint(0, 255, (800, 400, 3), dtype=np.uint8)
        
        result_wide = model.remove_background(wide_image)
        result_tall = model.remove_background(tall_image)
        
        assert result_wide.predicted_mask.shape == (400, 800)
        assert result_tall.predicted_mask.shape == (800, 400)
    
    @pytest.mark.skipif(not Path("model.pt").exists(),
                       reason="Model file not found")
    def test_multiple_masks_returned(self, test_image_with_mask):
        """Test that all masks and IoU scores are returned."""
        model = BackgroundRemoval(model_id="model.pt")
        result = model.remove_background(test_image_with_mask['image'])
        
        assert result.all_masks is not None
        assert result.all_ious is not None
        assert len(result.all_masks) == len(result.all_ious)
        assert result.all_masks.ndim == 3
