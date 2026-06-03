"""
Test inference using fixture image and mask.
"""
import pytest
import torch
import numpy as np
from PIL import Image
from pathlib import Path

from s3od import BackgroundRemoval


def calculate_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, threshold: float = 0.5) -> float:
    """Calculate IoU between predicted and ground truth masks."""
    pred_binary = (pred_mask > threshold).astype(np.uint8)
    gt_binary = (gt_mask > threshold).astype(np.uint8)
    
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    
    if union == 0:
        return 1.0
    
    return float(intersection / union)


@pytest.fixture
def fixture_image():
    """Load fixture image."""
    fixture_path = Path(__file__).parent / "fixture" / "image.jpg"
    return Image.open(fixture_path).convert('RGB')


@pytest.fixture
def fixture_mask():
    """Load fixture mask."""
    fixture_path = Path(__file__).parent / "fixture" / "mask.png"
    mask = Image.open(fixture_path).convert('L')
    return np.array(mask).astype(np.float32) / 255.0


@pytest.mark.slow
def test_fixture_inference_iou(fixture_image, fixture_mask):
    """Test inference on fixture image and verify IoU >= 0.9 with ground truth."""
    # Load model
    model = BackgroundRemoval()
    
    # Run inference
    result = model.remove_background(fixture_image)
    
    # Check result structure
    assert hasattr(result, 'predicted_mask')
    assert hasattr(result, 'all_masks')
    assert hasattr(result, 'all_ious')
    assert hasattr(result, 'rgba_image')
    
    # Check shapes
    assert result.predicted_mask.shape == fixture_mask.shape
    assert result.all_masks.shape[1:] == fixture_mask.shape
    assert len(result.all_ious) == result.all_masks.shape[0]
    
    # Calculate IoU with ground truth
    iou = calculate_iou(result.predicted_mask, fixture_mask, threshold=0.5)
    
    print(f"\nFixture inference IoU: {iou:.4f}")
    
    # Verify IoU is at least 0.9
    assert iou >= 0.9, f"IoU {iou:.4f} is below threshold of 0.9"


@pytest.mark.slow
def test_fixture_rgba_output(fixture_image, fixture_mask):
    """Test that RGBA output has correct properties."""
    model = BackgroundRemoval()
    result = model.remove_background(fixture_image)
    
    # Check RGBA image
    assert result.rgba_image.mode == 'RGBA'
    assert result.rgba_image.size == fixture_image.size
    
    # Check alpha channel matches predicted mask
    rgba_array = np.array(result.rgba_image)
    alpha_channel = rgba_array[:, :, 3].astype(np.float32) / 255.0
    
    # Alpha should be similar to predicted mask
    alpha_iou = calculate_iou(alpha_channel, result.predicted_mask, threshold=0.5)
    assert alpha_iou > 0.95, "Alpha channel doesn't match predicted mask"


@pytest.mark.slow
def test_fixture_multiple_masks(fixture_image, fixture_mask):
    """Test that model predicts multiple masks with IoU scores."""
    model = BackgroundRemoval()
    result = model.remove_background(fixture_image)
    
    # Should have 3 masks (default num_outputs)
    assert result.all_masks.shape[0] == 3
    assert result.all_ious.shape[0] == 3
    
    # All IoU scores should be between 0 and 1
    assert np.all(result.all_ious >= 0)
    assert np.all(result.all_ious <= 1)
    
    # Best mask should have highest IoU with ground truth
    best_idx = result.all_ious.argmax()
    best_mask = result.all_masks[best_idx]
    
    # Verify predicted_mask is the best one
    np.testing.assert_array_equal(result.predicted_mask, best_mask)
    
    # Calculate IoU for all masks
    ious_with_gt = [calculate_iou(mask, fixture_mask) for mask in result.all_masks]
    print(f"\nIoUs with ground truth: {[f'{iou:.4f}' for iou in ious_with_gt]}")
    print(f"Predicted IoU scores: {[f'{iou:.4f}' for iou in result.all_ious]}")


@pytest.mark.slow  
def test_fixture_threshold_variations(fixture_image, fixture_mask):
    """Test inference with different thresholds."""
    model = BackgroundRemoval()
    
    thresholds = [0.3, 0.5, 0.7]
    ious = []
    
    for threshold in thresholds:
        result = model.remove_background(fixture_image, threshold=threshold)
        iou = calculate_iou(result.predicted_mask, fixture_mask, threshold=0.5)
        ious.append(iou)
        print(f"Threshold {threshold}: IoU = {iou:.4f}")
    
    # At least one threshold should give good results
    assert max(ious) >= 0.9, f"Best IoU {max(ious):.4f} is below 0.9"

