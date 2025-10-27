import numpy as np
from PIL import Image
from typing import Union, Tuple

from .predictor import RemovalResult


def visualize_removal(
    image: Union[np.ndarray, Image.Image],
    result: RemovalResult,
    background_color: Tuple[int, int, int] = (0, 255, 0)
) -> Image.Image:
    if isinstance(image, Image.Image):
        image = np.array(image.convert('RGB'))
    
    # Use soft mask (0-1 range) for smooth edges, like BiRefNet
    mask = result.predicted_mask[..., None]  # Add channel dimension
    
    background = np.full_like(image, background_color, dtype=np.uint8)
    # Blend using soft mask for smooth transitions
    composite = (mask * image + (1 - mask) * background).astype(np.uint8)
    
    return Image.fromarray(composite)


def visualize_all_masks(
    image: Union[np.ndarray, Image.Image],
    result: RemovalResult
) -> Image.Image:
    if isinstance(image, Image.Image):
        image = np.array(image.convert('RGB'))
    
    h, w = image.shape[:2]
    num_masks = len(result.all_masks)
    
    grid_width = min(num_masks, 4)
    grid_height = (num_masks + grid_width - 1) // grid_width
    
    grid = np.zeros((h * grid_height, w * grid_width, 3), dtype=np.uint8)
    
    for idx, mask in enumerate(result.all_masks):
        row = idx // grid_width
        col = idx % grid_width
        
        mask_rgb = (mask[..., None] * image).astype(np.uint8)
        grid[row*h:(row+1)*h, col*w:(col+1)*w] = mask_rgb
    
    return Image.fromarray(grid)
