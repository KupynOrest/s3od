import pytest
import numpy as np
from PIL import Image, ImageDraw


@pytest.fixture
def test_image_with_mask():
    """Create a simple test image with a centered circle and corresponding mask."""
    size = 512
    
    # Create white image
    image = Image.new('RGB', (size, size), color='white')
    
    # Draw a red circle in the center
    draw = ImageDraw.Draw(image)
    center = size // 2
    radius = 150
    draw.ellipse(
        [center - radius, center - radius, center + radius, center + radius],
        fill='red'
    )
    
    # Create corresponding mask
    mask = Image.new('L', (size, size), color=0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.ellipse(
        [center - radius, center - radius, center + radius, center + radius],
        fill=255
    )
    
    return {
        'image': np.array(image),
        'mask': np.array(mask) / 255.0,
        'image_pil': image,
        'mask_pil': mask
    }


@pytest.fixture
def random_image():
    """Create a random RGB image for testing."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def small_image():
    """Create a small test image."""
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def large_image():
    """Create a large test image."""
    return np.random.randint(0, 255, (2000, 2000, 3), dtype=np.uint8)

