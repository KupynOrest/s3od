# S3OD - Simple and Efficient Background Removal

S3OD (Synthetic Salient Object Detection) provides state-of-the-art background removal with a simple, clean API.

## Installation

```bash
pip install s3od
```

## Quick Start

```python
from s3od import BackgroundRemoval
from PIL import Image

# Load default model (okupyn/s3od from HuggingFace)
model = BackgroundRemoval()

# Load image
image = Image.open("input.jpg")

# Remove background
result = model.remove_background(image)

# Save result
result.rgba_image.save("output.png")
```

## Advanced Usage

### Access Multiple Masks

```python
result = model.remove_background(image)

# Best mask (based on predicted IoU)
best_mask = result.predicted_mask

# All predicted masks
all_masks = result.all_masks  # Shape: [N, H, W]

# IoU scores for each mask
all_ious = result.all_ious  # Shape: [N]

# RGBA image with transparent background
rgba = result.rgba_image
```

### Custom Model

```python
# Load custom model from HuggingFace
model = BackgroundRemoval.from_pretrained("username/custom-model")

# Or load local model
model = BackgroundRemoval(model_id="path/to/model.pt")
```

### Visualization

```python
from s3od.visualizer import visualize_removal, visualize_all_masks

# Visualize with green background
vis = visualize_removal(image, result, background_color=(0, 255, 0))
vis.save("visualization.jpg")

# Visualize all masks in a grid
grid = visualize_all_masks(image, result)
grid.save("all_masks.jpg")
```

## Performance

S3OD achieves state-of-the-art performance on standard salient object detection benchmarks:

- DUTS-TE: 0.XXX mIoU
- DIS5K: 0.XXX mIoU
- HRSOD: 0.XXX mIoU

## Model Architecture

S3OD uses a DPT-based architecture with DINOv3 vision transformer backbone for robust feature extraction.

## License

MIT License

## Citation

If you use S3OD in your research, please cite:

```bibtex
@article{s3od2024,
  title={S3OD: Synthetic Salient Object Detection},
  author={Author et al.},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```
