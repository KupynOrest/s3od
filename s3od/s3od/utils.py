import numpy as np
import torch
from typing import Dict, Any, Tuple


def get_pad_info(image: np.ndarray, image_size: int = 1024) -> Dict[str, Any]:
    h, w = image.shape[:2]
    aspect_ratio = w / h

    if aspect_ratio > 1:
        new_w = image_size
        new_h = int(new_w / aspect_ratio)
        pad_h = (image_size - new_h) // 2
        return {
            'height_pad': pad_h,
            'width_pad': 0,
            'original_size': (h, w),
            'resized_size': (new_h, new_w)
        }
    else:
        new_h = image_size
        new_w = int(new_h * aspect_ratio)
        pad_w = (image_size - new_w) // 2
        return {
            'height_pad': 0,
            'width_pad': pad_w,
            'original_size': (h, w),
            'resized_size': (new_h, new_w)
        }


def remove_padding(masks: torch.Tensor, pad_info: Dict[str, Any]) -> torch.Tensor:
    if pad_info['height_pad'] > 0:
        masks = masks[:, pad_info['height_pad']:-pad_info['height_pad'], :]
    if pad_info['width_pad'] > 0:
        masks = masks[:, :, pad_info['width_pad']:-pad_info['width_pad']]
    return masks

