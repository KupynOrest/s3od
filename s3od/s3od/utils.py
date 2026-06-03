import numpy as np
import torch
from typing import Dict, Any, Tuple


def get_pad_info(image: np.ndarray, image_size: int = 1024) -> Dict[str, Any]:
    h, w = image.shape[:2]
    scale = image_size / max(h, w)
    new_h = round(h * scale)
    new_w = round(w * scale)
    pad_h = (image_size - new_h) // 2
    pad_w = (image_size - new_w) // 2
    return {
        'height_pad': pad_h,
        'width_pad': pad_w,
        'original_size': (h, w),
        'resized_size': (new_h, new_w)
    }


def remove_padding(masks: torch.Tensor, pad_info: Dict[str, Any]) -> torch.Tensor:
    rh, rw = pad_info['resized_size']
    h_pad, w_pad = pad_info['height_pad'], pad_info['width_pad']
    if h_pad > 0:
        masks = masks[:, h_pad:h_pad + rh, :]
    if w_pad > 0:
        masks = masks[:, :, w_pad:w_pad + rw]
    return masks

