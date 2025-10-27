from dataclasses import dataclass
from typing import Union, Optional, Dict, Any, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
from huggingface_hub import hf_hub_download

from .utils import get_pad_info, remove_padding
from .model import DPTSegmentation


@dataclass
class RemovalResult:
    predicted_mask: np.ndarray
    all_masks: np.ndarray
    all_ious: np.ndarray
    rgba_image: Image.Image


class BackgroundRemoval:
    DEFAULT_MODEL_ID = "okupyn/s3od"
    DEFAULT_CHECKPOINT_NAME = "s3od.pt"
    
    def __init__(
        self,
        model_id: Optional[str] = None,
        image_size: int = 1024,
        device: Optional[str] = None
    ):
        self.image_size = image_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        model_id = model_id or self.DEFAULT_MODEL_ID
        self.model = self._load_model(model_id)
        self.model.to(self.device)
        self.model.eval()
        
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
    
    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs):
        return cls(model_id=model_id, **kwargs)
    
    def _load_model(self, model_id: str) -> torch.nn.Module:
        try:
            checkpoint_path = hf_hub_download(
                repo_id=model_id, 
                filename=self.DEFAULT_CHECKPOINT_NAME
            )
        except Exception as e:
            if Path(model_id).exists():
                checkpoint_path = model_id
            else:
                raise ValueError(
                    f"Could not load model from {model_id}. "
                    f"Ensure model exists on HuggingFace or provide valid local path. "
                    f"Error: {e}"
                )
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        model = DPTSegmentation(
            num_classes=1,
            num_outputs=3,
            encoder_name='dinov3_base',
            features=256,
            use_bn=True,
            use_clstoken=False
        )
        
        model.load_state_dict(checkpoint['state_dict'])
        return model
    
    def _preprocess(self, image: np.ndarray) -> Tuple[torch.Tensor, Dict[str, Any]]:
        pad_info = get_pad_info(image, self.image_size)
        resized = cv2.resize(image, pad_info['resized_size'][::-1])
        
        padded = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        if pad_info['height_pad'] > 0:
            padded[pad_info['height_pad']:-pad_info['height_pad'], :] = resized
        elif pad_info['width_pad'] > 0:
            padded[:, pad_info['width_pad']:-pad_info['width_pad']] = resized
        else:
            padded = resized
        
        normalized = (padded.astype(np.float32) / 255.0 - self.mean) / self.std
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).float()
        
        return tensor, pad_info
    
    @torch.no_grad()
    def remove_background(
        self,
        image: Union[np.ndarray, Image.Image],
        threshold: float = 0.5
    ) -> RemovalResult:
        if isinstance(image, Image.Image):
            image_pil = image.convert('RGB')
            image = np.array(image_pil)
        else:
            image_pil = Image.fromarray(image)
        
        input_tensor, pad_info = self._preprocess(image)
        input_tensor = input_tensor.to(self.device)
        
        outputs = self.model(input_tensor)
        
        pred_masks = torch.sigmoid(outputs['pred_masks'])
        pred_ious = torch.sigmoid(outputs['pred_iou']).squeeze(0).cpu().numpy()
        
        all_masks_unpadded = remove_padding(pred_masks.squeeze(0), pad_info)
        
        all_masks_resized = F.interpolate(
            all_masks_unpadded.unsqueeze(0),
            size=pad_info['original_size'],
            mode='bilinear',
            align_corners=False,
            antialias=True
        ).squeeze(0).float().cpu().numpy()
        
        best_idx = pred_ious.argmax()
        predicted_mask = all_masks_resized[best_idx]
        
        # Use soft mask for smooth alpha channel (convert 0-1 float to 0-255 uint8)
        alpha_channel = (predicted_mask * 255).astype(np.uint8)
        rgba = np.dstack([image, alpha_channel])
        rgba_image = Image.fromarray(rgba, mode='RGBA')
        
        return RemovalResult(
            predicted_mask=predicted_mask,
            all_masks=all_masks_resized,
            all_ious=pred_ious,
            rgba_image=rgba_image
        )
