import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List
from PIL import Image
import hydra
import albumentations as A
import logging


class MaskGenerator:
    """Generates masks from images and FLUX features using a pretrained teacher model."""
    
    def __init__(
        self,
        teacher_checkpoint_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        
        # Load teacher model
        self.teacher_model = self._load_teacher_model(teacher_checkpoint_path)
        
        # Setup image normalization
        self.normalize = A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        logging.info("MaskGenerator initialized successfully")
    
    def _load_teacher_model(self, checkpoint_path: str):
        """Load teacher model from checkpoint."""
        logging.info(f"Loading teacher model from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'state_dict' in checkpoint:
            model = hydra.utils.instantiate(checkpoint['hyper_parameters']['config'].model)
            state_dict = {
                k.lstrip("model").lstrip("."): v for k, v in checkpoint["state_dict"].items() 
                if k.startswith("model.")
            }
            model.load_state_dict(state_dict)
        else:
            model = checkpoint
        
        model.to(self.device)
        model.eval()
        
        logging.info("Teacher model loaded successfully")
        return model
    
    @torch.no_grad()
    def generate_mask(
        self, 
        image: Image.Image, 
        transformer_features: List[torch.Tensor], 
        concept_maps: Dict[str, torch.Tensor]
    ) -> np.ndarray:
        """
        Generate mask from image and FLUX features.
        
        Args:
            image: PIL image
            transformer_features: List of transformer feature tensors
            concept_maps: Dictionary with concept attention maps
            
        Returns:
            Mask as numpy array [H, W]
        """
        # Prepare image tensor for teacher model
        image_np = np.array(image)
        normalized = self.normalize(image=image_np)['image']
        image_tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        
        # Move features to device and add batch dimension
        transformer_features = [feat.unsqueeze(0).to(self.device) for feat in transformer_features]
        concept_maps = {k: v.unsqueeze(0).to(self.device) for k, v in concept_maps.items()}
        
        # Predict mask using teacher model
        outputs = self.teacher_model(image_tensor, transformer_features, concept_maps)
        
        # Process mask predictions
        pred_masks = torch.sigmoid(outputs['pred_masks'])
        
        if pred_masks.size(1) == 1:
            best_mask = pred_masks.squeeze(1)
        else:
            pred_ious = torch.sigmoid(outputs['pred_iou'])
            best_indices = pred_ious.argmax(dim=1)
            best_mask = pred_masks[torch.arange(pred_masks.size(0)), best_indices]
        
        # Convert mask to numpy
        mask_np = best_mask.squeeze(0).float().cpu().numpy()
        
        # Clear cache
        torch.cuda.empty_cache()
        
        return mask_np


def create_mask_generator(teacher_checkpoint_path: str, **kwargs) -> MaskGenerator:
    """Factory function to create a mask generator."""
    return MaskGenerator(
        teacher_checkpoint_path=teacher_checkpoint_path,
        **kwargs
    ) 