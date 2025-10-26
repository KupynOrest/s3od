from typing import Union, Tuple, Dict, Any, Optional
import gc
from dataclasses import dataclass

import hydra
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import albumentations as A
from PIL import Image

from data_generation.resizer import FluxResizer
from data_generation.concept_attention.flux_with_concept_attention_pipeline import FluxWithConceptAttentionPipeline
from data_generation.concept_attention.flux_dit_with_concept_attention import FluxTransformer2DModelWithConceptAttention


@dataclass
class PredictionResult:
    """Standard prediction result structure."""
    binary_mask: np.ndarray  # Thresholded binary mask [H, W]
    soft_mask: np.ndarray    # Raw probability mask [H, W]
    all_masks: Optional[np.ndarray] = None  # All masks [N, H, W] if multiple predictions
    all_ious: Optional[np.ndarray] = None   # All IoU scores [N] if multiple predictions
    
    @property
    def has_multiple_masks(self) -> bool:
        """Check if multiple masks were predicted."""
        return self.all_masks is not None
    
    @property
    def num_masks(self) -> int:
        """Get number of predicted masks."""
        return len(self.all_masks) if self.has_multiple_masks else 1


class SODTeacherPredictor:
    """
    Teacher model predictor that uses FLUX features for segmentation.
    Follows the feature extraction pipeline: resize -> extract FLUX features -> predict.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        flux_model_path: str,
        feature_layers: list = [0, 1, 2, 3],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device
        self.feature_layers = feature_layers
        
        # Initialize FLUX feature extractor
        self.flux_resizer = FluxResizer()
        self._init_flux_pipeline(flux_model_path)
        
        # Load teacher model
        self.model = self._load_checkpoint(checkpoint_path)
        self.model.to(device)
        self.model.eval()
        
        # Setup image normalization (same as dataset)
        self.normalize = A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def _init_flux_pipeline(self, flux_model_path: str):
        """Initialize FLUX pipeline for feature extraction."""
        print(f"Loading FLUX model from {flux_model_path}")
        
        # Use the same efficient initialization as pipeline
        from data_generation.pipeline import get_pipeline
        
        # Load pipeline efficiently (same as FluxImageGeneratorWithFeatures)
        self.flux_pipeline = get_pipeline(flux_model_path, None, enable_concept_attention=True)
        
        # Enable memory optimizations
        self.flux_pipeline.enable_vae_slicing()
        self.flux_pipeline.enable_vae_tiling()
        
        # Move entire pipeline to GPU for speed (predictor needs fast inference)
        self.flux_pipeline.to(self.device)
        
        # Get reference to transformer for feature extraction
        self.transformer = self.flux_pipeline.transformer
        
        print("FLUX feature extractor initialized efficiently")
    
    def _load_checkpoint(self, checkpoint_path: str) -> torch.nn.Module:
        """Load teacher model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'state_dict' in checkpoint:
            model = hydra.utils.instantiate(checkpoint['hyper_parameters']['config'].model)
            state_dict = {
                k.lstrip("model").lstrip("."): v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")
            }
            model.load_state_dict(state_dict)
        else:
            model = checkpoint
        return model
    
    def _setup_scheduler(self, height: int, width: int):
        """Setup scheduler for specific image dimensions."""
        image_seq_len = (height // 16) * (width // 16)
        base_seq_len, max_seq_len = 256, 4096
        base_shift, max_shift = 0.5, 1.15
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = max(base_shift, min(max_shift, image_seq_len * m + b))
        
        self.flux_pipeline.scheduler.set_timesteps(50, device=self.device, mu=mu)
        return self.flux_pipeline.scheduler.timesteps
    
    def extract_flux_features(
        self, 
        image_pil: Image.Image, 
        caption: str = "salient object", 
        tag: str = "object"
    ) -> Dict[str, Any]:
        """
        Extract FLUX features using the same pipeline as training data.
        
        Args:
            image_pil: PIL image
            caption: Caption for attention guidance  
            tag: Object tag for concept attention
            
        Returns:
            Dictionary with transformer features and concept maps
        """
        # Resize image using FluxResizer (same as training)
        image_resized, (target_h, target_w) = self.flux_resizer.resize_pil_image(image_pil)
        
        # Verify dimensions are compatible
        assert FluxResizer.verify_compatibility(target_h, target_w), \
            f"Resized dimensions {target_h}x{target_w} not compatible with FLUX"
        
        # Setup scheduler for this image size
        timesteps = self._setup_scheduler(target_h, target_w)
        
        # Define concepts for attention extraction
        concepts = [tag, "background"]

        concept_attention_kwargs = {
            "concepts": concepts,
            "timesteps": [0],  # Single timestep for noise inversion
            "layers": list(range(18)),  # All dual transformer layers
        }
        
        result = self.flux_pipeline(
            prompt=caption,
            image=image_resized,
            height=target_h,
            width=target_w,
            timesteps=[int(timesteps[-1])],  # Single timestep for efficiency (noise inversion)
            num_inference_steps=1,  # Single denoising step for noise inversion
            guidance_scale=3.5,
            concept_attention_kwargs=concept_attention_kwargs,
            generator=torch.Generator(self.device).manual_seed(42)
        )
        
        # Process concept attention maps
        concept_maps = {}
        if result.concept_attention_maps and len(result.concept_attention_maps) > 0:
            concept_maps_batch = result.concept_attention_maps[0]  # First batch
            
            if len(concept_maps_batch) > 0:
                category_map = concept_maps_batch[0]  # First concept (tag)
                concept_maps['category'] = torch.from_numpy(np.array(category_map, dtype=np.float32)).float()
            
            if len(concept_maps_batch) > 1:
                background_map = concept_maps_batch[1]  # Second concept (background)
                concept_maps['background'] = torch.from_numpy(np.array(background_map, dtype=np.float32)).float()
        
        # Process transformer features (same as pipeline implementation)
        _, transformer_features = self.transformer.get_features()
        processed_features = []
        
        for layer_idx in self.feature_layers:
            if layer_idx < len(transformer_features):
                feat = transformer_features[layer_idx]
                # feat shape: [B=1, H*W, C=3072]
                feat_np = feat.detach().cpu().float().numpy()[0]  # Remove batch dim
                
                # Channel compression: 3072 -> 768
                seq_len, C = feat_np.shape
                new_C = C // 4
                feat_reshaped = feat_np[:, :new_C*4].reshape(seq_len, new_C, 4)
                feat_compressed = feat_reshaped.mean(axis=2)
                
                processed_features.append(torch.from_numpy(feat_compressed.astype(np.float32)))
        
        # Clear stored features (same as pipeline)
        self.transformer.stored_features.clear()
        
        return {
            'transformer_features': processed_features,
            'concept_maps': concept_maps,
            'target_size': (target_h, target_w),
            'resized_image': image_resized
        }
    
    @torch.no_grad()
    def predict(
        self,
        image: Union[np.ndarray, Image.Image],
        caption: str = "salient object",
        tag: str = "object", 
        threshold: float = 0.5
    ) -> PredictionResult:
        """
        Predict masks for input image using teacher model with FLUX features.
        
        Args:
            image: RGB numpy array [H, W, 3] or PIL Image
            caption: Caption for FLUX attention guidance
            tag: Object tag for concept attention
            threshold: Binary threshold for masks
            
        Returns:
            PredictionResult with binary_mask, soft_mask, and optionally all_masks/all_ious
        """
        # Convert to PIL if numpy
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
            original_size = (image.shape[0], image.shape[1])  # (H, W)
        else:
            image_pil = image
            original_size = (image_pil.size[1], image_pil.size[0])  # (H, W)
        
        # Extract FLUX features
        flux_data = self.extract_flux_features(image_pil, caption, tag)
        transformer_features = flux_data['transformer_features']
        concept_maps = flux_data['concept_maps']
        target_h, target_w = flux_data['target_size']
        resized_image = flux_data['resized_image']
        
        # Prepare image tensor for model
        image_np = np.array(resized_image)
        normalized = self.normalize(image=image_np)['image']
        image_tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        
        # Move features to device and add batch dimension
        transformer_features = [feat.unsqueeze(0).to(self.device) for feat in transformer_features]
        concept_maps = {k: v.unsqueeze(0).to(self.device) for k, v in concept_maps.items()}
        
        # Run teacher model
        outputs = self.model(image_tensor, transformer_features, concept_maps)
        
        # Process predictions
        pred_masks = torch.sigmoid(outputs['pred_masks'])  # [B, N, H, W]
        num_masks = pred_masks.size(1)
        
        # Resize all masks to original size
        all_masks_resized = F.interpolate(
            pred_masks.squeeze(0),  # [N, H, W]
            size=original_size,
            mode='bilinear',
            align_corners=False,
            antialias=True
        ).float().cpu().numpy()  # [N, H, W]
        
        if num_masks == 1:
            # Single mask case
            soft_mask = all_masks_resized.squeeze(0)  # [H, W]
            binary_mask = (soft_mask > threshold).astype(np.float32)
            
            return PredictionResult(
                binary_mask=binary_mask,
                soft_mask=soft_mask
            )
        else:
            # Multiple masks case
            pred_ious = torch.sigmoid(outputs['pred_iou']).squeeze(0).cpu().numpy()  # [N]
            best_idx = pred_ious.argmax()
            
            # Best mask
            soft_mask = all_masks_resized[best_idx]  # [H, W]
            binary_mask = (soft_mask > threshold).astype(np.float32)
            
            # All masks and IoUs
            all_binary_masks = (all_masks_resized > threshold).astype(np.float32)
            
            return PredictionResult(
                binary_mask=binary_mask,
                soft_mask=soft_mask,
                all_masks=all_binary_masks,
                all_ious=pred_ious
            )


class SODPredictor:
    def __init__(
            self,
            checkpoint_path: str,
            image_size: int = 840,
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        # Load model
        self.device = device
        self.image_size = image_size
        self.model = self._load_checkpoint(checkpoint_path)
        self.model.to(device)
        self.model.eval()

        # Setup transforms
        self.transform = A.Compose([
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def _load_checkpoint(self, checkpoint_path: str) -> torch.nn.Module:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'state_dict' in checkpoint:
            model = hydra.utils.instantiate(checkpoint['hyper_parameters']['config'].model)
            state_dict = {
                k.lstrip("model").lstrip("."): v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")
            }
            model.load_state_dict(state_dict)
        else:
            model = checkpoint
        return model

    def get_pad_info(self, image: np.ndarray) -> dict:
        """Calculate padding information for the image."""
        h, w = image.shape[:2]
        aspect_ratio = w / h

        if aspect_ratio > 1:
            new_w = self.image_size
            new_h = int(new_w / aspect_ratio)
            pad_h = (self.image_size - new_h) // 2
            return {
                'height_pad': pad_h,
                'width_pad': 0,
                'original_size': (h, w),
                'resized_size': (new_h, new_w)
            }
        else:
            new_h = self.image_size
            new_w = int(new_h * aspect_ratio)
            pad_w = (self.image_size - new_w) // 2
            return {
                'height_pad': 0,
                'width_pad': pad_w,
                'original_size': (h, w),
                'resized_size': (new_h, new_w)
            }

    def remove_padding(
            self,
            masks: torch.Tensor,
            pad_info: dict
    ) -> torch.Tensor:
        """Remove padding from predicted masks."""
        if pad_info['height_pad'] > 0:
            masks = masks[:, pad_info['height_pad']:-pad_info['height_pad'], :]
        if pad_info['width_pad'] > 0:
            masks = masks[:, :, pad_info['width_pad']:-pad_info['width_pad']]
        return masks

    @torch.no_grad()
    def predict(
            self,
            image: np.ndarray,
            threshold: float = 0.5
    ) -> PredictionResult:
        """
        Predict masks for input image.

        Args:
            image: RGB numpy array [H, W, 3]
            threshold: Binary threshold for masks

        Returns:
            PredictionResult with binary_mask, soft_mask, and optionally all_masks/all_ious
        """
        pad_info = self.get_pad_info(image)
        transformed = self.transform(image=image)
        input_tensor = torch.from_numpy(transformed['image']).permute(2, 0, 1).unsqueeze(0).to(self.device)
        outputs = self.model(input_tensor)

        pred_masks = torch.sigmoid(outputs['pred_masks'])  # [B, N, H, W]
        pred_ious = torch.sigmoid(outputs['pred_iou'])     # [B, N]
        num_masks = pred_masks.size(1)
        
        # Remove padding from all masks
        all_masks_unpadded = self.remove_padding(pred_masks.squeeze(0), pad_info)  # [N, H, W]
        
        # Resize all masks to original size
        all_masks_resized = F.interpolate(
            all_masks_unpadded.unsqueeze(0),  # Add batch dim for interpolate
            size=pad_info['original_size'],
            mode='bilinear',
            align_corners=False,
            antialias=True
        ).squeeze(0).float().cpu().numpy()  # [N, H, W]

        if num_masks == 1:
            # Single mask case
            soft_mask = all_masks_resized.squeeze(0)  # [H, W]
            binary_mask = (soft_mask > threshold).astype(np.float32)
            
            return PredictionResult(
                binary_mask=binary_mask,
                soft_mask=soft_mask
            )
        else:
            # Multiple masks case
            pred_ious_np = pred_ious.squeeze(0).cpu().numpy()  # [N]
            best_idx = pred_ious_np.argmax()
            
            # Best mask
            soft_mask = all_masks_resized[best_idx]  # [H, W]
            binary_mask = (soft_mask > threshold).astype(np.float32)
            
            # All masks and IoUs
            all_binary_masks = (all_masks_resized > threshold).astype(np.float32)
            
            return PredictionResult(
                binary_mask=binary_mask,
                soft_mask=soft_mask,
                all_masks=all_binary_masks,
                all_ious=pred_ious_np
            )
