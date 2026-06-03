import torch
from diffusers import FluxPipeline
from diffusers.models import FluxTransformer2DModel
import logging
import numpy as np
from typing import Dict, Any, Optional, List

from data_generation.concept_attention.flux_with_concept_attention_pipeline import FluxWithConceptAttentionPipeline
from data_generation.concept_attention.flux_dit_with_concept_attention import FluxTransformer2DModelWithConceptAttention


def optimize(pipe: FluxPipeline, compile: bool = True) -> FluxPipeline:
    # fuse QKV projections in Transformer and VAE
    pipe.vae.fuse_qkv_projections()

    # switch memory layout to Torch's preferred, channels_last
    pipe.transformer.to(memory_format=torch.channels_last)
    pipe.vae.to(memory_format=torch.channels_last)

    if not compile:
        return pipe

    # set torch compile flags
    config = torch._inductor.config
    config.disable_progress = False  # show progress bar
    config.conv_1x1_as_mm = True  # treat 1x1 convolutions as matrix muls
    # adjust autotuning algorithm
    config.coordinate_descent_tuning = True
    config.coordinate_descent_check_all_directions = True
    config.epilogue_fusion = False  # do not fuse pointwise ops into matmuls

    # tag the compute-intensive modules, the Transformer and VAE decoder, for compilation
    pipe.transformer = torch.compile(
        pipe.transformer, mode="max-autotune", fullgraph=True
    )
    pipe.vae.decode = torch.compile(
        pipe.vae.decode, mode="max-autotune", fullgraph=True
    )

    # trigger torch compilation
    print("🔦 running torch compiliation (may take up to 20 minutes)...")

    pipe(
        "dummy prompt to trigger torch compilation",
        output_type="pil",
        num_inference_steps=50,  # use ~50 for [dev], smaller for [schnell]
    ).images[0]

    print("🔦 finished torch compilation")

    return pipe


def get_pipeline(model_path: str, lora_path: Optional[str] = None, enable_concept_attention: bool = True):
    """Get FLUX pipeline with optional concept attention and LoRA."""
    
    if enable_concept_attention:
        # Load transformer with concept attention
        transformer = FluxTransformer2DModelWithConceptAttention.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, subfolder="transformer"
        )
        
        # Create pipeline with concept attention
        pipe = FluxWithConceptAttentionPipeline.from_pretrained(
            model_path, transformer=transformer, torch_dtype=torch.bfloat16
        )
    else:
        # Regular FLUX pipeline
        pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    
    if lora_path:
        pipe.load_lora_weights(lora_path)
        logging.info(f"Loaded LoRA weights from {lora_path}")
    
    # Enable optimizations
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    
    logging.info(f"Pipeline loaded from {model_path}")
    return pipe


class FluxImageGeneratorWithFeatures:
    """Enhanced FLUX image generator that returns features for mask prediction."""
    
    def __init__(
        self, 
        model_path: str, 
        lora_path: Optional[str] = None,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        feature_layers: List[int] = [0, 1, 2, 3],
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.feature_layers = feature_layers
        
        # Load pipeline with concept attention
        self.pipeline = get_pipeline(model_path, lora_path, enable_concept_attention=True)
        self.pipeline.to(device)
        
        # Get reference to transformer for feature extraction
        self.transformer = self.pipeline.transformer
        
        logging.info("FluxImageGeneratorWithFeatures initialized")
    
    def _setup_scheduler(self, height: int, width: int):
        """Setup scheduler for specific image dimensions."""
        image_seq_len = (height // 16) * (width // 16)
        base_seq_len, max_seq_len = 256, 4096
        base_shift, max_shift = 0.5, 1.15
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = max(base_shift, min(max_shift, image_seq_len * m + b))
        
        self.pipeline.scheduler.set_timesteps(50, device=self.device, mu=mu)
        return self.pipeline.scheduler.timesteps
    
    def _extract_features_from_result(self, result) -> Dict[str, Any]:
        """Extract transformer features and concept maps from generation result."""
        # Process concept attention maps
        concept_maps = {}
        if hasattr(result, 'concept_attention_maps') and result.concept_attention_maps:
            concept_maps_batch = result.concept_attention_maps[0]
            
            if len(concept_maps_batch) > 0:
                category_map = concept_maps_batch[0]  # First concept
                concept_maps['category'] = torch.from_numpy(np.array(category_map, dtype=np.float32)).float()
            
            if len(concept_maps_batch) > 1:
                background_map = concept_maps_batch[1]  # Background
                concept_maps['background'] = torch.from_numpy(np.array(background_map, dtype=np.float32)).float()
        
        # Process transformer features
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
        
        # Clear stored features
        self.transformer.stored_features.clear()
        
        return {
            'transformer_features': processed_features,
            'concept_maps': concept_maps
        }
    
    def generate_with_features(self, prompt: str, tag: str, width: int, height: int) -> Dict[str, Any]:
        """
        Generate image with FLUX features for mask prediction.
        
        Args:
            prompt: Text prompt for image generation
            tag: Class tag for concept attention (e.g., "dog", "car")
            width: Image width
            height: Image height
            
        Returns:
            Dictionary with 'image', 'transformer_features', and 'concept_maps'
        """
        # Setup scheduler
        timesteps = self._setup_scheduler(height, width)
        
        # Define concepts for attention
        concepts = [tag, "background"]
        
        # Generate image with concept attention
        concept_attention_kwargs = {
            "concepts": concepts,
            "timesteps": list(range(25, 28)),  # Last 3 timesteps
            "layers": list(range(18)),
        }
        
        result = self.pipeline(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            concept_attention_kwargs=concept_attention_kwargs,
            generator=torch.Generator(self.device).manual_seed(42)
        )
        
        generated_image = result.images[0]
        
        # Extract features from generation
        features_data = self._extract_features_from_result(result)
        
        # Clear cache
        torch.cuda.empty_cache()
        
        return {
            'image': generated_image,
            'transformer_features': features_data['transformer_features'],
            'concept_maps': features_data['concept_maps']
        }
    
    def generate(self, prompt: str, width: int, height: int):
        """Backward compatibility: generate image only."""
        return self.pipeline(
            prompt,
            output_type="pil",
            num_inference_steps=self.num_inference_steps,
            width=width,
            height=height,
            max_sequence_length=512
        ).images[0]
