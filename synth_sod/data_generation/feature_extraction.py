"""
Feature extraction pipeline using FLUX Concept Attention and Transformer features.
"""

import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
import torch
from PIL import Image
import fire
from tqdm import tqdm

from data_generation.resizer import FluxResizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from concept_attention.flux_with_concept_attention_pipeline import FluxWithConceptAttentionPipeline
from concept_attention.flux_dit_with_concept_attention import FluxTransformer2DModelWithConceptAttention

@dataclass 
class ImageMetadata:
    image_path: str
    caption: str
    tag: str


class FluxFeatureExtractor:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda"
    ):
        self.device = device
        self.resizer = FluxResizer()
        
        logger.info(f"Loading FLUX model from {model_path}")
        
        # Load transformer and pipeline
        self.transformer = FluxTransformer2DModelWithConceptAttention.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, subfolder="transformer"
        )
        self.pipeline = FluxWithConceptAttentionPipeline.from_pretrained(
            model_path, transformer=self.transformer, torch_dtype=torch.bfloat16
        ).to(device)
        
        self.pipeline.enable_vae_slicing()
        torch.cuda.empty_cache()
        
        logger.info("Feature extractor initialized")

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

    def extract_features(self, image_path: str, caption: str, tag: str) -> Dict[str, Any]:
        """
        Extract meaningful concept attention maps and transformer features.
        
        This extraction process:
        1. Resizes image to FLUX/DINO compatible resolution 
        2. Runs FLUX img2img with concept attention to extract:
           - Category-specific attention maps (what regions contain the specified object)
           - Background attention maps (what regions are background)
           - Multi-layer transformer features (deep semantic representations)
        3. Returns features that can be used to train a segmentation model
        
        Args:
            image_path: Path to input image
            caption: Descriptive caption for the image
            tag: Category/object tag for concept attention (e.g., "person", "car")
            
        Returns:
            Dictionary with extracted features and metadata
        """
        # Load and resize image using consistent resizer
        image_pil = Image.open(image_path).convert("RGB")
        image_resized, (target_h, target_w) = self.resizer.resize_pil_image(image_pil)
        
        # Verify dimensions are compatible
        assert FluxResizer.verify_compatibility(target_h, target_w), \
            f"Resized dimensions {target_h}x{target_w} not compatible with FLUX/DINO"
        
        # Setup scheduler for this image size
        timesteps = self._setup_scheduler(target_h, target_w)
        
        # Define concepts for attention extraction
        concepts = [tag, "background"]  # Dynamic tag + static background
        result = self.pipeline(
            prompt=caption,  # Description helps guide attention extraction
            image=image_resized,  # Input image to analyze
            height=target_h,
            width=target_w,
            timesteps=[int(timesteps[-1])],  # Single timestep for efficiency
            num_inference_steps=1,  # Minimal processing to preserve original content
            guidance_scale=3.5,  # Moderate guidance for stable attention
            concept_attention_kwargs={
                "concepts": concepts,
                "timesteps": [0],  # Extract attention from first timestep
                "layers": list(range(18)),  # All dual transformer layers
            },
            generator=torch.Generator(self.device).manual_seed(42)  # Reproducible
        )
        
        # Process concept attention maps
        features = {"image_resolution": (target_h, target_w)}
        
        if result.concept_attention_maps and len(result.concept_attention_maps) > 0:
            concept_maps = result.concept_attention_maps[0]  # First batch
            
            # Category attention: highlights regions containing the specified object
            if len(concept_maps) > 0:
                category_map = concept_maps[0]  # First concept (tag)
                features["category"] = np.array(category_map, dtype=np.float32)
            
            # Background attention: highlights background regions
            if len(concept_maps) > 1:
                background_map = concept_maps[1]  # Second concept (background)
                features["background"] = np.array(background_map, dtype=np.float32)
        
        # Process transformer features from multiple layers
        _, transformer_features = self.transformer.get_features()
        
        logger.debug(f"Extracted {len(transformer_features)} transformer layers")
        
        for i, feat in enumerate(transformer_features):
            # feat shape: [B=1, H*W, C=3072] - raw tokens without spatial reshaping
            feat_np = feat.detach().cpu().float().numpy()
            
            # Remove batch dimension: [H*W, C=3072]
            feat_np = feat_np[0]
            
            # Channel compression: 3072 -> 768 by averaging groups of 4
            # This reduces storage while preserving semantic information
            seq_len, C = feat_np.shape
            new_C = C // 4  # 768 channels
            feat_reshaped = feat_np[:, :new_C*4].reshape(seq_len, new_C, 4)
            feat_compressed = feat_reshaped.mean(axis=2)  # [H*W, 768]
            
            # Convert to fp16 for storage efficiency and store as tokens
            features[f"layer_{i}"] = feat_compressed.astype(np.float16)
        
        # Clear stored features for next image
        self.transformer.stored_features.clear()
        torch.cuda.empty_cache()
        
        # Return features with metadata
        return {
            "features": features,
            "metadata": {
                "image_path": image_path,
                "tag": tag,
                "original_size": image_pil.size,  # (W, H)
                "processed_size": (target_w, target_h),  # (W, H)
                "caption": caption
            }
        }


def load_metadata(caption_file: str, tag_file: str) -> List[ImageMetadata]:
    with open(caption_file) as f:
        captions = {item["image_path"]: item["caption"] for item in json.load(f)}
    with open(tag_file) as f:
        tags = {item["image_path"]: item["tag"] for item in json.load(f)}
    
    common_paths = sorted(set(captions.keys()) & set(tags.keys()))
    return [ImageMetadata(path, captions[path], tags[path]) for path in common_paths]


def get_task_subset(metadata: List[ImageMetadata], max_tasks: int = 12) -> List[ImageMetadata]:
    if "SLURM_ARRAY_TASK_ID" not in os.environ:
        return metadata
    
    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    total = len(metadata)
    base_size = total // max_tasks
    remainder = total % max_tasks
    
    if task_id < remainder:
        start = task_id * (base_size + 1)
        size = base_size + 1
    else:
        start = remainder * (base_size + 1) + (task_id - remainder) * base_size
        size = base_size
    
    end = min(start + size, total)
    subset = metadata[start:end]
    
    logger.info(f"Task {task_id}: Processing {len(subset)} images ({start}-{end-1})")
    return subset


def get_image_id(image_path: str) -> str:
    path_obj = Path(image_path)
    basename = path_obj.stem
    
    # Extract dataset name from path
    for part in path_obj.parts:
        if part in ['DUTS-TR', 'DIS-TR', 'HRSOD-TR', 'UHRSD-TR']:
            return f"{part}_{basename}"
    
    # Fallback to hash
    import hashlib
    path_hash = hashlib.md5(image_path.encode()).hexdigest()[:8]
    return f"hash_{path_hash}_{basename}"


def filter_processed(metadata: List[ImageMetadata], save_folder: str) -> List[ImageMetadata]:
    features_dir = Path(save_folder) / "features"
    processed = {f.stem for f in features_dir.glob("*.npz")} if features_dir.exists() else set()
    
    filtered = [m for m in metadata if get_image_id(m.image_path) not in processed]
    logger.info(f"Filtered {len(metadata)} -> {len(filtered)} unprocessed images")
    return filtered


class FeatureStorage:
    def __init__(self, save_folder: str, task_id: int = 0):
        self.save_folder = Path(save_folder)
        self.task_id = task_id
        
        # Create directories
        self.features_dir = self.save_folder / "features"
        self.metadata_dir = self.save_folder / "metadata"
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata = []

    def save_features(self, extraction_result: Dict[str, Any]) -> str:
        """Save features and return image ID."""
        features = extraction_result["features"]
        metadata = extraction_result["metadata"]
        
        image_id = get_image_id(metadata["image_path"])
        
        # Save NPZ file with all features
        npz_path = self.features_dir / f"{image_id}.npz"
        np.savez_compressed(npz_path, **features)
        
        # Store metadata
        self.metadata.append({
            "image_id": image_id,
            "image_path": metadata["image_path"],
            "features_path": str(npz_path.relative_to(self.save_folder)),
            "category": metadata["tag"],
            "original_size": metadata["original_size"],
            "processed_size": metadata["processed_size"],
            "caption": metadata["caption"]
        })
        
        return image_id

    def finalize(self):
        metadata_file = self.metadata_dir / f"task_{self.task_id}.json"
        with open(metadata_file, 'w') as f:
            json.dump({
                "task_id": self.task_id,
                "total_images": len(self.metadata),
                "images": self.metadata
            }, f, indent=2)
        
        logger.info(f"Saved metadata for {len(self.metadata)} images to {metadata_file}")


def extract_features(
    caption_file: str,
    tag_file: str,
    save_folder: str,
    model_path: str,
    device: str = "cuda",
    max_tasks: int = 12
):
    # Load and filter metadata
    metadata = load_metadata(caption_file, tag_file)
    task_metadata = get_task_subset(metadata, max_tasks)
    task_metadata = filter_processed(task_metadata, save_folder)
    
    if not task_metadata:
        logger.info("No images to process")
        return
    
    # Initialize extractor and storage
    extractor = FluxFeatureExtractor(model_path=model_path, device=device)
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    storage = FeatureStorage(save_folder, task_id)
    
    # Process images
    logger.info(f"Processing {len(task_metadata)} images")
    for metadata_item in tqdm(task_metadata, desc=f"Task {task_id}"):
        extraction_result = extractor.extract_features(
            metadata_item.image_path,
            metadata_item.caption,
            metadata_item.tag
        )
        storage.save_features(extraction_result)
    
    storage.finalize()
    logger.info("Feature extraction completed")


def main(
    caption_file: str,
    tag_file: str,
    save_folder: str,
    model_path: str = "/scratch/shared/beegfs/lorenza/FLUX.1-Krea-dev",
    device: str = "cuda",
    max_tasks: int = 12
):
    extract_features(caption_file, tag_file, save_folder, model_path, device, max_tasks)


if __name__ == "__main__":
    fire.Fire(main)
