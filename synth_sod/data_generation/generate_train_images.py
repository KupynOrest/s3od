import os
from os import environ
import yaml
import json
import random
from typing import List, Optional
from pathlib import Path
from dataclasses import dataclass

import logging
from fire import Fire

from data_generation.pipeline import FluxImageGeneratorWithFeatures
from data_generation.prompt_generator import ImagePromptGenerator, PromptEnhancer
from data_generation.mask_generator import create_mask_generator
import numpy as np

FLUX_RESOLUTIONS = [
    (1024, 1024), (896, 1024), (1024, 896), (832, 1024), (1024, 832),
    (1024, 768), (768, 1024), (960, 1024), (1024, 960), (1088, 1024), (1024, 1088)
]

logging.basicConfig(
   level=logging.INFO,
   format='%(asctime)s [%(levelname)s] %(message)s',
   datefmt='%Y-%m-%d %H:%M:%S'
)


@dataclass
class GenerationConfig:
    """Configuration for image generation."""
    model_path: str
    lora_path: Optional[str]
    class_names_path: str
    save_path: str
    num_inference_steps: int
    num_prompts: int
    teacher_checkpoint_path: str  # Now mandatory
    class_weights_path: Optional[str] = None
    max_tasks: int = 8
    use_all_existing_prompts: bool = False


@dataclass
class ClassInfo:
    """Information about a class for generation."""
    name: str
    folder_name: str
    num_samples: int


class TaskDistributor:
    """Handles distribution of classes across SLURM tasks."""
    
    def __init__(self, max_tasks: int = 8):
        self.max_tasks = max_tasks
    
    def get_task_classes(self, all_classes: List[str]) -> List[str]:
        """Get classes for current SLURM task."""
        if "SLURM_ARRAY_TASK_ID" not in environ:
            return all_classes
            
        task_id = int(environ["SLURM_ARRAY_TASK_ID"])
        num_per_task = len(all_classes) // self.max_tasks
        start = task_id * num_per_task
        end = min(len(all_classes), (task_id + 1) * num_per_task)
        return all_classes[start:end]


class ClassWeightLoader:
    """Loads and manages class weights for sampling."""
    
    def __init__(self, weights_path: Optional[str] = None):
        self.weights = {}
        if weights_path and os.path.exists(weights_path):
            self._load_weights(weights_path)
    
    def _load_weights(self, weights_path: str):
        """Load class weights from file."""
        with open(weights_path, 'r') as f:
            weights_data = json.load(f)
            self.weights = weights_data.get('new_samples', {})
        logging.info(f"Loaded class weights for {len(self.weights)} categories")
    
    def get_sample_count(self, class_folder: str, default: int) -> int:
        """Get sample count for a class."""
        return self.weights.get(class_folder, default)


class FilePromptProvider:
    """Provides prompts from files or generates new ones."""
    
    def __init__(self, prompt_generator: ImagePromptGenerator):
        self.prompt_generator = prompt_generator
        self.prompt_enhancer = PromptEnhancer()
    
    def get_prompts(self, class_info: ClassInfo, output_dir: Path, use_all_existing_prompts: bool = False) -> List[str]:
        """Get prompts for a class, loading from file or generating new ones."""
        prompts_file = output_dir / "prompts.txt"
        
        if prompts_file.exists():
            prompts = self._load_existing_prompts(prompts_file)
            
            if use_all_existing_prompts:
                # Use all existing prompts regardless of target number
                logging.info(f"Using all {len(prompts)} existing prompts for {class_info.name}")
                enhanced_prompts = [self.prompt_enhancer(prompt) for prompt in prompts]
                return enhanced_prompts
            elif len(prompts) >= class_info.num_samples:
                logging.info(f"Using existing prompts for {class_info.name}")
                enhanced_prompts = [self.prompt_enhancer(prompt) for prompt in prompts[:class_info.num_samples]]
                return enhanced_prompts
        else:
            prompts = []
        
        # If use_all_existing_prompts is True but no file exists, fall back to generating num_prompts
        if use_all_existing_prompts and not prompts:
            logging.warning(f"No existing prompts found for {class_info.name}, falling back to generating {class_info.num_samples} prompts")
        
        # Generate additional prompts if needed
        needed = class_info.num_samples - len(prompts)
        if needed > 0:
            logging.info(f"Generating {needed} new prompts for {class_info.name}")
            new_prompts = self.prompt_generator.generate_prompts(class_info.name, num_prompts=needed)
            prompts.extend(new_prompts)
            self._save_prompts(prompts, prompts_file)
        
        # Enhance prompts before returning
        enhanced_prompts = [self.prompt_enhancer(prompt) for prompt in prompts[:class_info.num_samples]]
        return enhanced_prompts
    
    def _load_existing_prompts(self, prompts_file: Path) -> List[str]:
        """Load prompts from file."""
        with open(prompts_file, 'r') as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    
    def _save_prompts(self, prompts: List[str], prompts_file: Path):
        """Save prompts to file."""
        with open(prompts_file, 'w') as f:
            for prompt in prompts:
                f.write(prompt + "\n")


class ImageMaskGenerationPipeline:
    """Main pipeline for generating training images and masks."""
    
    def __init__(
        self,
        config: GenerationConfig,
        prompt_provider: FilePromptProvider,
        image_generator: FluxImageGeneratorWithFeatures,
        mask_generator,
        task_distributor: TaskDistributor,
        class_weight_loader: ClassWeightLoader
    ):
        self.config = config
        self.prompt_provider = prompt_provider
        self.image_generator = image_generator
        self.mask_generator = mask_generator
        self.task_distributor = task_distributor
        self.class_weight_loader = class_weight_loader
    
    def run(self):
        """Execute the image and mask generation pipeline."""
        # Load and distribute classes
        all_classes = self._load_class_names()
        task_classes = self.task_distributor.get_task_classes(all_classes)
        
        logging.info(f"Processing {len(task_classes)} classes in this task")
        
        # Process each class
        for class_name in task_classes:
            class_info = self._create_class_info(class_name)
            self._process_class(class_info)
    
    def _load_class_names(self) -> List[str]:
        """Load class names from file."""
        class_path = Path(__file__).parent / self.config.class_names_path
        with open(class_path, "r") as f:
            class_names = json.load(f)
        return list(class_names.values())
    
    def _create_class_info(self, class_name: str) -> ClassInfo:
        """Create class info with proper folder name and sample count."""
        folder_name = class_name.split(',')[0].replace(" ", "_").replace("'", "")
        num_samples = self.class_weight_loader.get_sample_count(folder_name, self.config.num_prompts)
        
        return ClassInfo(
            name=class_name,
            folder_name=folder_name,
            num_samples=num_samples
        )
    
    def _process_class(self, class_info: ClassInfo):
        """Process a single class - generate prompts, images and masks."""
        # Create directory structure
        class_dir = Path(self.config.save_path) / class_info.folder_name
        images_dir = class_dir / "images"
        masks_dir = class_dir / "masks"
        
        class_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(exist_ok=True)
        masks_dir.mkdir(exist_ok=True)
        
        # Get prompts
        prompts = self.prompt_provider.get_prompts(class_info, class_dir, self.config.use_all_existing_prompts)
        
        # Log actual number of samples to be generated
        logging.info(f"Processing {class_info.folder_name}: generating {len(prompts)} samples with masks")
        
        # Extract tag for concept attention - use first word
        tag = class_info.name.split(',')[0].split()[0].lower()
        
        # Generate images and masks
        for idx, prompt in enumerate(prompts):
            image_path = images_dir / f"{idx}.jpg"
            mask_path = masks_dir / f"{idx}.png"
            
            # Skip if both files exist
            if image_path.exists() and mask_path.exists():
                continue
            
            # Choose random resolution
            width, height = random.choice(FLUX_RESOLUTIONS)
            
            try:
                result = self.image_generator.generate_with_features(prompt, tag, width, height)
                
                mask = self.mask_generator.generate_mask(
                    result['image'], 
                    result['transformer_features'], 
                    result['concept_maps']
                )
                
                result['image'].save(image_path, quality=95)
                
                # Proper rounding to avoid quantization artifacts
                mask_uint8 = np.clip(np.round(mask * 255), 0, 255).astype(np.uint8)
                from PIL import Image
                mask_image = Image.fromarray(mask_uint8, mode='L')
                mask_image.save(mask_path)
                
                logging.info(f"Generated sample {idx}: {image_path.name}")
                
            except Exception as e:
                logging.error(f"Failed to generate sample {idx} for {class_info.name}: {e}")
                continue


class PipelineFactory:
    """Factory for creating the image generation pipeline."""
    
    @staticmethod
    def create_from_config(config_path: str) -> ImageMaskGenerationPipeline:
        """Create pipeline from configuration file."""
        # Load configuration
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        config = GenerationConfig(**config_dict)
        logging.info(f"Loaded config: {config}")
        
        # Create components
        task_distributor = TaskDistributor(config.max_tasks)
        class_weight_loader = ClassWeightLoader(config.class_weights_path)
        
        # Create prompt provider
        prompt_generator = ImagePromptGenerator()
        prompt_provider = FilePromptProvider(prompt_generator)
        
        # Create image generator with features
        image_generator = FluxImageGeneratorWithFeatures(
            model_path=config.model_path,
            lora_path=config.lora_path,
            num_inference_steps=config.num_inference_steps
        )
        
        # Create mask generator
        mask_generator = create_mask_generator(
            teacher_checkpoint_path=config.teacher_checkpoint_path
        )
        
        return ImageMaskGenerationPipeline(
            config=config,
            prompt_provider=prompt_provider,
            image_generator=image_generator,
            mask_generator=mask_generator,
            task_distributor=task_distributor,
            class_weight_loader=class_weight_loader
        )


def generate_images(config_path: str = "generation_config.yaml"):
    """Main entry point for image generation."""
    pipeline = PipelineFactory.create_from_config(config_path)
    pipeline.run()


if __name__ == "__main__":
    Fire(generate_images)
