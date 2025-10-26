#!/usr/bin/env python3
"""
Unified script to generate captions and tags for test datasets.
Used for teacher model evaluation that requires FLUX features.
"""

import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import logging
from fire import Fire
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

# Test datasets to process (from compute_metrics.py)
DIS_DATASETS = ['DIS-TE1', 'DIS-TE2', 'DIS-TE3', 'DIS-TE4', 'DIS-VD']
SOD_DATASETS = ['HRSOD-TE', 'UHRSD-TE', 'ECSSD', 'DUTS-TE', 'HKU-IS', 'DUT-OMRON', 'DAVIS-S']
TEST_DATASETS = DIS_DATASETS + SOD_DATASETS


class TestMetadataGenerator:
    """Generates captions and tags for test datasets using Gemma3 VLM."""
    
    def __init__(self, model_id: str = "google/gemma-3-4b-it"):
        """Initialize the metadata generator with Gemma3 model."""
        logger.info(f"Loading model: {model_id}")
        
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        ).eval()
        
        self.processor = AutoProcessor.from_pretrained(model_id)
        logger.info("Model initialized successfully")
    
    def is_image_file(self, filepath: str) -> bool:
        """Check if file is a supported image format."""
        return Path(filepath).suffix.lower() in IMAGE_EXTENSIONS
    
    def load_image(self, image_path: str) -> Image.Image:
        """Load and convert image to RGB."""
        try:
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            raise Exception(f"Error loading image {image_path}: {e}")
    
    def generate_caption(self, image_path: str) -> str:
        """Generate detailed caption for an image."""
        image = self.load_image(image_path)
        
        system_prompt = """You are an expert image captioning model. Your task is to analyze the provided image and generate a detailed, accurate description.

The description should:
1. Be 1-2 sentences long
2. Describe the main subjects, objects, and scene elements
3. Include relevant details about colors, composition, and setting
4. Be clear and informative
5. Focus on what is actually visible in the image

Provide only the caption without any additional text."""

        messages = [
            {
                "role": "system", 
                "content": [{"type": "text", "text": system_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Please provide a detailed caption for this image:"}
                ]
            }
        ]

        model_inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        )
        
        if torch.cuda.is_available():
            model_inputs = {k: v.cuda() if hasattr(v, 'cuda') else v for k, v in model_inputs.items()}

        input_len = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **model_inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
            generation = generation[0][input_len:]

        caption = self.processor.decode(generation, skip_special_tokens=True).strip()
        caption = caption.replace('\u201c', '"').replace('\u201d', '"')
        
        return caption
    
    def generate_tag(self, image_path: str) -> str:
        """Generate short object tag for an image."""
        image = self.load_image(image_path)
        
        system_prompt = """You are an expert object detection model. Your task is to identify the main foreground object in the image and provide a short, high-level class name.

Rules:
1. Output ONLY 1-2 words maximum
2. Focus on the main foreground object (the most prominent subject)
3. Use high-level categories (e.g., "dog" not "labrador", "car" not "sedan")
4. Common classes include: person, dog, cat, bird, car, ship, plane, horse, cow, bike, etc.
5. If multiple objects, choose the most prominent/central one
6. Use simple, common English words
7. Do NOT include articles (a, an, the) or descriptive adjectives

Respond with ONLY the object class name, nothing else."""

        messages = [
            {
                "role": "system", 
                "content": [{"type": "text", "text": system_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "What is the main foreground object in this image? Provide only the class name."}
                ]
            }
        ]

        model_inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        )
        
        if torch.cuda.is_available():
            model_inputs = {k: v.cuda() if hasattr(v, 'cuda') else v for k, v in model_inputs.items()}

        input_len = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **model_inputs,
                max_new_tokens=5,
                do_sample=False,
                temperature=0.1,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
            generation = generation[0][input_len:]

        tag = self.processor.decode(generation, skip_special_tokens=True).strip()
        tag = tag.replace('\u201c', '"').replace('\u201d', '"')
        tag = tag.lower().strip()
        
        # Clean and limit to 2 words
        tag = ''.join(char for char in tag if char.isalnum() or char.isspace())
        words = tag.split()
        tag = ' '.join(words[:2])
        
        return tag


def get_image_files(data_folder: str, dataset: str) -> List[str]:
    """Get all image files from a dataset's images subfolder."""
    dataset_path = os.path.join(data_folder, dataset, "images")
    
    if not os.path.exists(dataset_path):
        logger.warning(f"Dataset path does not exist: {dataset_path}")
        return []
    
    image_files = []
    for filename in os.listdir(dataset_path):
        filepath = os.path.join(dataset_path, filename)
        if os.path.isfile(filepath) and Path(filepath).suffix.lower() in IMAGE_EXTENSIONS:
            image_files.append(filepath)
    
    return sorted(image_files)


def load_existing_data(file_path: str, data_type: str) -> Dict[str, str]:
    """Load existing captions or tags from JSON file."""
    if not os.path.exists(file_path):
        return {}
    
    try:
        with open(file_path, 'r') as f:
            existing_data = json.load(f)
        
        existing_dict = {item['image_path']: item[data_type] for item in existing_data}
        logger.info(f"Loaded {len(existing_dict)} existing {data_type}s from {file_path}")
        return existing_dict
    
    except Exception as e:
        logger.error(f"Error loading existing {data_type}s: {e}")
        return {}


def save_data(data_list: List[Dict[str, str]], output_path: str, data_type: str) -> None:
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(data_list, f, indent=2)
    
    logger.info(f"Saved {len(data_list)} {data_type}s to {output_path}")


def process_dataset(
    generator: TestMetadataGenerator,
    data_folder: str,
    dataset: str,
    save_folder: str,
    skip_existing: bool = True
) -> Dict[str, int]:
    """Process a single dataset to generate captions and tags."""
    logger.info(f"Processing dataset: {dataset}")
    
    # Get image files
    image_files = get_image_files(data_folder, dataset)
    if not image_files:
        logger.warning(f"No images found for {dataset}")
        return {"processed": 0, "skipped": 0}
    
    logger.info(f"Found {len(image_files)} images in {dataset}")
    
    # Setup output paths
    dataset_save_dir = Path(save_folder) / dataset
    caption_file = dataset_save_dir / "captions.json"
    tags_file = dataset_save_dir / "tags.json"
    
    # Load existing data
    existing_captions = load_existing_data(str(caption_file), "caption") if skip_existing else {}
    existing_tags = load_existing_data(str(tags_file), "tag") if skip_existing else {}
    
    # Initialize data structures
    captions_data = []
    tags_data = []
    
    # Add existing data
    for image_path, caption in existing_captions.items():
        captions_data.append({"image_path": image_path, "caption": caption})
    
    for image_path, tag in existing_tags.items():
        tags_data.append({"image_path": image_path, "tag": tag})
    
    processed_count = 0
    skipped_count = len(existing_captions)
    
    # Process each image
    for image_path in tqdm(image_files, desc=f"Processing {dataset}"):
        try:
            # Process captions
            if not skip_existing or image_path not in existing_captions:
                caption = generator.generate_caption(image_path)
                captions_data.append({
                    "image_path": image_path,
                    "caption": caption
                })
                logger.debug(f"Caption for {image_path}: {caption}")
            
            # Process tags
            if not skip_existing or image_path not in existing_tags:
                tag = generator.generate_tag(image_path)
                tags_data.append({
                    "image_path": image_path,
                    "tag": tag
                })
                logger.debug(f"Tag for {image_path}: {tag}")
                
                processed_count += 1
        
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            continue
    
    # Save results
    save_data(captions_data, str(caption_file), "caption")
    save_data(tags_data, str(tags_file), "tag")
    
    return {"processed": processed_count, "skipped": skipped_count}


def get_datasets(datasets: str) -> List[str]:
    """Get dataset list based on category selection."""
    if datasets == 'all':
        return TEST_DATASETS
    elif datasets == 'dis':
        return DIS_DATASETS
    elif datasets == 'sod':
        return SOD_DATASETS
    else:
        # Assume it's a custom list or single dataset
        return [datasets] if isinstance(datasets, str) else datasets


def main(
    data_folder: str,
    save_folder: str,
    datasets: str = 'all',
    skip_existing: bool = True,
    model_id: str = "google/gemma-3-4b-it"
):
    """
    Generate captions and tags for test datasets.
    
    Args:
        data_folder: Root folder containing dataset subfolders
        save_folder: Root folder to save metadata (structure: save_folder/dataset_name/captions.json|tags.json)
        datasets: Dataset selection ('all', 'dis', 'sod', or specific dataset name)
        skip_existing: Skip images that have already been processed
        model_id: Hugging Face model ID for caption/tag generation
    """
    
    datasets_list = get_datasets(datasets)
    
    logger.info(f"Starting test metadata generation pipeline...")
    logger.info(f"Data folder: {data_folder}")
    logger.info(f"Save folder: {save_folder}")
    logger.info(f"Dataset selection: {datasets}")
    logger.info(f"Datasets to process: {datasets_list}")
    logger.info(f"Model: {model_id}")
    
    # Initialize generator
    generator = TestMetadataGenerator(model_id)
    
    # Process each dataset
    total_processed = 0
    total_skipped = 0
    
    for dataset in datasets_list:
        stats = process_dataset(
            generator, 
            data_folder, 
            dataset, 
            save_folder, 
            skip_existing
        )
        total_processed += stats["processed"]
        total_skipped += stats["skipped"]
        
    
    logger.info(f"\nTest metadata generation completed!")
    logger.info(f"Total new images processed: {total_processed}")
    logger.info(f"Total images skipped: {total_skipped}")
    logger.info(f"Results saved to: {save_folder}")
    
    # Print summary structure
    logger.info(f"\nGenerated structure:")
    for dataset in datasets_list:
        dataset_dir = Path(save_folder) / dataset
        if dataset_dir.exists():
            caption_file = dataset_dir / "captions.json"
            tags_file = dataset_dir / "tags.json"
            logger.info(f"  {dataset}/")
            if caption_file.exists():
                logger.info(f"    captions.json")
            if tags_file.exists():
                logger.info(f"    tags.json")


if __name__ == "__main__":
    Fire(main) 
