import os
import json
from typing import List, Dict, Any
from fire import Fire
import torch
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

# Initialize model and processor
model_id = "google/gemma-3-4b-it"
model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
).eval()
processor = AutoProcessor.from_pretrained(model_id)

# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


def is_image_file(filepath: str) -> bool:
    """Check if file is a supported image format"""
    return Path(filepath).suffix.lower() in IMAGE_EXTENSIONS


def load_image(image_path: str) -> Image.Image:
    """Load and convert image to RGB"""
    try:
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    except Exception as e:
        raise Exception(f"Error loading image {image_path}: {e}")


def tag_image(image_path: str) -> str:
    """
    Generate a short tag for the main foreground object in an image using Gemma3 VLM
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Generated tag string (1-2 words max)
    """
    # Load image
    image = load_image(image_path)
    
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

    # Process input
    model_inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    )
    
    if torch.cuda.is_available():
        model_inputs = {k: v.cuda() if hasattr(v, 'cuda') else v for k, v in model_inputs.items()}

    input_len = model_inputs["input_ids"].shape[-1]

    # Generate tag
    with torch.inference_mode():
        generation = model.generate(
            **model_inputs,
            max_new_tokens=5,  # Reduced for short tags
            do_sample=False,   # Use greedy decoding for more consistent results
            temperature=0.1,   # Lower temperature for more focused outputs
            pad_token_id=processor.tokenizer.eos_token_id
        )
        generation = generation[0][input_len:]

    # Decode the generated text
    tag = processor.decode(generation, skip_special_tokens=True).strip()
    
    # Clean up tag and ensure it's short
    tag = tag.replace('\u201c', '"').replace('\u201d', '"')
    tag = tag.lower().strip()
    
    # Remove any extra punctuation and split to get first 1-2 words
    tag = ''.join(char for char in tag if char.isalnum() or char.isspace())
    words = tag.split()
    tag = ' '.join(words[:2])  # Keep max 2 words
    
    logger.info(f"Generated tag for {image_path}: {tag}")
    return tag


def get_image_files(data_folder: str, dataset: str) -> List[str]:
    """
    Get all image files from a dataset's images subfolder
    
    Args:
        data_folder: Root data folder path
        dataset: Dataset name
        
    Returns:
        List of image file paths
    """
    dataset_path = os.path.join(data_folder, dataset, "images")
    
    if not os.path.exists(dataset_path):
        print(f"Warning: Dataset path does not exist: {dataset_path}")
        return []
    
    image_files = []
    for filename in os.listdir(dataset_path):
        filepath = os.path.join(dataset_path, filename)
        if os.path.isfile(filepath) and is_image_file(filepath):
            image_files.append(filepath)
    
    return sorted(image_files)


def load_existing_tags(output_path: str) -> Dict[str, str]:
    """
    Load existing tags from output file if it exists
    
    Args:
        output_path: Path to output JSON file
        
    Returns:
        Dictionary mapping image paths to tags
    """
    if not os.path.exists(output_path):
        return {}
    
    try:
        with open(output_path, 'r') as f:
            existing_data = json.load(f)
        
        # Convert list to dict for faster lookup
        # Support both old format (caption) and new format (tag)
        existing_tags = {}
        for item in existing_data:
            if 'tag' in item:
                existing_tags[item['image_path']] = item['tag']
            elif 'caption' in item:
                existing_tags[item['image_path']] = item['caption']
        
        print(f"Loaded {len(existing_tags)} existing tags")
        return existing_tags
    
    except Exception as e:
        print(f"Error loading existing tags: {e}")
        return {}


def save_tags(tags_data: List[Dict[str, str]], output_path: str) -> None:
    """
    Save tags data to JSON file
    
    Args:
        tags_data: List of tag dictionaries
        output_path: Path to save the JSON file
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(tags_data, f, indent=2)
    
    print(f"Saved {len(tags_data)} tags to {output_path}")


def main(
    data_folder: str,
    output_path: str,
    save_interval: int = 50,
    skip_existing: bool = True
):
    """
    Main function for image tagging
    
    Args:
        data_folder: Root folder containing dataset subfolders
        output_path: Path to save the output JSON file
        save_interval: Save progress every N images (0 to save only at the end)
        skip_existing: Skip images that have already been tagged
    """
    
    # Define datasets
    datasets = ['DIS-TR', 'HRSOD-TR', 'UHRSD-TR', 'DUTS-TR']
    
    print(f"Starting image tagging pipeline...")
    print(f"Data folder: {data_folder}")
    print(f"Output path: {output_path}")
    print(f"Datasets: {datasets}")
    
    # Load existing tags if skip_existing is True
    existing_tags = load_existing_tags(output_path) if skip_existing else {}
    
    # Collect all image files from all datasets
    all_image_files = []
    for dataset in datasets:
        dataset_images = get_image_files(data_folder, dataset)
        all_image_files.extend(dataset_images)
        print(f"Found {len(dataset_images)} images in {dataset}")
    
    print(f"Total images to process: {len(all_image_files)}")
    
    if not all_image_files:
        print("No images found. Exiting.")
        return
    
    # Initialize results list
    tags_data = []
    
    # Add existing tags to results
    for image_path, tag in existing_tags.items():
        tags_data.append({
            "image_path": image_path,
            "tag": tag
        })
    
    processed_count = 0
    skipped_count = len(existing_tags)
    
    # Process each image
    for idx, image_path in enumerate(all_image_files):
        
        # Skip if already processed
        if skip_existing and image_path in existing_tags:
            continue
        
        print(f"\nProcessing image {idx + 1}/{len(all_image_files)}")
        print(f"Image: {image_path}")
        
        try:
            # Generate tag
            tag = tag_image(image_path)
            
            # Add to results
            tags_data.append({
                "image_path": image_path,
                "tag": tag
            })
            
            processed_count += 1
            print(f"Tag: {tag}")
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
        
        # Save progress at intervals
        if save_interval > 0 and processed_count % save_interval == 0:
            save_tags(tags_data, output_path)
            print(f"Saved intermediate results - {processed_count} new tags processed")
    
    # Final save
    save_tags(tags_data, output_path)
    
    print(f"\nImage tagging completed!")
    print(f"Total new images processed: {processed_count}")
    print(f"Total images skipped: {skipped_count}")
    print(f"Total tags in output: {len(tags_data)}")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    Fire(main)
