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


def caption_image(image_path: str) -> str:
    """
    Generate caption for an image using Gemma3 VLM
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Generated caption string
    """
    # Load image
    image = load_image(image_path)
    
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

    # Process input
    model_inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    )
    
    if torch.cuda.is_available():
        model_inputs = {k: v.cuda() if hasattr(v, 'cuda') else v for k, v in model_inputs.items()}

    input_len = model_inputs["input_ids"].shape[-1]

    # Generate caption
    with torch.inference_mode():
        generation = model.generate(
            **model_inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=processor.tokenizer.eos_token_id
        )
        generation = generation[0][input_len:]

    # Decode the generated text
    caption = processor.decode(generation, skip_special_tokens=True).strip()
    
    # Clean up caption
    caption = caption.replace('\u201c', '"').replace('\u201d', '"')
    
    logger.info(f"Generated caption for {image_path}: {caption}")
    return caption


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


def load_existing_captions(output_path: str) -> Dict[str, str]:
    """
    Load existing captions from output file if it exists
    
    Args:
        output_path: Path to output JSON file
        
    Returns:
        Dictionary mapping image paths to captions
    """
    if not os.path.exists(output_path):
        return {}
    
    try:
        with open(output_path, 'r') as f:
            existing_data = json.load(f)
        
        # Convert list to dict for faster lookup
        existing_captions = {item['image_path']: item['caption'] for item in existing_data}
        print(f"Loaded {len(existing_captions)} existing captions")
        return existing_captions
    
    except Exception as e:
        print(f"Error loading existing captions: {e}")
        return {}


def save_captions(captions_data: List[Dict[str, str]], output_path: str) -> None:
    """
    Save captions data to JSON file
    
    Args:
        captions_data: List of caption dictionaries
        output_path: Path to save the JSON file
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(captions_data, f, indent=2)
    
    print(f"Saved {len(captions_data)} captions to {output_path}")


def main(
    data_folder: str,
    output_path: str,
    save_interval: int = 50,
    skip_existing: bool = True
):
    """
    Main function for image captioning
    
    Args:
        data_folder: Root folder containing dataset subfolders
        output_path: Path to save the output JSON file
        save_interval: Save progress every N images (0 to save only at the end)
        skip_existing: Skip images that have already been captioned
    """
    
    # Define datasets
    datasets = ['DIS-TR', 'HRSOD-TR', 'UHRSD-TR', 'DUTS-TR']
    
    print(f"Starting image captioning pipeline...")
    print(f"Data folder: {data_folder}")
    print(f"Output path: {output_path}")
    print(f"Datasets: {datasets}")
    
    # Load existing captions if skip_existing is True
    existing_captions = load_existing_captions(output_path) if skip_existing else {}
    
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
    captions_data = []
    
    # Add existing captions to results
    for image_path, caption in existing_captions.items():
        captions_data.append({
            "image_path": image_path,
            "caption": caption
        })
    
    processed_count = 0
    skipped_count = len(existing_captions)
    
    # Process each image
    for idx, image_path in enumerate(all_image_files):
        
        # Skip if already processed
        if skip_existing and image_path in existing_captions:
            continue
        
        print(f"\nProcessing image {idx + 1}/{len(all_image_files)}")
        print(f"Image: {image_path}")
        
        try:
            # Generate caption
            caption = caption_image(image_path)
            
            # Add to results
            captions_data.append({
                "image_path": image_path,
                "caption": caption
            })
            
            processed_count += 1
            print(f"Caption: {caption}")
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
        
        # Save progress at intervals
        if save_interval > 0 and processed_count % save_interval == 0:
            save_captions(captions_data, output_path)
            print(f"Saved intermediate results - {processed_count} new captions processed")
    
    # Final save
    save_captions(captions_data, output_path)
    
    print(f"\nImage captioning completed!")
    print(f"Total new images processed: {processed_count}")
    print(f"Total images skipped: {skipped_count}")
    print(f"Total captions in output: {len(captions_data)}")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    Fire(main)
