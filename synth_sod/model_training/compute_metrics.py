import cv2
from glob import glob
import torch
from tqdm import tqdm
from fire import Fire
import json
import os
from pathlib import Path
from typing import Dict, Optional
import numpy as np

from model_training.predictor import SODPredictor, SODTeacherPredictor
from model_training.metrics import EvaluationMetrics


def load_metadata(metadata_dir: str, dataset: str) -> Dict[str, Dict[str, str]]:
    metadata = {}
    
    captions_file = Path(metadata_dir) / dataset / "captions.json"
    if captions_file.exists():
        with open(captions_file) as f:
            captions_data = json.load(f)
        for item in captions_data:
            path = item["image_path"]
            if path not in metadata:
                metadata[path] = {}
            metadata[path]["caption"] = item["caption"]
    
    tags_file = Path(metadata_dir) / dataset / "tags.json"
    if tags_file.exists():
        with open(tags_file) as f:
            tags_data = json.load(f)
        for item in tags_data:
            path = item["image_path"]
            if path not in metadata:
                metadata[path] = {}
            metadata[path]["tag"] = item["tag"]
    
    return metadata


def process_dataset(data_dir: str, predictor: SODPredictor, compute_best_metrics: bool = False):
    images = glob(f"{data_dir}/images/*")
    metric_counter = EvaluationMetrics(device='cuda')
    best_metric_counter = EvaluationMetrics(device='cuda') if compute_best_metrics else None
    
    for image_path in tqdm(images, desc="Evaluation"):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = predictor.predict(image)
        
        gt_mask_path = find_gt_mask_path(image_path, data_dir)
        if gt_mask_path and os.path.exists(gt_mask_path):
            gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE) > 128
            gt_mask = gt_mask.astype(float)
            gt_mask_tensor = torch.tensor(gt_mask).to('cuda')
            
            # Evaluate predicted (best) mask
            soft_mask = result.soft_mask
            metric_counter.step(torch.tensor(soft_mask).to('cuda'), gt_mask_tensor)
            
            # Evaluate all masks if requested and available
            if compute_best_metrics and result.has_multiple_masks:
                best_iou = -1
                best_mask = None
                
                # Try all masks and find the one with best IoU with ground truth
                for i in range(result.num_masks):
                    mask = result.all_masks[i]  # Binary mask [H, W]
                    
                    # Calculate IoU
                    mask_bool = mask > 0.5
                    gt_bool = gt_mask > 0.5
                    intersection = np.logical_and(mask_bool, gt_bool).sum()
                    union = np.logical_or(mask_bool, gt_bool).sum()
                    iou = intersection / union if union > 0 else 1.0
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_mask = mask
                
                # Use the best mask for metrics (convert to soft mask approximation)
                if best_mask is not None:
                    best_metric_counter.step(torch.tensor(best_mask).to('cuda'), gt_mask_tensor)
                else:
                    # Fallback to predicted mask
                    best_metric_counter.step(torch.tensor(soft_mask).to('cuda'), gt_mask_tensor)
            elif compute_best_metrics:
                # Single mask case - use the same mask for both metrics
                best_metric_counter.step(torch.tensor(soft_mask).to('cuda'), gt_mask_tensor)
        else:
            print(f"Warning: GT mask not found for {image_path}")
    
    pred_metrics = metric_counter.compute_metrics()
    
    if compute_best_metrics:
        best_metrics = best_metric_counter.compute_metrics()
        return {'pred_metrics': pred_metrics, 'best_metrics': best_metrics}
    else:
        return pred_metrics


def process_dataset_teacher(data_dir: str, dataset: str, predictor: SODTeacherPredictor, metadata: Dict[str, Dict[str, str]], compute_best_metrics: bool = False):
    images = glob(f"{data_dir}/images/*")
    metric_counter = EvaluationMetrics(device='cuda')
    best_metric_counter = EvaluationMetrics(device='cuda') if compute_best_metrics else None
    processed_count = 0
    missing_metadata_count = 0
    
    for image_path in tqdm(images, desc="Teacher evaluation"):
        if image_path not in metadata:
            missing_metadata_count += 1
            print(f"Warning: No metadata found for {image_path}")
            continue
        
        image_metadata = metadata[image_path]
        caption = image_metadata.get("caption", "salient object")
        tag = image_metadata.get("tag", "object")
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        result = predictor.predict(image, caption=caption, tag=tag)
        
        gt_mask_path = find_gt_mask_path(image_path, data_dir)
        if gt_mask_path and os.path.exists(gt_mask_path):
            gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE) > 128
            gt_mask = gt_mask.astype(float)
            gt_mask_tensor = torch.tensor(gt_mask).to('cuda')
            
            # Evaluate predicted (best) mask
            soft_mask = result.soft_mask
            metric_counter.step(torch.tensor(soft_mask).to('cuda'), gt_mask_tensor)
            
            # Evaluate all masks if requested and available
            if compute_best_metrics and result.has_multiple_masks:
                best_iou = -1
                best_mask = None
                
                # Try all masks and find the one with best IoU with ground truth
                for i in range(result.num_masks):
                    mask = result.all_masks[i]  # Binary mask [H, W]
                    
                    # Calculate IoU
                    mask_bool = mask > 0.5
                    gt_bool = gt_mask > 0.5
                    intersection = np.logical_and(mask_bool, gt_bool).sum()
                    union = np.logical_or(mask_bool, gt_bool).sum()
                    iou = intersection / union if union > 0 else 1.0
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_mask = mask
                
                # Use the best mask for metrics (convert to soft mask approximation)
                if best_mask is not None:
                    best_metric_counter.step(torch.tensor(best_mask).to('cuda'), gt_mask_tensor)
                else:
                    # Fallback to predicted mask
                    best_metric_counter.step(torch.tensor(soft_mask).to('cuda'), gt_mask_tensor)
            elif compute_best_metrics:
                # Single mask case - use the same mask for both metrics
                best_metric_counter.step(torch.tensor(soft_mask).to('cuda'), gt_mask_tensor)
            
            processed_count += 1
        else:
            print(f"Warning: GT mask not found for {image_path}")
    
    print(f"Processed {processed_count} images, {missing_metadata_count} missing metadata")
    
    pred_metrics = metric_counter.compute_metrics()
    
    if compute_best_metrics:
        best_metrics = best_metric_counter.compute_metrics()
        return {'pred_metrics': pred_metrics, 'best_metrics': best_metrics}
    else:
        return pred_metrics


def find_gt_mask_path(image_path: str, data_dir: str) -> Optional[str]:
    image_name = Path(image_path).stem
    mask_extensions = ['.png', '.jpg', '.jpeg']
    
    masks_dir = os.path.join(data_dir, 'masks')
    for ext in mask_extensions:
        mask_path = os.path.join(masks_dir, image_name + ext)
        if os.path.exists(mask_path):
            return mask_path
    
    for ext in mask_extensions:
        mask_path = image_path.replace('/images/', '/masks/').replace(Path(image_path).suffix, ext)
        if os.path.exists(mask_path):
            return mask_path
    
    return None


def get_datasets(datasets: str):
    dis_datasets = ['DIS-TE1', 'DIS-TE2', 'DIS-TE3', 'DIS-TE4']
    sod_datasets = ['DUTS-TE', 'DUT-OMRON', 'HRSOD-TE', 'UHRSD-TE', 'DAVIS-S']
    if datasets == 'all':
        return dis_datasets + sod_datasets
    elif datasets == 'dis':
        return dis_datasets
    elif datasets == 'sod':
        return sod_datasets
    else:
        raise ValueError(f"Invalid dataset: {datasets}")


def main(
    input_dir: str, 
    model_path: str, 
    model_type: str = 'student',
    metadata_dir: Optional[str] = None,
    img_size: int = 224, 
    datasets: str = 'all',
    compute_best_metrics: bool = False,
    flux_model_path: str
):
    datasets_list = get_datasets(datasets=datasets)
    print(f"Model type: {model_type}")
    print(f"Model path: {model_path}")
    print(f"Datasets: {datasets_list}")
    
    if model_type == 'student':
        predictor = SODPredictor(model_path, img_size, device='cuda')
        for dataset in tqdm(datasets_list, desc="Evaluating datasets"):
            data_input_dir = f"{input_dir}/{dataset}"
            dataset_metrics = process_dataset(data_input_dir, predictor, compute_best_metrics=compute_best_metrics)
            
            if compute_best_metrics and isinstance(dataset_metrics, dict):
                print(f"\nDataset: {dataset}")
                print(f"  Predicted mask metrics: {dataset_metrics['pred_metrics']}")
                print(f"  Best mask metrics: {dataset_metrics['best_metrics']}")
            else:
                print(f"Dataset: {dataset}, Metrics: {dataset_metrics}")
            
    elif model_type == 'teacher':
        if metadata_dir is None:
            raise ValueError("metadata_dir is required for teacher evaluation")
        
        print(f"Metadata directory: {metadata_dir}")
        print(f"FLUX model path: {flux_model_path}")
        
        predictor = SODTeacherPredictor(
            checkpoint_path=model_path,
            flux_model_path=flux_model_path,
            device='cuda'
        )
        
        for dataset in tqdm(datasets_list, desc="Evaluating datasets"):
            print(f"\nEvaluating {dataset}...")
            metadata = load_metadata(metadata_dir, dataset)
            print(f"Loaded metadata for {len(metadata)} images")
            
            data_input_dir = f"{input_dir}/{dataset}"
            dataset_metrics = process_dataset_teacher(data_input_dir, dataset, predictor, metadata, compute_best_metrics=compute_best_metrics)
            
            if compute_best_metrics and isinstance(dataset_metrics, dict):
                print(f"\nDataset: {dataset}")
                print(f"  Predicted mask metrics: {dataset_metrics['pred_metrics']}")
                print(f"  Best mask metrics: {dataset_metrics['best_metrics']}")
            else:
                print(f"Dataset: {dataset}, Metrics: {dataset_metrics}")
            
    else:
        raise ValueError(f"Invalid model_type: {model_type}. Must be 'student' or 'teacher'")


if __name__ == "__main__":
    Fire(main)
