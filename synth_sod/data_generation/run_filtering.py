#!/usr/bin/env python3
"""
Dataset Filtering Pipeline - Focused on IoU Consistency and VLM Quality

Usage:
    python run_filtering.py config_path=filtering_config.yaml
"""

import os
import logging
from pathlib import Path
from typing import List
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

from filter_dataset import DatasetLoader, DatasetFilter, Sample


def get_task_subset(samples: List[Sample], max_tasks: int = 12) -> List[Sample]:
    """Split samples for SLURM array jobs, similar to feature extraction."""
    if "SLURM_ARRAY_TASK_ID" not in os.environ:
        return samples
    
    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    total = len(samples)
    base_size = total // max_tasks
    remainder = total % max_tasks
    
    if task_id < remainder:
        start = task_id * (base_size + 1)
        size = base_size + 1
    else:
        start = remainder * (base_size + 1) + (task_id - remainder) * base_size
        size = base_size
    
    end = min(start + size, total)
    subset = samples[start:end]
    
    logging.info(f"Task {task_id}: Processing {len(subset)} samples ({start}-{end-1})")
    return subset


def filter_already_processed(samples: List[Sample], output_path: str) -> List[Sample]:
    """Filter out samples that have already been processed to allow resuming."""
    output_dir = Path(output_path)
    images_dir = output_dir / "images"
    
    if not images_dir.exists():
        return samples
    
    # Get processed filenames
    processed_files = {f.stem for f in images_dir.glob("*.jpg")}
    
    # Filter unprocessed samples
    unprocessed = []
    for sample in samples:
        expected_filename = f"{sample.class_name}_{sample.sample_id}"
        if expected_filename not in processed_files:
            unprocessed.append(sample)
    
    logging.info(f"Filtered {len(samples)} -> {len(unprocessed)} unprocessed samples")
    return unprocessed


@hydra.main(version_base=None, config_path=".", config_name="filtering_config")
def main(cfg: DictConfig) -> None:
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, cfg.log_level, logging.INFO),
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Load all samples
    loader = DatasetLoader(cfg.input_path)
    all_samples = loader.load_samples()
    
    # Get task subset for SLURM array jobs
    max_tasks = cfg.get('max_tasks', 12)
    task_samples = get_task_subset(all_samples, max_tasks)
    
    # Filter already processed samples
    task_samples = filter_already_processed(task_samples, cfg.output_path)
    
    if not task_samples:
        logging.info("No samples to process in this task")
        return
    
    # Apply max_samples_per_class limit within this task
    if cfg.get('max_samples_per_class'):
        samples_by_class = {}
        for sample in task_samples:
            if sample.class_name not in samples_by_class:
                samples_by_class[sample.class_name] = []
            samples_by_class[sample.class_name].append(sample)
        
        # Limit each class
        limited_samples = []
        for class_name, class_samples in samples_by_class.items():
            limited = class_samples[:cfg.max_samples_per_class]
            limited_samples.extend(limited)
            logging.info(f"Class {class_name}: {len(class_samples)} -> {len(limited)} samples")
        
        task_samples = limited_samples
    
    # Create filters from config
    filters = []
    for filter_config in cfg.filters:
        filter_instance = instantiate(filter_config)
        filters.append(filter_instance)
    
    # Initialize filter pipeline
    filter_pipeline = DatasetFilter(filters)
    
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    logging.info(f"Task {task_id}: Starting filtering of {len(task_samples)} samples")
    
    # Create temporary samples by class structure for the filtering function
    samples_by_class = {}
    for sample in task_samples:
        if sample.class_name not in samples_by_class:
            samples_by_class[sample.class_name] = []
        samples_by_class[sample.class_name].append(sample)
    
    # Process each class in this task
    output_dir = Path(cfg.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_processed = 0
    total_passed = 0
    total_failed = 0
    
    for class_name, class_samples in samples_by_class.items():
        logging.info(f"Task {task_id}: Filtering class {class_name} ({len(class_samples)} samples)")
        
        class_processed = 0
        class_passed = 0
        class_failed = 0
        
        for sample in class_samples:
            passed, filter_results = filter_pipeline.filter_sample(sample)
            class_processed += 1
            total_processed += 1
            
            if passed:
                # Copy valid sample to flat output structure
                filter_pipeline._copy_sample(sample, output_dir)
                class_passed += 1
                total_passed += 1
                logging.debug(f"✓ Passed: {sample.image_path.name}")
            else:
                # Save failed sample if enabled
                if cfg.get('save_fail_cases', True):
                    filter_pipeline._copy_failed_sample(sample, output_dir, filter_results)
                
                class_failed += 1
                total_failed += 1
                
                # Log failure reason
                failed_filter = next(name for name, result in filter_results.items() if not result.passed)
                reason = filter_results[failed_filter].reason or "Unknown"
                logging.debug(f"✗ Failed ({failed_filter}): {sample.image_path.name} - {reason}")
        
        logging.info(f"Task {task_id}: Class {class_name} - {class_passed}/{class_processed} passed ({class_passed/class_processed:.1%})")
    
    # Log task summary
    logging.info(f"Task {task_id} completed: {total_passed}/{total_processed} passed ({total_passed/total_processed:.1%})")
    
    # Update filter stats for logging
    filter_pipeline.global_stats.update({
        'total_samples': total_processed,
        'passed_samples': total_passed,
        'failed_samples': total_failed,
        'overall_pass_rate': total_passed / total_processed if total_processed > 0 else 0.0,
        'task_id': task_id
    })


if __name__ == "__main__":
    main() 