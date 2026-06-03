import logging
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import torch
from PIL import Image
import cv2


@dataclass
class Sample:
    """Represents a single image-mask pair."""
    image_path: Path
    mask_path: Path
    class_name: str
    sample_id: str
    
    def load_image(self) -> np.ndarray:
        """Load image as RGB numpy array."""
        return np.array(Image.open(self.image_path).convert('RGB'))
    
    def load_mask(self, binary: bool = True) -> np.ndarray:
        """Load mask."""
        if binary:
            return (np.array(Image.open(self.mask_path).convert('L')) > 128).astype(np.uint8)
        else:
            return np.array(Image.open(self.mask_path).convert('L'))
    
    def get_relative_path(self) -> str:
        """Get relative path for copying."""
        return f"{self.class_name}/images/{self.sample_id}.jpg"
    
    def get_mask_relative_path(self) -> str:
        """Get relative mask path for copying."""
        return f"{self.class_name}/masks/{self.sample_id}.png"


@dataclass
class FilterResult:
    """Result of filtering operation."""
    passed: bool
    reason: Optional[str] = None
    score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseFilter(ABC):
    """Base class for all filtering methods."""
    
    def __init__(self, name: str):
        self.name = name
        self.stats = {
            'total_processed': 0,
            'passed': 0,
            'failed': 0
        }
    
    @abstractmethod
    def filter(self, sample: Sample) -> FilterResult:
        """Filter a single sample. Return FilterResult with pass/fail."""
        pass
    
    def update_stats(self, result: FilterResult):
        """Update internal statistics."""
        self.stats['total_processed'] += 1
        if result.passed:
            self.stats['passed'] += 1
        else:
            self.stats['failed'] += 1
    
    def get_pass_rate(self) -> float:
        """Get current pass rate."""
        if self.stats['total_processed'] == 0:
            return 0.0
        return self.stats['passed'] / self.stats['total_processed']
    
    def reset_stats(self):
        """Reset statistics."""
        self.stats = {'total_processed': 0, 'passed': 0, 'failed': 0}


class DatasetLoader:
    """Loads and organizes dataset samples."""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
    
    def load_samples(self) -> List[Sample]:
        """Load all image-mask pairs from dataset."""
        samples = []
        
        for class_dir in self.dataset_path.iterdir():
            if not class_dir.is_dir():
                continue
                
            images_dir = class_dir / "images"
            masks_dir = class_dir / "masks"
            
            if not (images_dir.exists() and masks_dir.exists()):
                logging.warning(f"Skipping {class_dir.name}: missing images or masks directory")
                continue
            
            # Find matching image-mask pairs
            for image_path in images_dir.glob("*.jpg"):
                mask_path = masks_dir / f"{image_path.stem}.png"
                
                if mask_path.exists():
                    sample = Sample(
                        image_path=image_path,
                        mask_path=mask_path,
                        class_name=class_dir.name,
                        sample_id=image_path.stem
                    )
                    samples.append(sample)
                else:
                    logging.warning(f"Missing mask for {image_path}")
        
        logging.info(f"Loaded {len(samples)} samples from {len(set(s.class_name for s in samples))} classes")
        return samples


class DatasetFilter:
    """Main filtering pipeline orchestrator."""
    
    def __init__(self, filters: List[BaseFilter]):
        self.filters = filters
        self.global_stats = {
            'total_samples': 0,
            'passed_samples': 0,
            'failed_samples': 0,
            'filter_breakdown': {}
        }
    
    def filter_sample(self, sample: Sample) -> Tuple[bool, Dict[str, FilterResult]]:
        """Run all filters on a single sample."""
        results = {}
        
        for filter_obj in self.filters:
            result = filter_obj.filter(sample)
            filter_obj.update_stats(result)
            results[filter_obj.name] = result
            
            if not result.passed:
                return False, results
        return True, results
    
    def filter_dataset(self, 
                      input_path: str, 
                      output_path: str, 
                      max_samples_per_class: Optional[int] = None,
                      save_fail_cases: bool = True) -> Dict[str, Any]:
        """Filter entire dataset and copy valid samples to output."""
        
        loader = DatasetLoader(input_path)
        samples = loader.load_samples()
        
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        samples_by_class = {}
        for sample in samples:
            if sample.class_name not in samples_by_class:
                samples_by_class[sample.class_name] = []
            samples_by_class[sample.class_name].append(sample)
        
        total_processed = 0
        total_passed = 0
        total_failed = 0
        class_stats = {}
        for class_name, class_samples in samples_by_class.items():
            logging.info(f"Filtering class: {class_name} ({len(class_samples)} samples)")
            
            class_passed = 0
            class_failed = 0
            class_processed = 0
            
            if max_samples_per_class:
                class_samples = class_samples[:max_samples_per_class]
            
            for sample in class_samples:
                passed, filter_results = self.filter_sample(sample)
                class_processed += 1
                total_processed += 1
                
                if passed:
                    # Copy valid sample to flat output structure
                    self._copy_sample(sample, output_dir)
                    class_passed += 1
                    total_passed += 1
                    logging.debug(f"✓ Passed: {sample.image_path.name}")
                else:
                    # Save failed sample if enabled
                    if save_fail_cases:
                        self._copy_failed_sample(sample, output_dir, filter_results)
                    
                    class_failed += 1
                    total_failed += 1
                    
                    # Log failure reason
                    failed_filter = next(name for name, result in filter_results.items() if not result.passed)
                    reason = filter_results[failed_filter].reason or "Unknown"
                    logging.debug(f"✗ Failed ({failed_filter}): {sample.image_path.name} - {reason}")
            
            class_stats[class_name] = {
                'processed': class_processed,
                'passed': class_passed,
                'failed': class_failed,
                'pass_rate': class_passed / class_processed if class_processed > 0 else 0.0
            }
        
        # Update global stats
        self.global_stats.update({
            'total_samples': total_processed,
            'passed_samples': total_passed,
            'failed_samples': total_failed,
            'overall_pass_rate': total_passed / total_processed if total_processed > 0 else 0.0,
            'class_stats': class_stats,
            'filter_breakdown': {f.name: f.stats for f in self.filters},
            'save_fail_cases': save_fail_cases
        })
        
        return self.global_stats
    
    def _copy_sample(self, sample: Sample, output_dir: Path):
        """Copy sample files to flat output structure: images/class_name_index.jpg, masks/class_name_index.png"""
        # Create flat directory structure
        images_dir = output_dir / "images"
        masks_dir = output_dir / "masks"
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate flat filename: class_name_index
        flat_filename = f"{sample.class_name}_{sample.sample_id}"
        
        output_image = images_dir / f"{flat_filename}.jpg"
        output_mask = masks_dir / f"{flat_filename}.png"
        
        shutil.copy2(sample.image_path, output_image)
        shutil.copy2(sample.mask_path, output_mask)
    
    def _copy_failed_sample(self, sample: Sample, output_dir: Path, filter_results: Dict[str, Any]):
        """Copy failed sample as stacked visualization: image | mask | overlay [| predictions] [with text header]."""
        # Get the first failed filter name and result
        failed_filter_name = next(name for name, result in filter_results.items() if not result.passed)
        failed_result = filter_results[failed_filter_name]
        
        # Create filter-specific fail_cases directory
        filter_fail_dir = output_dir / "fail_cases" / failed_filter_name
        filter_fail_dir.mkdir(parents=True, exist_ok=True)
        
        # Load image and mask
        image = sample.load_image()  # RGB [H, W, 3]
        mask = sample.load_mask(binary=False)  # Grayscale [H, W]
        
        # Convert mask to 3-channel for stacking
        mask_rgb = np.stack([mask, mask, mask], axis=-1)
        
        # Create overlay (image with red mask overlay)
        overlay = image.copy()
        mask_norm = mask.astype(np.float32) / 255.0
        red_overlay = np.zeros_like(image)
        red_overlay[:, :, 0] = mask_norm * 255  # Red channel
        overlay = (overlay * 0.7 + red_overlay * 0.3).astype(np.uint8)
        
        # Start with base stack: [image | mask | overlay]
        panels = [image, mask_rgb, overlay]
        
        # Append prediction visualizations if available
        if (failed_result.metadata and 
            'prediction_visualizations' in failed_result.metadata):
            
            pred_viz = failed_result.metadata['prediction_visualizations']
            
            for pred_name, pred_mask in pred_viz.items():
                # Convert prediction to 3-channel
                pred_mask_norm = (pred_mask * 255).astype(np.uint8)
                pred_rgb = np.stack([pred_mask_norm, pred_mask_norm, pred_mask_norm], axis=-1)
                panels.append(pred_rgb)
        
        # Stack all panels horizontally
        stacked = np.hstack(panels)
        
        # Add text header if available
        if (failed_result.metadata and 
            'problem_description' in failed_result.metadata):
            
            text = failed_result.metadata['problem_description']
            stacked = self._add_text_header(stacked, text)
        
        # Save stacked visualization
        flat_filename = f"{sample.class_name}_{sample.sample_id}.jpg"
        output_path = filter_fail_dir / flat_filename
        
        stacked_image = Image.fromarray(stacked)
        stacked_image.save(output_path, quality=95)
    
    def _add_text_header(self, image: np.ndarray, text: str) -> np.ndarray:
        """Add text header to image."""
        from PIL import ImageDraw, ImageFont
        
        # Convert to PIL for text drawing
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)
        
        # Try to use a decent font, fallback to default
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Calculate text dimensions and position
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Create new image with space for text header
        header_height = text_height + 20  # 10px padding top/bottom
        new_height = image.shape[0] + header_height
        new_width = image.shape[1]
        
        # Create white header background
        header_image = np.ones((header_height, new_width, 3), dtype=np.uint8) * 255
        
        # Combine header and original image
        final_image = np.vstack([header_image, image])
        
        # Draw text on the combined image
        final_pil = Image.fromarray(final_image)
        final_draw = ImageDraw.Draw(final_pil)
        
        # Center text horizontally, position vertically in header
        text_x = (new_width - text_width) // 2
        text_y = 10  # 10px from top
        
        final_draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)
        
        return np.array(final_pil)
    
    def print_statistics(self):
        """Print detailed filtering statistics."""
        print("\n" + "="*60)
        print("DATASET FILTERING STATISTICS")
        print("="*60)
        
        print(f"Overall Results:")
        print(f"  Total Samples: {self.global_stats['total_samples']}")
        print(f"  Passed: {self.global_stats['passed_samples']}")
        print(f"  Failed: {self.global_stats['failed_samples']}")
        print(f"  Pass Rate: {self.global_stats.get('overall_pass_rate', 0.0):.2%}")
        
        if self.global_stats.get('save_fail_cases', False):
            print(f"  Failed samples saved to: fail_cases/ directory")
        
        print(f"\nPer-Filter Breakdown:")
        for filter_name, stats in self.global_stats.get('filter_breakdown', {}).items():
            pass_rate = stats['passed'] / stats['total_processed'] if stats['total_processed'] > 0 else 0.0
            print(f"  {filter_name}: {pass_rate:.2%} pass rate ({stats['passed']}/{stats['total_processed']})")
        
        print(f"\nPer-Class Results:")
        for class_name, stats in self.global_stats.get('class_stats', {}).items():
            print(f"  {class_name}: {stats['pass_rate']:.2%} ({stats['passed']}/{stats['processed']}) | Failed: {stats.get('failed', 0)}")


def calculate_iou(mask1: np.ndarray, mask2: np.ndarray, threshold: float = 0.5) -> float:
    """Calculate IoU between two masks with robust binarization."""
    # Ensure masks are 2D
    if mask1.ndim > 2:
        mask1 = mask1.squeeze()
    if mask2.ndim > 2:
        mask2 = mask2.squeeze()
    
    # Robust binarization - handle already binary masks
    if mask1.dtype == bool:
        binary1 = mask1
    else:
        binary1 = mask1 > threshold
        
    if mask2.dtype == bool:
        binary2 = mask2  
    else:
        binary2 = mask2 > threshold
    
    # Calculate intersection and union using boolean operations
    intersection = np.logical_and(binary1, binary2).sum()
    union = np.logical_or(binary1, binary2).sum()
    
    # Handle edge case (both masks empty)
    if union == 0:
        return 1.0  # Perfect match when both are empty
    
    return float(intersection / union)
