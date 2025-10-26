import time
import os
import random
import gc
import psutil
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import albumentations as A
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from model_training.transforms import get_transforms
from data_generation.resizer import FluxResizer


def log_memory_usage(stage: str):
    process = psutil.Process(os.getpid())
    ram_gb = process.memory_info().rss / 1024**3
    
    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated() / 1024**3
        gpu_reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[{stage}] RAM: {ram_gb:.2f}GB, GPU Allocated: {gpu_allocated:.2f}GB, GPU Reserved: {gpu_reserved:.2f}GB")
    else:
        print(f"[{stage}] RAM: {ram_gb:.2f}GB")


class MaskDataset(Dataset):
    def __init__(
            self,
            root_dir: str,
            image_size: int,
            split: str = 'train',
            val_split: float = 0.1,
            transform_mode: str = 'regular',
            seed: int = 42,
            debug_subset_fraction: Optional[float] = None,
    ):
        self.root_dir = root_dir
        self.image_size = image_size
        self.split = split
        self.transform = get_transforms(image_size, transform_mode)
        self.debug_subset_fraction = debug_subset_fraction

        self.images_dir = os.path.join(root_dir, 'images')
        self.masks_dir = os.path.join(root_dir, 'masks')

        valid_extensions = {'.jpg', '.jpeg', '.png'}
        all_files = [f for f in os.listdir(self.images_dir)
                     if any(f.lower().endswith(ext) for ext in valid_extensions)]

        split_files = self._get_splits(val_split, seed)
        if split == 'train':
            self.files = split_files[0]
        else:  # val
            self.files = split_files[1]
        
        # Reduce dataset size for efficient debugging if fraction is specified
        if self.debug_subset_fraction is not None:
            num_samples = int(len(self.files) * self.debug_subset_fraction)
            self.files = self.files[:num_samples]
            print(f"🔥 DEBUG MODE: Using {num_samples} ({self.debug_subset_fraction:.0%}) of {self.split} files.")

    def _get_splits(self, val_split: float, seed: int = 42) -> List[str]:
        valid_extensions = {'.jpg', '.jpeg', '.png'}
        all_files = [f for f in os.listdir(self.images_dir)
                     if any(f.lower().endswith(ext) for ext in valid_extensions)]

        valid_files = []
        for img_file in all_files:
            mask_path = self.get_mask_path(img_file)
            if os.path.exists(mask_path):
                valid_files.append(img_file)

        valid_files.sort()

        random.seed(seed)
        random.shuffle(valid_files)

        n_val = int(len(valid_files) * val_split)
        val_files = valid_files[:n_val]
        train_files = valid_files[n_val:]
        
        return train_files, val_files

    def get_mask_path(self, img_file: str) -> str:
        base_name = os.path.splitext(img_file)[0]
        possible_extensions = ['.png', '.jpg', '.jpeg']
        
        for ext in possible_extensions:
            mask_path = os.path.join(self.masks_dir, base_name + ext)
            if os.path.exists(mask_path):
                return mask_path
        
        return os.path.join(self.masks_dir, base_name + '.png')

    def __len__(self) -> int:
        return len(self.files)

    def _array_to_batch(self, x: np.array) -> torch.Tensor:
        if x.ndim == 2:
            return torch.from_numpy(x).float()
        return torch.from_numpy(x).permute(2, 0, 1).float()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        try:
            img_path = os.path.join(self.images_dir, self.files[idx])
            mask_path = self.get_mask_path(self.files[idx])

            image = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')

            # Check if shapes match (after converting to numpy arrays)
            img_array = np.array(image)
            mask_array = np.array(mask)
            if img_array.shape[:2] != mask_array.shape[:2]:
                return self.__getitem__(random.randint(0, len(self) - 1))

            # Apply transforms
            augmented = self.transform(image=img_array, mask=mask_array)
            image, mask = augmented['image'], augmented['mask']

            return {"images": self._array_to_batch(image), "masks": self._array_to_batch(mask / 255.0)}

        except Exception as e:
            logging.error(f"Error loading {self.files[idx]}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))


class FluxMaskDataset(MaskDataset):

    def __init__(
            self,
            root_dir: str,
            image_size: int,
            split: str = 'train',
            val_split: float = 0.1,
            transform_mode: str = 'regular',
            seed: int = 42,
            flux_features_dir: Optional[str] = None,
            feature_layers: List[int] = [0, 1, 2, 3],
            debug_subset_fraction: Optional[float] = None,
    ):
        super().__init__(
            root_dir, image_size, split, val_split, transform_mode, seed,
            debug_subset_fraction=debug_subset_fraction
        )
        
        self.flux_features_dir = Path(flux_features_dir) if flux_features_dir else None
        self.feature_layers = feature_layers
        self._error_count = 0
        self._sample_times = []
        
        self.flux_resizer = FluxResizer()
        self.transform = A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self._create_feature_mapping()
        
        print(f"FluxMaskDataset: {len(self.files)} images with {len(self.feature_mapping)} available features")
        log_memory_usage("Dataset initialized")

    def _create_feature_mapping(self):
        self.feature_mapping = {}
        if self.flux_features_dir is None:
            print(f"Warning: Flux features directory not provided. Cannot create feature mapping.")
            return

        features_dir = self.flux_features_dir / "features"
        
        if not features_dir.exists():
            print(f"Warning: Features directory {features_dir} does not exist")
            return
        
        available_npz = {f.stem: f for f in features_dir.glob("*.npz")}
        
        for img_file in self.files:
            base_name = Path(img_file).stem
            
            if base_name in available_npz:
                self.feature_mapping[img_file] = available_npz[base_name]
                continue
            
            for dataset_prefix in ['DUTS-TR', 'DIS-TR', 'HRSOD-TR', 'UHRSD-TR']:
                candidate_key = f"{dataset_prefix}_{base_name}"
                if candidate_key in available_npz:
                    self.feature_mapping[img_file] = available_npz[candidate_key]
                    break
        
        original_count = len(self.files)
        self.files = [f for f in self.files if f in self.feature_mapping]
        
        print(f"Filtered {original_count} → {len(self.files)} files with FLUX features "
              f"({len(self.files)/original_count*100:.1f}% coverage)")

    def _load_flux_features(self, img_file: str) -> Dict[str, torch.Tensor]:
        npz_path = self.feature_mapping[img_file]
        
        with np.load(npz_path, mmap_mode='r') as data:
            concept_maps = {}
            if 'category' in data:
                concept_maps['category'] = torch.from_numpy(data['category'].copy()).float()
            if 'background' in data:
                concept_maps['background'] = torch.from_numpy(data['background'].copy()).float()
            
            transformer_features = []
            for layer_idx in self.feature_layers:
                layer_key = f'layer_{layer_idx}'
                if layer_key in data:
                    layer_data = data[layer_key].copy()
                    if layer_data.dtype == np.float16:
                        layer_data = layer_data.astype(np.float32)
                    transformer_features.append(torch.from_numpy(layer_data))
        
        gc.collect()
        return {'concept_maps': concept_maps, 'transformer_features': transformer_features}
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start_time = time.time()
        try:
            img_file = self.files[idx]
            img_path = os.path.join(self.images_dir, img_file)
            mask_path = self.get_mask_path(img_file)

            with Image.open(img_path) as image_pil:
                image_pil = image_pil.convert('RGB')
                with Image.open(mask_path) as mask_pil:
                    mask_pil = mask_pil.convert('L')

                    if image_pil.size != mask_pil.size:
                        return self.__getitem__(random.randint(0, len(self) - 1))

                    image_resized, target_size = self.flux_resizer.resize_pil_image(image_pil)
                    mask_resized = self.flux_resizer.resize_mask(np.array(mask_pil), target_size)

            image_np = np.array(image_resized)
            mask_np = mask_resized
            
            del image_resized, mask_resized
            
            augmented = self.transform(image=image_np, mask=mask_np)
            image_transformed, mask_transformed = augmented['image'], augmented['mask']
            
            del image_np, mask_np, augmented

            flux_features = self._load_flux_features(img_file)

            result = {
                'images': self._array_to_batch(image_transformed),
                'masks': self._array_to_batch(mask_transformed / 255.0),
                'transformer_features': flux_features['transformer_features'],
                'concept_maps': flux_features['concept_maps'],
            }
            
            del image_transformed, mask_transformed, flux_features
            
            load_time = time.time() - start_time
            self._sample_times.append(load_time)
            
            if idx % 100 == 0:
                gc.collect()
                if idx % 500 == 0:
                    avg_time = np.mean(self._sample_times[-500:]) if self._sample_times else 0
                    log_memory_usage(f"Dataset sample {idx}")
                    print(f"Average loading time (last 500 samples): {avg_time:.3f}s")
                    
                if len(self._sample_times) > 500:
                    self._sample_times = self._sample_times[-250:]
            
            return result

        except Exception as e:
            print(f"Error loading {img_file}: {e}")
            if hasattr(self, '_error_count'):
                self._error_count += 1
                if self._error_count > 10:
                    raise RuntimeError(f"Too many consecutive errors in dataset loading: {e}")
            else:
                self._error_count = 1
            return self.__getitem__(random.randint(0, len(self) - 1))


def create_dataloaders(
        dataset_paths: List[str],
        image_size: int,
        train_batch_size: int,
        val_batch_size: int,
        num_workers: int,
        val_split: float = 0.1,
        seed: int = 42,
        transform_mode: str = 'regular',
        flux_features_dir: Optional[str] = None,
        feature_layers: List[int] = [0, 1, 2, 3],
        debug_subset_fraction: Optional[float] = None,
) -> Tuple[DataLoader, DataLoader]:
    def worker_init_fn(worker_id):
        import gc
        import torch
        import random
        import numpy as np
        
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    if flux_features_dir:
        dataset_class = FluxMaskDataset
        dataset_kwargs = {
            'flux_features_dir': flux_features_dir,
            'feature_layers': feature_layers,
            'debug_subset_fraction': debug_subset_fraction,
        }
        actual_train_batch_size = 1
        actual_val_batch_size = 1
    else:
        dataset_class = MaskDataset
        dataset_kwargs = {
            'debug_subset_fraction': debug_subset_fraction,
        }
        actual_train_batch_size = train_batch_size
        actual_val_batch_size = val_batch_size

    train_datasets = []
    val_datasets = []
    
    for dataset_path in dataset_paths:
        # Train dataset
        train_dataset = dataset_class(
            root_dir=dataset_path,
            image_size=image_size,
            split='train',
            val_split=val_split,
            transform_mode=transform_mode,
            seed=seed,
            **dataset_kwargs
        )
        train_datasets.append(train_dataset)

        val_dataset = dataset_class(
            root_dir=dataset_path,
            image_size=image_size,
            split='val',
            val_split=val_split,
            transform_mode='test',
            seed=seed,
            **dataset_kwargs
        )
        val_datasets.append(val_dataset)

    if len(train_datasets) > 1:
        train_dataset = ConcatDataset(train_datasets)
        val_dataset = ConcatDataset(val_datasets)
    else:
        train_dataset = train_datasets[0]
        val_dataset = val_datasets[0]

    train_loader = DataLoader(
        train_dataset,
        batch_size=actual_train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=actual_val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_fn,
        persistent_workers=False
    )

    return train_loader, val_loader
