from typing import Dict, List, Tuple

import torch
import hydra
import numpy as np
import pytorch_lightning as pl
import albumentations as A
from torchmetrics.segmentation import DiceScore
from torchmetrics.classification import BinaryJaccardIndex
import matplotlib.pyplot as plt
import torch.nn.functional as F

from model_training.loss import LossModule


class ImageLogger:
    def __init__(self):
        self.inv_normalize = A.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], 
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        )

    def denormalize_image(self, image: torch.Tensor) -> torch.Tensor:
        image = np.transpose(image, (1, 2, 0))[:, :, :3]
        image = self.inv_normalize(image=image * 255)["image"] * 255 if image.max() > 1.0 else image * 255.0
        return image.astype("uint8")

    @staticmethod
    def mask_to_rgb(mask: np.ndarray, is_best: bool = False) -> np.ndarray:
        from scipy.ndimage import binary_dilation
        mask_rgb = (mask * 255).astype(np.uint8)
        rgb = np.stack([mask_rgb] * 3, axis=2)
        if is_best:
            kernel = np.ones((3, 3))
            dilated = binary_dilation(mask > 0.5, kernel)
            border = dilated & ~(mask > 0.5)
            rgb[border] = [0, 255, 0]
        return rgb

    @staticmethod
    def concept_map_to_rgb(concept_map: np.ndarray, target_size: Tuple[int, int] = None) -> np.ndarray:
        if target_size is not None and concept_map.shape != target_size:
            concept_tensor = torch.from_numpy(concept_map).unsqueeze(0).unsqueeze(0).float()
            resized_tensor = F.interpolate(concept_tensor, size=target_size, mode='bilinear', align_corners=False, antialias=True)
            concept_map = resized_tensor.squeeze(0).squeeze(0).numpy()
        
        vmin, vmax = concept_map.min(), concept_map.max()
        normalized = (concept_map - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(concept_map)
        colored = plt.get_cmap("plasma")(normalized)
        return (colored[:, :, :3] * 255).astype(np.uint8)

    def create_visualization(self, image: torch.Tensor, predictions: torch.Tensor, 
                           pred_ious: torch.Tensor, target: torch.Tensor, 
                           concept_maps: Dict[str, torch.Tensor] = None) -> np.ndarray:
        image_np = self.denormalize_image(image.float().cpu().numpy())
        predictions_np = predictions.float().detach().cpu().numpy()
        target_np = target.float().cpu().numpy()
        
        if predictions_np.ndim == 2:
            predictions_np = predictions_np[None, :]
            best_idx = 0
        else:
            best_idx = pred_ious.argmax().item()

        image_height, image_width = image_np.shape[:2]
        target_size = (image_height, image_width)
        viz_components = [image_np]
        
        if concept_maps is not None:
            if 'category' in concept_maps:
                category_map = concept_maps['category'].float().cpu().numpy()
                category_viz = ImageLogger.concept_map_to_rgb(category_map, target_size)
                viz_components.append(category_viz)
            
            if 'background' in concept_maps:
                background_map = concept_maps['background'].float().cpu().numpy()
                background_viz = ImageLogger.concept_map_to_rgb(background_map, target_size)
                viz_components.append(background_viz)
        
        mask_viz = [ImageLogger.mask_to_rgb(pred, idx == best_idx) 
                   for idx, pred in enumerate(predictions_np)]
        viz_components.extend(mask_viz)
        viz_components.append(ImageLogger.mask_to_rgb(target_np))
        
        return np.hstack(viz_components)

    def get_images(self, images: torch.Tensor, predictions: Dict[str, torch.Tensor], 
                  targets: torch.Tensor, concept_maps: Dict[str, torch.Tensor] = None) -> List[np.ndarray]:
        pred_masks = torch.sigmoid(predictions['pred_masks'])
        pred_ious = predictions['pred_iou'].squeeze(-1)

        use_concept_maps = (
            concept_maps is not None and 
            'category' in concept_maps and 
            'background' in concept_maps and
            concept_maps['category'].shape[0] == images.shape[0] and
            concept_maps['background'].shape[0] == images.shape[0]
        )

        return [
            self.create_visualization(
                images[idx], pred_masks[idx], pred_ious[idx], targets[idx],
                {k: v[idx] for k, v in concept_maps.items()} if use_concept_maps else None
            ) for idx in range(len(images))
        ]


class SegmentationLightningModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.images = []
        self.max_images = config.train_stage.max_images
        self.enable_image_logging = config.train_stage.get('enable_image_logging', False)
        self.save_hyperparameters()
        self.config = config
        self.model = hydra.utils.instantiate(config.model)

        self.loss_module = LossModule(
            config.loss.criterions,
            full_mask_lambda=config.loss.get('full_mask_lambda', 0.01),
            decay_rate=config.loss.get('decay_rate', 0.2),
        )
        
        self.iou = BinaryJaccardIndex()
        self.dice = DiceScore(num_classes=1, average='micro')
        self.image_logger = ImageLogger()

    def on_train_epoch_end(self):
        self.images.clear()
        self.iou.reset()
        self.dice.reset()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_validation_epoch_end(self):
        self.images.clear()
        self.iou.reset()
        self.dice.reset()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def configure_optimizers(self):
        param_groups = [
            {"params": self.model.encoder.parameters(), "lr": self.config.optimizer.lr},
            {"params": self.model.seg_head.parameters(), "lr": self.config.optimizer.lr * 10}
        ]
        optimizer = torch.optim.AdamW(params=param_groups, weight_decay=0.05, betas=[0.9, 0.999], eps=1e-8)
        
        if 'schedulers' in self.config.scheduler.keys():
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                schedulers=[hydra.utils.instantiate(scheduler, optimizer=optimizer) 
                           for scheduler in self.config.scheduler.schedulers],
                milestones=self.config.scheduler.milestones,
                optimizer=optimizer,
            )
        else:
            scheduler = hydra.utils.instantiate(self.config.scheduler, optimizer=optimizer)
            
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, 'val')

    def calculate_metrics(self, predictions: Dict[str, torch.Tensor], targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        pred_masks = torch.sigmoid(predictions['pred_masks'])
        pred_ious = predictions['pred_iou'].squeeze(-1)
        
        if pred_masks.size(1) == 1:
            best_masks = pred_masks.squeeze(1)
        else:
            best_indices = pred_ious.argmax(dim=1)
            best_masks = pred_masks[torch.arange(pred_masks.size(0)), best_indices]
        
        return {
            'iou': self.iou(best_masks, (targets > 0.5).int()),
            'dice': self.dice(best_masks, (targets > 0.5).int())
        }

    def _step(self, batch, batch_idx, split: str = 'train'):
        images, targets = batch['images'], batch['masks']
        transformer_features = batch.get('transformer_features', None)
        concept_maps = batch.get('concept_maps', None)
        
        if hasattr(self.model, 'use_flux_features') and self.model.use_flux_features:
            predictions = self.model(images, transformer_features, concept_maps)
        else:
            predictions = self.model(images)

        loss, loss_parts = self.loss_module(predictions, batch, self.current_epoch)

        for name, value in loss_parts.items():
            self.log(f'{split}_{name}', value, on_step=True, on_epoch=True, sync_dist=True)

        metrics = self.calculate_metrics(predictions, targets)
        self.log(f'{split}_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        for metric_name, metric_value in metrics.items():
            self.log(f'{split}_{metric_name}', metric_value, on_step=True, on_epoch=True, sync_dist=True)

        if self.enable_image_logging and len(self.images) < self.max_images:
            viz = self.image_logger.get_images(images, predictions, targets, concept_maps)
            for idx, image_viz in enumerate(viz):
                if len(self.images) >= self.max_images:
                    break
                self.images.append(image_viz)
            
            if len(self.images) >= self.max_images:
                for idx, image_viz in enumerate(self.images):
                    image_tag = f'{split}_images/epoch_{self.current_epoch}_img_{idx}'
                    self.logger.experiment.add_image(image_tag, image_viz, self.current_epoch, dataformats='HWC')
        
        return loss
