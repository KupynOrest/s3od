import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra


@dataclass
class LossComponent:
    name: str
    weight: float
    target_key: str
    output_key: str
    loss: nn.Module
    add_sigmoid: bool = True

    def __post_init__(self):
        assert self.weight >= 0.0, "Weight must be non-negative"

    @classmethod
    def from_dict(cls, loss_config: Dict[str, Any]) -> 'LossComponent':
        return cls(
            name=loss_config['name'],
            weight=loss_config['weight'],
            target_key=loss_config['target_key'],
            output_key=loss_config['output_key'],
            loss=hydra.utils.instantiate(loss_config['loss'])
        )


class SSIMLoss(nn.Module):
    def __init__(self, window_size: int = 11, reduction: str = 'mean'):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.reduction = reduction
        self.channel = 1
        self.window = self._create_window(window_size)

    def _create_window(self, window_size):
        def gaussian(window_size: int, sigma: float = 1.5):
            gauss = torch.exp(torch.tensor([-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)
                                            for x in range(window_size)]))
            return gauss / gauss.sum()

        _1D_window = gaussian(window_size).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
        return _2D_window

    def forward(self, img1, img2):
        if img1.device != self.window.device:
            self.window = self.window.to(img1.device)

        mu1 = F.conv2d(img1, self.window, padding=self.window_size // 2)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size // 2)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.window, padding=self.window_size // 2) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.window, padding=self.window_size // 2) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size // 2) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if self.reduction == 'mean':
            return 1 - ssim_map.mean()
        elif self.reduction == 'none':
            return 1 - ssim_map.mean((1, 2, 3))


class IoULoss(nn.Module):
    def __init__(self, smooth: float = 1e-6, reduction: str = 'mean'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
        assert reduction in ['mean', 'sum', 'none'], "Invalid reduction"

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)

        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1) - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        loss = 1 - iou

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6, reduction: str = 'mean'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
        assert reduction in ['mean', 'sum', 'none'], "Invalid reduction"

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)

        intersection = (pred * target).sum(dim=1)
        dice = (2. * intersection + self.smooth) / (
                pred.sum(dim=1) + target.sum(dim=1) + self.smooth
        )
        loss = 1 - dice

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        assert reduction in ['mean', 'sum', 'none'], "Invalid reduction"

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class MaskLossHandler:
    def __init__(self, components: List[LossComponent], full_mask_lambda: float = 0.01, decay_rate: float = 0.2):
        self.mask_components = [c for c in components
                                if c.target_key == "masks" and c.output_key == "pred_masks"]
        self.aux_components = [c for c in components
                               if c.target_key != "masks" or c.output_key != "pred_masks"]
        self.full_mask_lambda = full_mask_lambda
        self.decay_rate = decay_rate

    @staticmethod
    def compute_iou(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
        def op_sum(x: torch.Tensor) -> torch.Tensor:
            return x.view(x.shape[0], x.shape[1], -1).sum(2)

        intersection = op_sum(target * pred)
        union = op_sum(target ** 2) + op_sum(pred ** 2) - intersection
        iou = (intersection + smooth) / (union + smooth)
        iou = torch.mean(iou, dim=1)
        return iou

    def compute_single_mask_loss(
            self,
            pred_masks: torch.Tensor,
            target_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Simple loss computation for single mask prediction - only segmentation losses."""
        # Remove mask dimension: [batch, 1, H, W] -> [batch, H, W]
        pred_masks = pred_masks.squeeze(1)
        
        total_loss = torch.tensor(0.0, device=pred_masks.device)
        loss_dict = {}

        for comp in self.mask_components:
            pred = torch.sigmoid(pred_masks) if comp.add_sigmoid else pred_masks
            component_loss = comp.loss(pred, target_masks)
            
            if component_loss.dim() > 0:
                component_loss = component_loss.mean()
            
            total_loss += comp.weight * component_loss
            loss_dict[comp.name] = component_loss

        return total_loss, loss_dict

    def compute_multi_mask_losses(
            self,
            pred_masks: torch.Tensor,
            target_masks: torch.Tensor,
            epoch: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Original multi-mask loss computation with best mask selection + decay."""
        batch_size, num_masks = pred_masks.shape[:2]
        target_expanded = target_masks.unsqueeze(1).expand(-1, num_masks, -1, -1)
        exp_decay = self.full_mask_lambda * math.exp(-self.decay_rate * epoch)

        # Compute IoU once for mask selection
        pred_sigmoid = torch.sigmoid(pred_masks)
        pred_masks_flat = pred_sigmoid.contiguous().reshape(batch_size * num_masks, 1, *pred_masks.shape[2:])
        gt_masks_flat = target_expanded.contiguous().reshape(batch_size * num_masks, 1, *target_masks.shape[1:])
        with torch.no_grad():
            ious = self.compute_iou(
                pred_masks_flat,
                gt_masks_flat,
            ).reshape(batch_size, num_masks)
        best_indices = ious.argmax(dim=1)

        total_loss = torch.tensor(0.0, device=pred_masks.device)
        loss_dict = {'best_iou': ious.max(dim=1)[0].mean(), 'gt_ious': ious}

        for comp in self.mask_components:
            pred = pred_sigmoid if comp.add_sigmoid else pred_masks
            pred_flat = pred.reshape(batch_size * num_masks, 1, *pred.shape[2:])
            target_flat = target_expanded.reshape(batch_size * num_masks, 1, *target_expanded.shape[2:])

            all_losses = comp.loss(pred_flat, target_flat)
            if all_losses.dim() == 4:
                all_losses = all_losses.mean(dim=(1, 2, 3))
            all_losses = all_losses.reshape(batch_size, num_masks)
            best_loss = all_losses.gather(1, best_indices.unsqueeze(1)).mean()

            component_loss = best_loss + all_losses.mean() * exp_decay
            total_loss += comp.weight * component_loss
            loss_dict.update({
                f"{comp.name}_best": best_loss,
                f"{comp.name}_full": all_losses,
            })

        return total_loss, loss_dict


class LossModule(nn.Module):
    def __init__(self, loss_config: List[Dict[str, Any]], full_mask_lambda: float = 0.01, decay_rate: float = 0.2):
        super().__init__()
        self.components = [LossComponent.from_dict(conf) for conf in loss_config]
        self.mask_handler = MaskLossHandler(self.components, full_mask_lambda, decay_rate)

    def forward(
            self,
            outputs: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor],
            epoch: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        pred_masks = outputs["pred_masks"]
        num_masks = pred_masks.size(1)
        
        if num_masks == 1:
            mask_loss, loss_dict = self.mask_handler.compute_single_mask_loss(
                pred_masks,
                targets["masks"]
            )
            return mask_loss, loss_dict
        
        mask_loss, loss_dict = self.mask_handler.compute_multi_mask_losses(
            pred_masks,
            targets["masks"],
            epoch
        )
        targets = {**targets, **loss_dict}

        for comp in self.mask_handler.aux_components:
            output = outputs[comp.output_key]
            target = targets[comp.target_key]
            if comp.add_sigmoid:
                output = torch.sigmoid(output)
            aux_loss = comp.loss(output, target)
            mask_loss += comp.weight * aux_loss
            loss_dict[comp.name] = aux_loss

        return mask_loss, {k: v.mean() if torch.is_tensor(v) and v.dim() > 0 else v 
                          for k, v in loss_dict.items()}
