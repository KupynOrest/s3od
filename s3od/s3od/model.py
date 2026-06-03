from typing import List, Optional
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor, AutoConfig


class BaseDPTSegmentation(nn.Module):
    """Base DPT segmentation model with DINO-v3 encoder."""
    
    def __init__(self,
                 num_classes,
                 num_outputs: int = 3,
                 encoder_name: str = 'dinov3_base',
                 features: int = 256,
                 out_channels: Optional[List[int]] = None,
                 use_bn: bool = True,
                 use_clstoken: bool = False):
        
        super().__init__()
        
        self.patch_size = 16
        self.encoder_name = encoder_name
        
        config_dir = os.path.join(os.path.dirname(__file__), 'dinov3_config')
        config = AutoConfig.from_pretrained(config_dir, local_files_only=True)
        self.encoder = AutoModel.from_config(config)
        self.processor = AutoImageProcessor.from_pretrained(config_dir, local_files_only=True)
        
        self.intermediate_layer_idx = {
            'dinov3_base': [2, 5, 8, 11],
            'dinov3_small': [2, 5, 8, 11], 
            'dinov3_large': [4, 11, 17, 23],
        }
        
        dim = self.encoder.config.hidden_size
        self.n_register_tokens = getattr(self.encoder.config, 'num_register_tokens', 4)
        
        out_channels = out_channels or [256, 512, 1024, 1024]
        
        self.seg_head = self._create_segmentation_head(
            nclass=num_classes,
            num_outputs=num_outputs,
            in_channels=dim,
            features=features,
            use_bn=use_bn,
            out_channels=out_channels,
            use_clstoken=use_clstoken,
            patch_size=self.patch_size
        )
    
    def _create_segmentation_head(self, **kwargs):
        """Override in subclasses to create specific segmentation heads."""
        raise NotImplementedError
    
    def extract_intermediate_features(self, x):
        """Extract intermediate features from DINO-v3 encoder using standard output_hidden_states."""
        layer_indices = self.intermediate_layer_idx.get(
            self.encoder_name, 
            [2, 5, 8, 11]
        )
        
        outputs = self.encoder(
            pixel_values=x,
            output_hidden_states=True,
            return_dict=True
        )
        
        hidden_states = outputs.hidden_states
        
        features = []
        for layer_idx in layer_indices:
            hidden_state = hidden_states[layer_idx]
            
            if self.seg_head.use_clstoken:
                cls_token = hidden_state[:, 0]
                patch_tokens = hidden_state[:, 1 + self.n_register_tokens:]
                features.append((patch_tokens, cls_token))
            else:
                patch_tokens = hidden_state[:, 1 + self.n_register_tokens:]
                features.append((patch_tokens,))
        
        return features


class DPTSegmentation(BaseDPTSegmentation):
    """Standard DPT segmentation model without FLUX features."""
    
    def __init__(self, **kwargs):
        self.use_flux_features = False
        super().__init__(**kwargs)
    
    def _create_segmentation_head(self, **kwargs):
        return DPTSegmentationHead(**kwargs)

    def forward(self, x):
        """Standard forward pass without FLUX features."""
        h, w = x.shape[-2:]
        features = self.extract_intermediate_features(x)
        patch_h, patch_w = h // 16, w // 16
        
        outputs = self.seg_head(features, patch_h, patch_w)
        return outputs



class BaseDPTSegmentationHead(nn.Module):
    """Base DPT segmentation head."""
    
    def __init__(self, nclass, in_channels, num_outputs: int = 3, features=256, use_bn=False,
                 out_channels=[256, 512, 1024, 1024], use_clstoken=False, patch_size=16):
        super().__init__()
        
        self.nclass = nclass
        self.patch_size = patch_size
        self.use_clstoken = use_clstoken

        self._init_projections(in_channels, out_channels)
        self._init_resize_layers(out_channels)
        self._init_readout_projections(in_channels, use_clstoken)
        self._init_dpt_components(out_channels, features, use_bn)
        self._init_output_heads(features, num_outputs)
    
    def _init_projections(self, in_channels, out_channels):
        """Initialize feature projection layers."""
        self.projects = nn.ModuleList([
            nn.Conv2d(in_channels, out_channel, kernel_size=1, stride=1, padding=0)
            for out_channel in out_channels
        ])
    
    def _init_resize_layers(self, out_channels):
        """Initialize resize layers for different scales."""
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1)
        ])
    
    def _init_readout_projections(self, in_channels, use_clstoken):
        """Initialize readout projections for class tokens."""
        if use_clstoken:
            self.readout_projects = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(2 * in_channels, in_channels),
                    nn.GELU()
                ) for _ in range(4)  # 4 layers
            ])
        else:
            self.readout_projects = None
    
    def _init_dpt_components(self, out_channels, features, use_bn):
        """Initialize DPT-specific components."""
        self.scratch = _make_scratch(out_channels, features, groups=1, expand=False)
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
    
    def _init_output_heads(self, features, num_outputs):
        """Initialize output heads for masks and IoU."""
        self.mask_head = MultiMaskHead(features, num_outputs, inter_features=32)
        self.classifier_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(features, 64),
            nn.ReLU(True),
            nn.Linear(64, num_outputs)
        )
    
    def process_encoder_features(self, out_features, patch_h, patch_w):
        """Process features from DINO-v3 encoder."""
        processed_features = []
        
        for i, feature_tuple in enumerate(out_features):
            if self.use_clstoken:
                patch_tokens, cls_token = feature_tuple[0], feature_tuple[1]
                readout = cls_token.unsqueeze(1).expand_as(patch_tokens)
                x = self.readout_projects[i](torch.cat((patch_tokens, readout), -1))
            else:
                patch_tokens = feature_tuple[0]
                x = patch_tokens

            x = x.permute(0, 2, 1).reshape(x.shape[0], x.shape[-1], patch_h, patch_w)
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            processed_features.append(x)
        
        return processed_features


class DPTSegmentationHead(BaseDPTSegmentationHead):
    """Standard DPT segmentation head without FLUX features."""
    
    def forward(self, out_features, patch_h, patch_w):
        """Standard forward pass without FLUX features."""
        processed_features = self.process_encoder_features(out_features, patch_h, patch_w)
        
        layer_1_rn = self.scratch.layer1_rn(processed_features[0])
        layer_2_rn = self.scratch.layer2_rn(processed_features[1])
        layer_3_rn = self.scratch.layer3_rn(processed_features[2])
        layer_4_rn = self.scratch.layer4_rn(processed_features[3])
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        aux_out = self.classifier_head(path_1)
        pred_masks = self.mask_head(path_1, (int(patch_h * self.patch_size), int(patch_w * self.patch_size)))

        return {
            'pred_masks': pred_masks,
            'pred_iou': aux_out,
            'features': path_1
        }

# Utility classes and functions from original DPT implementation

def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    """Create scratch module for DPT."""
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    out_shape4 = out_shape if len(in_shape) >= 4 else None

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        out_shape4 = out_shape * 8 if len(in_shape) >= 4 else None

    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)

    return scratch


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn):
        super().__init__()
        self.bn = bn
        self.groups = 1

        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        if self.bn:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn:
            out = self.bn2(out)

        return out + x


class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=False, size=None):
        super().__init__()

        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = 1
        self.expand = expand
        self.size = size
        
        out_features = features // 2 if expand else features

        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)
        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)

    def forward(self, *xs, size=None):
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = output + res

        output = self.resConfUnit2(output)

        # Determine interpolation parameters
        if size is not None:
            modifier = {"size": size}
        elif self.size is not None:
            modifier = {"size": self.size}
        else:
            modifier = {"scale_factor": 2}

        output = F.interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)

        return output


def _make_fusion_block(features, use_bn, size=None):
    """Create fusion block."""
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=False,
        size=size,
    )


class MultiMaskHead(nn.Module):
    """Multi-mask prediction head."""
    
    def __init__(self, in_features: int, n_masks: int, inter_features: int = 32):
        super().__init__()

        self.output_conv1 = nn.Conv2d(in_features, in_features // 2, kernel_size=3, stride=1, padding=1)
        self.upsample_2x = nn.Sequential(
            nn.ConvTranspose2d(in_features // 2, inter_features * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_features * 2, inter_features * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.mask_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(inter_features * 2, inter_features, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_features, 1, kernel_size=1, stride=1, padding=0),
            ) for _ in range(n_masks)
        ])

    def forward(self, x, target_size):
        feat = self.output_conv1(x)
        feat = self.upsample_2x(feat)
        feat = F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False, antialias=True)

        masks = []
        for head in self.mask_heads:
            mask = head(feat)
            masks.append(mask)

        return torch.cat(masks, dim=1)
