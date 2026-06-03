import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor
from typing import List, Optional


class BaseDPTSegmentation(nn.Module):
    """Base DPT segmentation model with DINO-v3 encoder."""
    
    def __init__(self,
                 num_classes,
                 num_outputs: int = 3,
                 encoder_name: str = 'facebook/dinov3-vitb16-pretrain-lvd1689m',
                 features: int = 256,
                 out_channels: Optional[List[int]] = None,
                 use_bn: bool = True,
                 use_clstoken: bool = False):
        
        super().__init__()
        
        self.patch_size = 16
        self.encoder_name = encoder_name
        
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.processor = AutoImageProcessor.from_pretrained(encoder_name)
        
        self.intermediate_layer_idx = {
            'facebook/dinov3-vitb16-pretrain-lvd1689m': [2, 5, 8, 11],
            'facebook/dinov3-vits16-pretrain-lvd1689m': [2, 5, 8, 11], 
            'facebook/dinov3-vitl16-pretrain-lvd1689m': [4, 11, 17, 23],
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


class FluxDPTSegmentation(BaseDPTSegmentation):
    """FLUX-enhanced DPT segmentation model with DINO-v3 encoder (patch size 16)."""
    
    def __init__(self, 
                 flux_feature_dims: List[int] = [768, 768, 768, 768], 
                 use_concept_maps: bool = True,
                 use_flux_features: bool = True,
                 use_dino_features: bool = True,
                 **kwargs):
        self.flux_feature_dims = flux_feature_dims
        self.use_flux_features = True  # Keep this for compatibility
        self.use_concept_maps = use_concept_maps
        self.use_flux_fusion = use_flux_features
        self.use_dino_features = use_dino_features
        super().__init__(**kwargs)
    
    def _create_segmentation_head(self, **kwargs):
        return FluxDPTSegmentationHead(
            flux_dim=self.flux_feature_dims[0], 
            use_concept_maps=self.use_concept_maps,
            use_flux_features=self.use_flux_fusion,
            use_dino_features=self.use_dino_features,
            **kwargs
        )
    
    def convert_flux_features_to_spatial(self, transformer_features, h, w):
        """Convert FLUX transformer features to spatial format."""
        flux_features_spatial = []
        flux_patch_h, flux_patch_w = h // 16, w // 16  # FLUX stride = 16
        
        for tf in transformer_features:
            # tf shape: [B, seq_len, 768] where seq_len = flux_patch_h * flux_patch_w
            batch_size = tf.shape[0]
            tf_spatial = tf.permute(0, 2, 1).reshape(batch_size, -1, flux_patch_h, flux_patch_w)
            flux_features_spatial.append(tf_spatial)
        
        return flux_features_spatial
    
    def prepare_concept_maps(self, concept_maps):
        """Prepare concept maps in spatial format."""
        concept_maps_spatial = torch.stack([
            concept_maps['category'],   # [B, H_concept, W_concept]
            concept_maps['background']  # [B, H_concept, W_concept]  
        ], dim=1)  # [B, 2, H_concept, W_concept]
        
        return concept_maps_spatial
    
    def forward(self, x, transformer_features, concept_maps):
        """
        Args:
            x: Input images [B, 3, H, W] 
            transformer_features: List of 4 FLUX transformer features [B, seq_len, 768]
            concept_maps: Dict with 'category' and 'background' maps [B, H_concept, W_concept]
        """
        h, w = x.shape[-2:]
        
        # Extract features from DINO-v3 
        features = self.extract_intermediate_features(x)
        patch_h, patch_w = h // 16, w // 16  # DINO-v3 stride = 16 (same as FLUX)
        
        # Convert FLUX transformer features to spatial format
        flux_features_spatial = self.convert_flux_features_to_spatial(transformer_features, h, w)
        
        # Prepare concept maps in spatial format
        concept_maps_spatial = self.prepare_concept_maps(concept_maps)
        
        # Forward through segmentation head
        outputs = self.seg_head(
            features, 
            patch_h, 
            patch_w, 
            flux_features_spatial=flux_features_spatial,
            concept_maps_spatial=concept_maps_spatial
        )

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


class FluxDPTSegmentationHead(BaseDPTSegmentationHead):
    """DPT segmentation head enhanced with FLUX feature fusion."""
    
    def __init__(self, flux_dim=768, use_concept_maps: bool = True, use_flux_features: bool = True, use_dino_features: bool = True, **kwargs):
        self.use_concept_maps = use_concept_maps
        self.use_flux_features = use_flux_features
        self.use_dino_features = use_dino_features
        super().__init__(**kwargs)
        self._init_fusion_modules(kwargs.get('features', 256), flux_dim)
    
    def _init_fusion_modules(self, features, flux_dim):
        """Initialize FLUX feature fusion modules."""
        self.fusion_modules = nn.ModuleList([
            FluxFeatureFusion(
                vit_dim=features,
                flux_dim=flux_dim,
                output_dim=features,
                num_concept_channels=2,
                use_concept_maps=self.use_concept_maps,
                use_flux_features=self.use_flux_features,
                use_dino_features=self.use_dino_features
            ) for _ in range(4)
        ])
    
    def apply_flux_fusion(self, dpt_features, flux_features_spatial, concept_maps_spatial):
        """Apply FLUX feature fusion to DPT features."""
        fused_features = []
        
        for i, (dpt_feat, flux_feat) in enumerate(zip(dpt_features, flux_features_spatial)):
            fused_feat = self.fusion_modules[i](dpt_feat, flux_feat, concept_maps_spatial)
            fused_features.append(fused_feat)
        
        return fused_features
    
    def forward(self, out_features, patch_h, patch_w, flux_features_spatial, concept_maps_spatial):
        """Forward pass with FLUX feature fusion."""
        processed_features = self.process_encoder_features(out_features, patch_h, patch_w)
        
        layer_1_rn = self.scratch.layer1_rn(processed_features[0])
        layer_2_rn = self.scratch.layer2_rn(processed_features[1])
        layer_3_rn = self.scratch.layer3_rn(processed_features[2])
        layer_4_rn = self.scratch.layer4_rn(processed_features[3])
        
        dpt_features = [layer_1_rn, layer_2_rn, layer_3_rn, layer_4_rn]
        
        fused_features = self.apply_flux_fusion(dpt_features, flux_features_spatial, concept_maps_spatial)
        
        path_4 = self.scratch.refinenet4(fused_features[3], size=fused_features[2].shape[2:])
        path_3 = self.scratch.refinenet3(path_4, fused_features[2], size=fused_features[1].shape[2:])
        path_2 = self.scratch.refinenet2(path_3, fused_features[1], size=fused_features[0].shape[2:])
        path_1 = self.scratch.refinenet1(path_2, fused_features[0])

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


class FluxFeatureFusion(nn.Module):
    """
    Optimal fusion module combining DINO-v3, FLUX, and concept features
    with a focus on robust normalization and simplified, effective fusion.
    """
    def __init__(self, vit_dim: int = 256, flux_dim: int = 768, output_dim: int = 256, num_concept_channels: int = 2,
                 use_concept_maps: bool = True, use_flux_features: bool = True, use_dino_features: bool = True):
        super().__init__()

        # Ablation flags
        self.use_concept_maps = use_concept_maps
        self.use_flux_features = use_flux_features
        self.use_dino_features = use_dino_features

        # 1. Per-modality projection and normalization
        if self.use_dino_features:
            self.vit_projection = nn.Sequential(
                nn.Conv2d(vit_dim, output_dim, kernel_size=1),
                nn.BatchNorm2d(output_dim),
                nn.ReLU(inplace=True),
            )
        
        if self.use_flux_features:
            self.flux_projection = nn.Sequential(
                nn.Conv2d(flux_dim, output_dim, kernel_size=1),
                nn.BatchNorm2d(output_dim),
                nn.ReLU(inplace=True),
            )
        
        if self.use_concept_maps:
            self.concept_projection = nn.Sequential(
                # Use a 3x3 conv to give concept maps some spatial processing
                nn.Conv2d(num_concept_channels, output_dim // 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(output_dim // 2),
                nn.ReLU(inplace=True),
            )

        # 2. Dynamic fusion block based on enabled features
        fusion_input_dim = 0
        if self.use_dino_features:
            fusion_input_dim += output_dim
        if self.use_flux_features:
            fusion_input_dim += output_dim
        if self.use_concept_maps:
            fusion_input_dim += output_dim // 2
        
        if fusion_input_dim > 0:
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(fusion_input_dim, output_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(output_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(output_dim, output_dim, kernel_size=1),
                nn.BatchNorm2d(output_dim),
            )
        
        # Final combination layer
        if self.use_dino_features:
            self.final_conv = nn.Conv2d(output_dim * 2, output_dim, kernel_size=1)
        else:
            self.final_conv = nn.Identity()

    def forward(self, vit_features: torch.Tensor, flux_features_spatial: torch.Tensor, concept_maps_spatial: torch.Tensor):
        target_size = vit_features.shape[2:]
        
        # Prepare feature list for fusion
        fusion_features = []
        
        # Always include DINO features (base features)
        if self.use_dino_features:
            vit_proj = self.vit_projection(vit_features)
            fusion_features.append(vit_proj)
        
        # Optionally include FLUX features
        if self.use_flux_features:
            flux_resized = F.interpolate(flux_features_spatial, size=target_size, mode='bilinear', align_corners=False, antialias=True)
            flux_proj = self.flux_projection(flux_resized)
            fusion_features.append(flux_proj)
        
        # Optionally include concept maps
        if self.use_concept_maps:
            concept_resized = F.interpolate(concept_maps_spatial, size=target_size, mode='bilinear', align_corners=False, antialias=True)
            concept_proj = self.concept_projection(concept_resized)
            fusion_features.append(concept_proj)
        
        # If no fusion features are enabled, return original features
        if not fusion_features:
            return vit_features
        
        # If only one feature type, apply simple processing
        if len(fusion_features) == 1:
            if self.use_dino_features and not self.use_flux_features and not self.use_concept_maps:
                # Only DINO features - return as is
                return vit_features
            else:
                # Single modality fusion
                fused_features = fusion_features[0]
        else:
            # Multi-modality fusion
            fusion_input = torch.cat(fusion_features, dim=1)
            fused_features = self.fusion_conv(fusion_input)
        
        # Final combination with original features (if using DINO)
        if self.use_dino_features:
            output = self.final_conv(torch.cat([vit_features, fused_features], dim=1))
        else:
            # If not using DINO features, just return fused features
            output = fused_features
        
        return output
