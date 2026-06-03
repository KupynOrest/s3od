from typing import List, Dict

import torch
import numpy as np
from typing import Any, Dict, Optional, Tuple, Union
from torch import nn
from torch import Tensor

from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.models.transformers.transformer_flux import FluxSingleTransformerBlock
from diffusers.models.normalization import AdaLayerNormContinuous
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers, BaseOutput
from diffusers.utils.import_utils import is_torch_npu_available
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.embeddings import CombinedTimestepGuidanceTextProjEmbeddings, CombinedTimestepTextProjEmbeddings, FluxPosEmbed

from data_generation.concept_attention.flux_dit_block_with_concept_attention import FluxTransformerBlockWithConceptAttention

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class FluxTransformer2DOutputWithConceptAttention(BaseOutput):
    sample: torch.Tensor
    concept_attention_maps: torch.Tensor


class FluxTransformer2DModelWithConceptAttention(FluxTransformer2DModel):
    """
        The Transformer model introduced in Flux with Concept Attention.
    """

    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: Optional[int] = None,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = True,
        axes_dims_rope: Tuple[int] = (16, 56, 56),
        feature_locations: Optional[Dict[str, List[int]]] = None,
    ):
        super().__init__(
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            num_single_layers=num_single_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            joint_attention_dim=joint_attention_dim,
            pooled_projection_dim=pooled_projection_dim,
            guidance_embeds=guidance_embeds,
            axes_dims_rope=axes_dims_rope,
        )
        self.out_channels = out_channels or in_channels
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim

        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)

        text_time_guidance_cls = (
            CombinedTimestepGuidanceTextProjEmbeddings if self.config.guidance_embeds else CombinedTimestepTextProjEmbeddings
        )
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=self.inner_dim, pooled_projection_dim=self.config.pooled_projection_dim
        )

        self.context_embedder = nn.Linear(self.config.joint_attention_dim, self.inner_dim)
        self.x_embedder = nn.Linear(self.config.in_channels, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                FluxTransformerBlockWithConceptAttention(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                )
                for i in range(self.config.num_layers)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                FluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                )
                for i in range(self.config.num_single_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False

        self.stored_features: Dict[str, Tensor] = {}
        self.feature_locations = feature_locations or {
            "transformer_blocks": [],
            "single_transformer_blocks": [4, 16, 27, 36],
        }
        self._register_feature_hooks()
    
    def get_features(self) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Get the stored feature maps as raw tokens for downstream reshaping.

        For dual stream transformer blocks: Returns the second item in the tuple (image tokens)
          Shape: [B, H*W, C] where H*W is the actual spatial size

        For single stream transformer blocks: Extracts image tokens from full sequence
          Shape: [B, H*W, C] where H*W is the actual spatial size (excluding text tokens)

        Returns:
            Tuple containing:
                - List of transformer block features as tokens [B, H*W, C]
                - List of single transformer block features as tokens [B, H*W, C]
        """
        transformer_features = []
        single_transformer_features = []

        for name, feature_output in self.stored_features.items():
            if "single_transformer_blocks" in name:
                batch_size, seq_len, embed_dim = feature_output.shape
                text_tokens = 512  # Number of text tokens
                image_feature = feature_output[:, text_tokens:, :]  # [B, H*W, C]
                single_transformer_features.append(image_feature)
            elif "transformer_blocks" in name:
                if isinstance(feature_output, tuple) and len(feature_output) >= 2:
                    image_feature = feature_output[1]  # [B, H*W, C]
                    transformer_features.append(image_feature)
        
        return (transformer_features, single_transformer_features)
    
    def _get_hook(self, name: str):
        """
        Create a forward hook function for feature extraction.

        Args:
            name: Identifier for the layer where the hook will be attached

        Returns:
            Callable hook function that stores the layer's output tensor
        """

        def hook(
                module: nn.Module, input: Union[Tensor, Tuple[Tensor, ...]], output: Tensor
        ) -> None:
            self.stored_features[name] = output

        return hook

    def _register_feature_hooks(self) -> None:
        """
        Register forward hooks on the specified layers to capture their outputs.

        Attaches hooks based on the feature_locations configuration:
        - transformer_blocks: Main transformer blocks (indexed from 0 to 18)
        - single_transformer_blocks: Single transformer blocks (indexed from 0 to 37)
        """
        for block_type, indices in self.feature_locations.items():
            if block_type == "transformer_blocks":
                for idx in indices:
                    if 0 <= idx < len(self.transformer_blocks):
                        self.transformer_blocks[idx].register_forward_hook(
                            self._get_hook(f"{block_type}_{idx}")
                        )
            elif block_type == "single_transformer_blocks":
                for idx in indices:
                    if 0 <= idx < len(self.single_transformer_blocks):
                        self.single_transformer_blocks[idx].register_forward_hook(
                            self._get_hook(f"{block_type}_{idx}")
                        )

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        concept_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        pooled_concept_embeds: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        concept_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        concept_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ) -> Union[torch.Tensor, FluxTransformer2DOutputWithConceptAttention]:
        """
        The [`FluxTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.Tensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            concept_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary with parameters for Concept Attention.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        concept_temb = None
        if pooled_concept_embeds is not None:
            if guidance is None:
                concept_temb = self.time_text_embed(timestep, pooled_concept_embeds)
            else:
                concept_temb = self.time_text_embed(timestep, guidance, pooled_concept_embeds)

        # Apply the context embedder to the concept_hidden_states
        if concept_hidden_states is not None:
            concept_hidden_states = self.context_embedder(concept_hidden_states)

        if txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        concept_image_ids = torch.cat((concept_ids, img_ids), dim=0)
        concept_rotary_emb = self.pos_embed(concept_image_ids)

        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
            joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

        # Initialize concept attention processing (collect raw maps only - no processing!)
        all_concept_attention_maps = []

        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                raise NotImplementedError("Gradient checkpointing is not implemented for concept attention.")
                # encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                #     block,
                #     hidden_states,
                #     encoder_hidden_states,
                #     temb,
                #     image_rotary_emb,
                #     concept_rotary_emb,
                # )
            else:
                encoder_hidden_states, hidden_states, concept_hidden_states, current_concept_attention_maps = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    concept_hidden_states=concept_hidden_states,
                    temb=temb,
                    concept_temb=concept_temb,
                    image_rotary_emb=image_rotary_emb,
                    concept_rotary_emb=concept_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                    concept_attention_kwargs=concept_attention_kwargs,
                )
                
                # Collect raw attention maps only (no processing here!)
                if (current_concept_attention_maps is not None and 
                    concept_attention_kwargs is not None and 
                    index_block in concept_attention_kwargs["layers"]):
                    all_concept_attention_maps.append(current_concept_attention_maps)

            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                # For Xlabs ControlNet.
                if controlnet_blocks_repeat:
                    hidden_states = (
                        hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                    )
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

        if concept_hidden_states is not None:
            concept_hidden_states = concept_hidden_states.cpu()
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        for index_block, block in enumerate(self.single_transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    temb,
                    image_rotary_emb,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )
            # controlnet residual
            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                    + controlnet_single_block_samples[index_block // interval_control]
                )

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        # Process collected attention maps (pass through dictionaries to pipeline)
        concept_attention_maps = None
        if all_concept_attention_maps:
            # Return the collected dictionaries as-is for pipeline postprocessing
            concept_attention_maps = all_concept_attention_maps

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output, concept_attention_maps)
        
        return FluxTransformer2DOutputWithConceptAttention(sample=output, concept_attention_maps=concept_attention_maps)
