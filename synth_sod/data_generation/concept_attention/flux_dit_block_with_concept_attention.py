import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple
from torch import nn
import einops

from diffusers.models.transformers.transformer_flux import FluxTransformerBlock
from diffusers.models.attention import Attention
from diffusers.models.embeddings import apply_rotary_emb


class FluxConceptAttentionProcessor:
    """
    Custom attention processor for FLUX that implements concept attention.
    Exactly matches original FLUX attention pattern while adding concept observation stream.
    """
    
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxConceptAttentionProcessor requires PyTorch 2.0")
    
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        concept_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        concept_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        
        batch_size, _, _ = encoder_hidden_states.shape
        
        # *** MAIN GENERATION STREAM *** - Exact FLUX Pattern
        
        # 1. `sample` projections (image hidden states)
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        image_query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        image_key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        image_value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            image_query = attn.norm_q(image_query)
        if attn.norm_k is not None:
            image_key = attn.norm_k(image_key)

        # 2. `context` projections (text encoder hidden states) - FLUX specific
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)

        if attn.norm_added_q is not None:
            encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
        if attn.norm_added_k is not None:
            encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

        # 3. FLUX concatenation order: [encoder, image]
        query = torch.cat([encoder_hidden_states_query_proj, image_query], dim=2)
        key = torch.cat([encoder_hidden_states_key_proj, image_key], dim=2)
        value = torch.cat([encoder_hidden_states_value_proj, image_value], dim=2)

        # 4. Apply rotary embeddings to FULL concatenated tensors (FLUX way)
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # 5. Main text+image attention (FLUX standard)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # *** CONCEPT STREAM *** - Exact same logic as text-image (following reference)
        concept_hidden_states_output = None
        concept_attention_maps = None
        
        if concept_hidden_states is not None:
            # Use TEXT projections for concepts (like reference: txt_attn.qkv)
            concept_query = attn.add_q_proj(concept_hidden_states)   # Same as text!
            concept_key = attn.add_k_proj(concept_hidden_states)     # Same as text!
            concept_value = attn.add_v_proj(concept_hidden_states)   # Same as text!
            
            concept_query = concept_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            concept_key = concept_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            concept_value = concept_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            
            if attn.norm_added_q is not None:
                concept_query = attn.norm_added_q(concept_query)    # Text normalization
            if attn.norm_added_k is not None:
                concept_key = attn.norm_added_k(concept_key)        # Text normalization

            concept_image_q = torch.cat([concept_query, image_query], dim=2)
            concept_image_k = torch.cat([concept_key, image_key], dim=2)
            concept_image_v = torch.cat([concept_value, image_value], dim=2)
            
            # Apply concept rotary embeddings to FULL concatenated tensor (like reference)
            if concept_rotary_emb is not None:
                concept_image_q = apply_rotary_emb(concept_image_q, concept_rotary_emb)
                concept_image_k = apply_rotary_emb(concept_image_k, concept_rotary_emb)
            
            # Do the joint attention operation (like reference)
            concept_image_attn = F.scaled_dot_product_attention(
                concept_image_q,
                concept_image_k, 
                concept_image_v,
                dropout_p=0.0,
                is_causal=False
            )
            
            # Separate the concept attention (like reference: concept_attn = concept_image_attn[:, :, :concepts.shape[1]])
            concept_attn = concept_image_attn[:, :, :concept_hidden_states.size(1)]
            concept_hidden_states_output = concept_attn.transpose(1, 2).reshape(
                batch_size, -1, attn.heads * head_dim
            )
            
            # Compute attention maps from concept and image queries (before attention)
            # Save vectors for postprocessing (like reference implementation)
            concept_attention_maps = {
                'concept_vectors': concept_query,  # (batch, heads, concepts, dim)
                'image_vectors': image_query       # (batch, heads, patches, dim)
            }

        # 6. FLUX output processing
        encoder_hidden_states_out, hidden_states_out = (
            hidden_states[:, : encoder_hidden_states.shape[1]],
            hidden_states[:, encoder_hidden_states.shape[1] :],
        )

        # linear proj
        hidden_states_out = attn.to_out[0](hidden_states_out)
        # dropout
        hidden_states_out = attn.to_out[1](hidden_states_out)

        # FLUX specific: separate output projection for encoder
        encoder_hidden_states_out = attn.to_add_out(encoder_hidden_states_out)
        
        # Process concept outputs with same projections
        if concept_hidden_states_output is not None:
            concept_hidden_states_output = attn.to_out[0](concept_hidden_states_output)
            concept_hidden_states_output = attn.to_out[1](concept_hidden_states_output)
            
            # Return vectors for postprocessing (like reference implementation)
            concept_attention_maps = {
                'concept_vectors': concept_hidden_states_output,  # Final processed concept features
                'image_vectors': hidden_states_out               # Final processed image features
            }

        return hidden_states_out, encoder_hidden_states_out, concept_hidden_states_output, concept_attention_maps


class FluxTransformerBlockWithConceptAttention(FluxTransformerBlock):
    """
    Simplified FLUX transformer block with concept attention.
    Uses the elegant CogVideoX approach with custom attention processor.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn.processor = FluxConceptAttentionProcessor()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        concept_hidden_states: Optional[torch.Tensor],
        temb: torch.Tensor,
        concept_temb: Optional[torch.Tensor],
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        concept_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        concept_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, 
            emb=temb
        )
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, 
            emb=temb
        )
        
        # Concept normalization (use same temb as main stream if concept_temb is None)
        norm_concept_hidden_states = None
        concept_gate_msa = concept_shift_mlp = concept_scale_mlp = concept_gate_mlp = None
        concept_gate_ff = None
        
        if concept_hidden_states is not None:
            effective_concept_temb = concept_temb if concept_temb is not None else temb
            norm_concept_hidden_states, concept_gate_msa, concept_shift_mlp, concept_scale_mlp, concept_gate_mlp = self.norm1_context(
                concept_hidden_states, 
                emb=effective_concept_temb
            )
        
        joint_attention_kwargs = joint_attention_kwargs or {}
        
        # Attention with concept attention (using our custom processor)
        attention_outputs = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            concept_hidden_states=norm_concept_hidden_states,
            image_rotary_emb=image_rotary_emb,
            concept_rotary_emb=concept_rotary_emb,
            **joint_attention_kwargs,
        )

        if len(attention_outputs) == 4:
            attn_output, context_attn_output, concept_attn_output, concept_attention_maps = attention_outputs
            ip_attn_output = None
        elif len(attention_outputs) == 5:
            attn_output, context_attn_output, concept_attn_output, concept_attention_maps, ip_attn_output = attention_outputs
        else:
            # Fallback for when no concept attention
            attn_output, context_attn_output = attention_outputs[:2]
            concept_attn_output = None
            concept_attention_maps = None
            ip_attn_output = attention_outputs[2] if len(attention_outputs) > 2 else None

        ################## Process Concept Features FIRST (like CogVideoX) ##################
        if concept_attn_output is not None and concept_hidden_states is not None:
            # Apply concept attention gate and residual
            concept_attn_output = concept_gate_msa.unsqueeze(1) * concept_attn_output
            concept_hidden_states = concept_hidden_states + concept_attn_output
            
            # Concept feedforward processing (norm2_context is regular LayerNorm, not adaptive)
            norm_concept_hidden_states = self.norm2_context(concept_hidden_states)
            norm_concept_hidden_states = norm_concept_hidden_states * (1 + concept_scale_mlp[:, None]) + concept_shift_mlp[:, None]
            concept_ff_output = self.ff_context(norm_concept_hidden_states)
            concept_hidden_states = concept_hidden_states + concept_gate_mlp.unsqueeze(1) * concept_ff_output
            
            if concept_hidden_states.dtype == torch.float16:
                concept_hidden_states = concept_hidden_states.clip(-65504, 65504)

        ################## Now Process Main Generation Stream ##################
        # Standard FLUX processing for image features
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output
        hidden_states = hidden_states + ff_output
        
        if ip_attn_output is not None:
            hidden_states = hidden_states + ip_attn_output

        # Standard FLUX processing for text features
        if context_attn_output is not None:
            context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
            encoder_hidden_states = encoder_hidden_states + context_attn_output

            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

            context_ff_output = self.ff_context(norm_encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
            
            if encoder_hidden_states.dtype == torch.float16:
                encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states, concept_hidden_states, concept_attention_maps
    