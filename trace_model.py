import torch
from torch import Tensor
from dataclasses import dataclass
from typing import List, Optional, Union

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel, MixerModel
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba2 import Mamba2
from pseudo_mamba_introspect import trace_mamba_sequence, MambaStateTrace

@dataclass
class ModelTrace:
    """
    Holds state traces for an entire model.
    """
    layer_traces: List[Optional[MambaStateTrace]]
    logits: Optional[Tensor] = None

    def to(self, device: torch.device | str) -> "ModelTrace":
        return ModelTrace(
            layer_traces=[t.to(device) if t else None for t in self.layer_traces],
            logits=self.logits.to(device) if self.logits is not None else None,
        )

@torch.no_grad()
def trace_model(
    model: Union[MambaLMHeadModel, MixerModel],
    input_ids: Tensor,
    *,
    clone_states: bool = True,
    detach: bool = True,
) -> ModelTrace:
    """
    Runs a full Mamba model step-by-step and captures internal states from all Mamba layers.
    
    Args:
        model: MambaLMHeadModel or MixerModel.
        input_ids: [B, L] integer tensor of token IDs.
        clone_states: Whether to clone states at each step (safer, more memory).
        detach: Whether to detach tensors from graph.
        
    Returns:
        ModelTrace object containing a list of MambaStateTrace (one per layer).
    """
    # 1. Identify backbone
    if isinstance(model, MambaLMHeadModel):
        backbone = model.backbone
    elif isinstance(model, MixerModel):
        backbone = model
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")

    # 2. Embed inputs
    # [B, L] -> [B, L, D]
    hidden_states = backbone.embedding(input_ids)
    
    # 3. Iterate layers
    layer_traces = []
    residual = None
    
    for i, layer in enumerate(backbone.layers):
        # layer is a Block
        # We need to manually replicate the Block's forward pass logic 
        # BUT replace the mixer call with trace_mamba_sequence.
        
        # Block forward logic (simplified from mamba_ssm/modules/block.py):
        # 1. Residual connection & Norm
        # 2. Mixer (Mamba) -> Trace this!
        # 3. MLP (if exists)
        
        # --- Step 1: Pre-Mixer Norm ---
        if not layer.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states_norm = layer.norm(residual.to(dtype=layer.norm.weight.dtype))
            if layer.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            # Fused add+norm
            # We need to import the fused kernel if we want to match exactly, 
            # or just rely on the layer's forward if we weren't tracing.
            # Since we need to intercept the mixer input, we must execute the norm.
            # For simplicity in this lab tool, let's use the layer's norm directly 
            # if fused kernel is tricky to invoke manually without side effects.
            # Actually, let's try to use the public API if possible.
            # But we need to intercept the mixer.
            
            # Let's use the layer's norm module directly, assuming standard behavior if fused is hard.
            # Wait, layer.forward calls layer_norm_fn.
            from mamba_ssm.ops.triton.layer_norm import layer_norm_fn, RMSNorm
            
            hidden_states_norm, residual = layer_norm_fn(
                hidden_states,
                layer.norm.weight,
                layer.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=layer.residual_in_fp32,
                eps=layer.norm.eps,
                is_rms_norm=isinstance(layer.norm, RMSNorm)
            )

        # --- Step 2: Mixer (Mamba) ---
        # Check if mixer is Mamba/Mamba2
        if isinstance(layer.mixer, (Mamba, Mamba2)):
            # Trace it!
            # trace_mamba_sequence expects [B, L, D]
            # It runs the mixer step-by-step.
            
            # Note: trace_mamba_sequence currently only supports Mamba(1).
            # Mamba2 has a different step API? Let's assume Mamba1 for now as per user request.
            if isinstance(layer.mixer, Mamba2):
                print(f"Warning: Layer {i} is Mamba2. Tracing might be partial or unsupported. Falling back to standard forward.")
                hidden_states = layer.mixer(hidden_states_norm)
                layer_traces.append(None)
            else:
                # Mamba1
                outputs, trace = trace_mamba_sequence(
                    layer.mixer, 
                    hidden_states_norm, 
                    clone_states=clone_states, 
                    detach=detach
                )
                hidden_states = outputs
                layer_traces.append(trace)
        else:
            # MLP or Attention or Identity
            hidden_states = layer.mixer(hidden_states_norm)
            layer_traces.append(None)

        # --- Step 3: MLP (Optional) ---
        if layer.mlp is not None:
            if not layer.fused_add_norm:
                residual = hidden_states + residual
                hidden_states_norm = layer.norm2(residual.to(dtype=layer.norm2.weight.dtype))
                if layer.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                hidden_states_norm, residual = layer_norm_fn(
                    hidden_states,
                    layer.norm2.weight,
                    layer.norm2.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=layer.residual_in_fp32,
                    eps=layer.norm2.eps,
                    is_rms_norm=isinstance(layer.norm2, RMSNorm)
                )
            hidden_states = layer.mlp(hidden_states_norm)

    # 4. Final Norm
    if not backbone.fused_add_norm:
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = backbone.norm_f(residual.to(dtype=backbone.norm_f.weight.dtype))
    else:
        hidden_states = layer_norm_fn(
            hidden_states,
            backbone.norm_f.weight,
            backbone.norm_f.bias,
            eps=backbone.norm_f.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=backbone.residual_in_fp32,
            is_rms_norm=isinstance(backbone.norm_f, RMSNorm)
        )

    # 5. LM Head (if applicable)
    logits = None
    if isinstance(model, MambaLMHeadModel):
        logits = model.lm_head(hidden_states)

    return ModelTrace(layer_traces=layer_traces, logits=logits)
