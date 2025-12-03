import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List
from pseudo_mamba_introspect import MambaStateTrace
from trace_model import ModelTrace

def plot_state_heatmap(
    trace: MambaStateTrace, 
    layer_idx: int, 
    dim_idx: int = 0, 
    state_type: str = "ssm",
    save_path: Optional[str] = None
):
    """
    Plots a heatmap of the state evolution for a specific channel/dimension.
    
    Args:
        trace: MambaStateTrace object.
        layer_idx: Index of the layer (for labeling).
        dim_idx: Which channel dimension (D) to visualize.
        state_type: "ssm" or "conv".
        save_path: If provided, saves the plot to this path.
    """
    # trace.ssm_states: [L+1, B, D_expand, d_state]
    if state_type == "ssm":
        states = trace.ssm_states
        d_inner_name = "d_state"
    elif state_type == "conv":
        states = trace.conv_states
        d_inner_name = "d_conv"
    else:
        raise ValueError("state_type must be 'ssm' or 'conv'")
        
    # Select batch 0, specific dimension
    # [L+1, B, D, N] -> [L+1, N]
    data = states[:, 0, dim_idx, :].cpu().numpy()
    
    plt.figure(figsize=(10, 6))
    plt.imshow(data.T, aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(label='State Value')
    plt.xlabel('Time Step (Token)')
    plt.ylabel(f'Latent State Dimension ({d_inner_name})')
    plt.title(f'Layer {layer_idx} {state_type.upper()} State Heatmap (Dim {dim_idx})')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_state_norm(
    model_trace: ModelTrace,
    save_path: Optional[str] = None
):
    """
    Plots the L2 norm of the SSM state averaged across batch and dimensions,
    for each layer over time.
    """
    plt.figure(figsize=(12, 6))
    
    for i, trace in enumerate(model_trace.layer_traces):
        if trace is None:
            continue
            
        # ssm_states: [L+1, B, D, N]
        # Compute norm over N (last dim)
        # [L+1, B, D]
        norms = torch.norm(trace.ssm_states, p=2, dim=-1)
        
        # Average over B and D
        # [L+1]
        avg_norm = norms.mean(dim=(1, 2)).cpu().numpy()
        
        plt.plot(avg_norm, label=f'Layer {i}')
        
    plt.xlabel('Time Step')
    plt.ylabel('Average SSM State L2 Norm')
    plt.title('State Magnitude Evolution Across Layers')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def analyze_state_pca(
    trace: MambaStateTrace,
    layer_idx: int,
    n_components: int = 3,
    save_path: Optional[str] = None
):
    """
    Performs PCA on the flattened state trajectories to visualize trajectory in low-D space.
    """
    # ssm_states: [L+1, B, D, N]
    # We want to see the trajectory of the "state" as a whole.
    # Flatten D and N? Or just average?
    # Let's look at the trajectory of the entire hidden state [D*N]
    
    L, B, D, N = trace.ssm_states.shape
    
    # Focus on batch 0 for trajectory
    # [L, D*N]
    flat_states = trace.ssm_states[:, 0, :, :].reshape(L, -1).cpu().float()
    
    # Center data
    mean = flat_states.mean(dim=0)
    centered = flat_states - mean
    
    # SVD
    # U, S, V = torch.svd(centered)
    # PCA coords = U * S
    try:
        U, S, V = torch.pca_lowrank(centered, q=n_components)
        coords = torch.matmul(centered, V[:, :n_components]).numpy()
        
        fig = plt.figure(figsize=(10, 8))
        if n_components >= 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], marker='o', markersize=2, alpha=0.6)
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
        else:
            plt.plot(coords[:, 0], coords[:, 1], marker='o', markersize=2, alpha=0.6)
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            
        plt.title(f'Layer {layer_idx} State Trajectory (PCA)')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
    except Exception as e:
        print(f"PCA analysis failed: {e}")

