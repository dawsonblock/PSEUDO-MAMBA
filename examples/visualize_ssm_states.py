"""
Visualize Mamba SSM State Trajectories

This script demonstrates how to use the pseudo_mamba_lab tools to:
1. Trace SSM state evolution over a sequence
2. Visualize state dynamics with matplotlib
3. Analyze memory patterns in Mamba layers

Example usage:
    python examples/visualize_ssm_states.py
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

from mamba_ssm import Mamba
from pseudo_mamba_lab import trace_layer, extract_ssm_state_sequence


def create_synthetic_sequence(length=100, dim=64, pattern="random"):
    """
    Create a synthetic test sequence.

    Args:
        length: Sequence length
        dim: Hidden dimension
        pattern: Type of pattern ("random", "pulse", "sine")

    Returns:
        Tensor [1, length, dim]
    """
    if pattern == "random":
        return torch.randn(1, length, dim)
    elif pattern == "pulse":
        seq = torch.zeros(1, length, dim)
        # Add pulses at specific positions
        seq[0, 10, :] = 5.0  # First pulse
        seq[0, 50, :] = 5.0  # Second pulse
        seq[0, 90, :] = 5.0  # Third pulse
        return seq
    elif pattern == "sine":
        t = torch.linspace(0, 4 * np.pi, length).unsqueeze(0).unsqueeze(-1)
        return torch.sin(t).expand(1, length, dim)
    else:
        raise ValueError(f"Unknown pattern: {pattern}")


def plot_ssm_state_heatmap(trace, layer_name="Mamba Layer", batch_idx=0):
    """
    Plot SSM state as a heatmap over time.

    Args:
        trace: LayerTrace object
        layer_name: Name for the plot title
        batch_idx: Which batch element to visualize
    """
    # Extract SSM state for first batch element
    ssm_seq = extract_ssm_state_sequence(trace, batch_idx=batch_idx)  # [L+1, D*expand, dstate]

    # Reshape to 2D for visualization [L+1, D*expand*dstate]
    L, D_inner, dstate = ssm_seq.shape
    ssm_flat = ssm_seq.reshape(L, D_inner * dstate).cpu().numpy()

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(ssm_flat.T, aspect='auto', cmap='RdBu', origin='lower',
                   vmin=-ssm_flat.std() * 2, vmax=ssm_flat.std() * 2)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('SSM State Dimension')
    ax.set_title(f'{layer_name}: SSM State Evolution')
    plt.colorbar(im, ax=ax, label='State Value')

    return fig


def plot_state_norms(trace, batch_idx=0):
    """
    Plot the L2 norm of SSM and conv states over time.

    Args:
        trace: LayerTrace object
        batch_idx: Which batch element to visualize
    """
    ssm_seq = extract_ssm_state_sequence(trace, batch_idx=batch_idx)
    conv_seq = trace.conv_states[:, batch_idx, :, :].cpu()

    # Compute norms
    ssm_norms = torch.norm(ssm_seq.reshape(ssm_seq.shape[0], -1), dim=1).numpy()
    conv_norms = torch.norm(conv_seq.reshape(conv_seq.shape[0], -1), dim=1).numpy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # SSM state norm
    ax1.plot(ssm_norms, linewidth=2, color='blue')
    ax1.set_ylabel('L2 Norm')
    ax1.set_title('SSM State Norm Over Time')
    ax1.grid(True, alpha=0.3)

    # Conv state norm
    ax2.plot(conv_norms, linewidth=2, color='green')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('L2 Norm')
    ax2.set_title('Convolution State Norm Over Time')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_state_pca(trace, n_components=2, batch_idx=0):
    """
    Plot PCA projection of SSM states over time.

    Args:
        trace: LayerTrace object
        n_components: Number of PCA components (2 or 3)
        batch_idx: Which batch element to visualize
    """
    from sklearn.decomposition import PCA

    ssm_seq = extract_ssm_state_sequence(trace, batch_idx=batch_idx)
    L, D_inner, dstate = ssm_seq.shape
    ssm_flat = ssm_seq.reshape(L, D_inner * dstate).cpu().numpy()

    # Fit PCA
    pca = PCA(n_components=n_components)
    ssm_pca = pca.fit_transform(ssm_flat)

    if n_components == 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(ssm_pca[:, 0], ssm_pca[:, 1],
                            c=np.arange(L), cmap='viridis', s=20)
        ax.plot(ssm_pca[:, 0], ssm_pca[:, 1], 'k-', alpha=0.3, linewidth=0.5)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax.set_title('SSM State Trajectory (PCA)')
        plt.colorbar(scatter, ax=ax, label='Time Step')

    elif n_components == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(ssm_pca[:, 0], ssm_pca[:, 1], ssm_pca[:, 2],
                            c=np.arange(L), cmap='viridis', s=20)
        ax.plot(ssm_pca[:, 0], ssm_pca[:, 1], ssm_pca[:, 2], 'k-', alpha=0.3, linewidth=0.5)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
        ax.set_title('SSM State Trajectory (3D PCA)')
        plt.colorbar(scatter, ax=ax, label='Time Step')

    return fig


def main():
    """Run visualization examples."""
    print("Pseudo-Mamba State Visualization Demo")
    print("=" * 60)

    # Create a Mamba layer
    print("\n1. Creating Mamba layer...")
    layer = Mamba(
        d_model=64,
        d_state=16,
        d_conv=4,
        expand=2,
        layer_idx=0
    )
    layer.eval()

    # Create synthetic sequence with pulses
    print("2. Creating synthetic sequence with pulses...")
    x = create_synthetic_sequence(length=100, dim=64, pattern="pulse")

    # Trace the layer
    print("3. Tracing layer state evolution...")
    with torch.no_grad():
        trace = trace_layer(layer, x)

    print(f"   Output shape: {trace.outputs.shape}")
    print(f"   SSM state trajectory shape: {trace.ssm_states.shape}")
    print(f"   Conv state trajectory shape: {trace.conv_states.shape}")

    # Create visualizations
    print("\n4. Generating visualizations...")

    try:
        print("   - SSM state heatmap...")
        fig1 = plot_ssm_state_heatmap(trace, "Mamba Layer (Pulse Input)")
        fig1.savefig('examples/ssm_state_heatmap.png', dpi=150, bbox_inches='tight')
        print("     Saved: examples/ssm_state_heatmap.png")

        print("   - State norms over time...")
        fig2 = plot_state_norms(trace)
        fig2.savefig('examples/state_norms.png', dpi=150, bbox_inches='tight')
        print("     Saved: examples/state_norms.png")

        try:
            from sklearn.decomposition import PCA
            print("   - PCA projection (2D)...")
            fig3 = plot_state_pca(trace, n_components=2)
            fig3.savefig('examples/ssm_state_pca_2d.png', dpi=150, bbox_inches='tight')
            print("     Saved: examples/ssm_state_pca_2d.png")

            print("   - PCA projection (3D)...")
            fig4 = plot_state_pca(trace, n_components=3)
            fig4.savefig('examples/ssm_state_pca_3d.png', dpi=150, bbox_inches='tight')
            print("     Saved: examples/ssm_state_pca_3d.png")
        except ImportError:
            print("     [Skipped] scikit-learn not available for PCA plots")

    except Exception as e:
        print(f"   Error during visualization: {e}")
        print("   This is a headless environment; plots saved but not displayed")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("\nTo explore further:")
    print("  - Try different input patterns: 'random', 'sine'")
    print("  - Adjust layer parameters: d_state, d_conv, expand")
    print("  - Trace multiple layers and compare dynamics")


if __name__ == "__main__":
    main()
