import json
import os

def create_notebook():
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    def add_cell(source, cell_type="code"):
        notebook["cells"].append({
            "cell_type": cell_type,
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source if isinstance(source, list) else [line + "\n" for line in source.split("\n")]
        })

    # Cell 1: Title
    add_cell([
        "# Pseudo-Mamba State Lab: Black Box Mapping Demo",
        "",
        "This notebook demonstrates how to use the **Pseudo-Mamba State Lab** tools to extract, visualize, and analyze the internal states of a Mamba model.",
        "",
        "**Tools Used:**",
        "- `trace_model.py`: Extracts full state trajectories from all layers.",
        "- `visualize_states.py`: Visualizes state evolution (heatmaps, norms, PCA)."
    ], cell_type="markdown")

    # Cell 2: Imports
    add_cell([
        "import torch",
        "import matplotlib.pyplot as plt",
        "from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel",
        "from mamba_ssm.models.config_mamba import MambaConfig",
        "from trace_model import trace_model",
        "from visualize_states import plot_state_heatmap, plot_state_norm, analyze_state_pca",
        "",
        "%matplotlib inline"
    ])

    # Cell 3: Setup Model
    add_cell([
        "# 1. Setup a small Mamba model for demonstration",
        "config = MambaConfig(",
        "    d_model=256,",
        "    n_layer=4,",
        "    d_state=16,",
        "    expand=2,",
        "    vocab_size=1000",
        ")",
        "device = 'cuda' if torch.cuda.is_available() else 'cpu'",
        "model = MambaLMHeadModel(config).to(device)",
        "print(f'Model created on {device}')"
    ])

    # Cell 4: Create Input
    add_cell([
        "# 2. Create a dummy input sequence",
        "batch_size = 2",
        "seq_len = 64",
        "input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)",
        "print(f'Input shape: {input_ids.shape}')"
    ])

    # Cell 5: Trace Model
    add_cell([
        "# 3. Run the model with tracing",
        "print('Tracing model execution...')",
        "trace = trace_model(model, input_ids)",
        "print(f'Captured traces for {len(trace.layer_traces)} layers.')"
    ])

    # Cell 6: Visualize Norms
    add_cell([
        "# 4. Visualize State Norms",
        "# This shows how the magnitude of the SSM state evolves over time across layers.",
        "plot_state_norm(trace)"
    ])

    # Cell 7: Heatmaps
    add_cell([
        "# 5. Visualize State Heatmaps",
        "# Let's look at the internal state of Layer 0, Dimension 0.",
        "layer_idx = 0",
        "dim_idx = 0",
        "plot_state_heatmap(trace.layer_traces[layer_idx], layer_idx, dim_idx, state_type='ssm')"
    ])

    # Cell 8: PCA
    add_cell([
        "# 6. PCA Trajectory Analysis",
        "# Visualize the state trajectory in low-dimensional space.",
        "analyze_state_pca(trace.layer_traces[0], layer_idx=0)"
    ])
    
    # Cell 9: Conclusion
    add_cell([
        "## Conclusion",
        "You have successfully mapped the 'Black Box'! 🕵️‍♂️",
        "",
        "Use these tools to:",
        "- Debug why a model fails on specific tokens.",
        "- Analyze how memory is maintained over long sequences.",
        "- Correlate internal states with specific input patterns."
    ], cell_type="markdown")

    with open("lab_demo.ipynb", "w") as f:
        json.dump(notebook, f, indent=4)
    
    print("Generated lab_demo.ipynb")

if __name__ == "__main__":
    create_notebook()
