"""
Pseudo-Mamba Lab: Tools for introspecting and analyzing Mamba internal states

This package provides utilities for:
- Tracing layer-wise state trajectories
- Visualizing SSM and convolution states
- Analyzing memory dynamics in Mamba models
"""

from .trace import LayerTrace, trace_layer, trace_model

__all__ = ['LayerTrace', 'trace_layer', 'trace_model']
