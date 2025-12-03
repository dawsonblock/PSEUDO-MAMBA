import torch
import torch.nn as nn

# Try to import the compiled C++/CUDA extension
try:
    import pseudo_mamba_ext  # built by pseudo_mamba/kernels/pseudo_mamba/setup.py
    HAS_EXT = True
except ImportError:
    HAS_EXT = False


class PseudoMambaFunction(torch.autograd.Function):
    """
    Autograd wrapper around the Pseudo-Mamba recurrence:

        y = tanh(x + h)

    If the CUDA extension is available we call:
        pseudo_mamba_ext.pseudo_mamba_forward(x, h)
    Otherwise we fall back to pure PyTorch.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # x, h: [B, D]
        if HAS_EXT:
            y = pseudo_mamba_ext.pseudo_mamba_forward(x, h)
        else:
            y = torch.tanh(x + h)

        ctx.save_for_backward(x, h, y)
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # We implement backward in Python; the CUDA kernel is only for forward
        x, h, y = ctx.saved_tensors
        # dy/d(x+h) = 1 - tanh^2(x+h) = 1 - y^2
        dtanh = 1.0 - y.pow(2)
        grad_common = grad_output * dtanh  # [B, D]
        # dy/dx = dy/dh = dtanh
        grad_x = grad_common
        grad_h = grad_common
        return grad_x, grad_h


def pseudo_mamba_forward(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """
    Functional API:

        y = pseudo_mamba_forward(x, h)

    x, h: [B, D]
    """
    return PseudoMambaFunction.apply(x, h)


class PseudoMambaCore(nn.Module):
    """
    Simple module wrapper to match the recurrent-core interface used in RL:

        - init_state(batch_size, device) -> [B, D] tensor
        - forward(x_t, h_t) -> h_next tensor

    PseudoMambaExtBlock handles projection and output head; this core is only
    responsible for the recurrence in hidden space.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(self, x_t: torch.Tensor, h_t: torch.Tensor) -> torch.Tensor:
        """
        x_t, h_t: [B, D] where D == hidden_dim
        Returns:
            h_next: [B, D]
        """
        h_next = pseudo_mamba_forward(x_t, h_t)
        return h_next
