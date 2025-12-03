import torch
import torch.nn as nn
import pseudo_mamba_ext  # the compiled C++/CUDA extension


class _PseudoMambaCoreFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, h):
        """
        x, h: [B, D] (already projected to the model dimension)
        Returns:
            y: [B, D]
        """
        # Ensure contiguous for C++/CUDA
        x_c = x.contiguous()
        h_c = h.contiguous()
        y = pseudo_mamba_ext.pseudo_mamba_forward(x_c, h_c)
        ctx.save_for_backward(x_c, h_c, y)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, h, y = ctx.save_for_backward  # type: ignore[attr-defined]
        grad_y_c = grad_y.contiguous()
        grad_x, grad_h = pseudo_mamba_ext.pseudo_mamba_backward(
            grad_y_c, x, h, y
        )
        return grad_x, grad_h


class PseudoMambaCore(nn.Module):
    """
    Minimal recurrent core:
        h_{t+1} = f(x_t, h_t)
    currently using the extension op for f.

    You will usually wrap this inside a higher-level block that
    does input/output projections, gating, etc.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, x_t: torch.Tensor, h_t: torch.Tensor) -> torch.Tensor:
        """
        x_t, h_t: [B, D], with D = hidden_dim
        """
        return _PseudoMambaCoreFn.apply(x_t, h_t)
