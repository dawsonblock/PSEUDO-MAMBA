import torch
import pseudo_mamba_ext

class PseudoMambaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, h):
        # x: [B, D]
        # h: [B, D]
        # y = tanh(x + h)
        # We assume the C++ extension implements this.
        # If the extension implements "forward_with_state", we use that.
        # Let's assume the extension exposes `forward(x, h)` returning `y`.
        
        output = pseudo_mamba_ext.forward(x, h)
        ctx.save_for_backward(x, h, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, h, output = ctx.saved_tensors
        
        # dy/dx = 1 - tanh^2(x+h) = 1 - y^2
        # dy/dh = 1 - y^2
        
        # grad_x = grad_output * (1 - y^2)
        # grad_h = grad_output * (1 - y^2)
        
        # We can implement backward in CUDA too, but for now let's do it in Python 
        # or assume the extension has backward.
        # If we use torch.autograd.Function, we define backward here.
        
        dtanh = 1 - output.pow(2)
        grad_x = grad_output * dtanh
        grad_h = grad_output * dtanh
        
        return grad_x, grad_h

def pseudo_mamba_forward(x, h):
    return PseudoMambaFunction.apply(x, h)
