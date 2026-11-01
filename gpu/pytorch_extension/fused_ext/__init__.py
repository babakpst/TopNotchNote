"""Python side of the topnotch::scale_add_relu operator.

Importing this package loads the compiled library (which defines the op and
its CUDA kernel), then adds what is easier to write in Python: a fake kernel
for torch.compile and an autograd formula.
"""
import torch

from . import _C  # noqa: F401  (loads the library, registers the op)


@torch.library.register_fake("topnotch::scale_add_relu")
def _(x, y, s):
    torch._check(x.shape == y.shape)
    return torch.empty_like(x)


def _setup_context(ctx, inputs, output):
    x, y, s = inputs
    ctx.s = s
    ctx.save_for_backward(output)


def _backward(ctx, grad):
    (out,) = ctx.saved_tensors
    g = grad * (out > 0)  # relu passes the gradient where the output is > 0
    return g * ctx.s, g, None  # d/dx, d/dy, nothing for the float s


torch.library.register_autograd(
    "topnotch::scale_add_relu", _backward, setup_context=_setup_context)


def scale_add_relu(x, y, s):
    """relu(x * s + y) in one CUDA kernel."""
    return torch.ops.topnotch.scale_add_relu.default(x, y, s)
