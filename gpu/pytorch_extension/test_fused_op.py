"""Checks and times the topnotch::scale_add_relu extension.

  1. torch.library.opcheck: the operator is registered correctly
  2. forward and backward match eager PyTorch
  3. speed: one fused kernel against eager mul + add + relu (three kernels)

Run from this folder after building:  python test_fused_op.py
"""
import torch
from torch.profiler import ProfilerActivity, profile

import fused_ext

dev = torch.device("cuda")
print(f"torch {torch.__version__} on {torch.cuda.get_device_name(0)}")


def eager(x, y, s):
    return torch.relu(x * s + y)


# ---- 1 ----
x = torch.randn(1000, device=dev, requires_grad=True)
y = torch.randn(1000, device=dev, requires_grad=True)
torch.library.opcheck(torch.ops.topnotch.scale_add_relu.default, (x, y, 0.5))
print("1. opcheck passed")

# ---- 2 ----
out = fused_ext.scale_add_relu(x, y, 0.5)
ref = eager(x, y, 0.5)
g = torch.randn_like(out)
gx, gy = torch.autograd.grad(out, (x, y), g)
rx, ry = torch.autograd.grad(ref, (x, y), g)
print("2. forward max error {:.1e}, grad x {:.1e}, grad y {:.1e}".format(
    (out - ref).abs().max().item(), (gx - rx).abs().max().item(),
    (gy - ry).abs().max().item()))


# ---- 3 ----
def gpu_ms(fn, reps=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(reps):
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    return sorted(ts)[len(ts) // 2]


def kernels(fn):
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        fn()
        torch.cuda.synchronize()
    return sum(1 for e in p.events() if e.device_type.name == "CUDA")


print("3. relu(x * s + y), forward only")
with torch.no_grad():
    for n in (1 << 12, 1 << 24):
        x = torch.randn(n, device=dev)
        y = torch.randn(n, device=dev)
        te = gpu_ms(lambda: eager(x, y, 0.5))
        tf = gpu_ms(lambda: fused_ext.scale_add_relu(x, y, 0.5))
        print(f"   {n:>9,} floats: eager {te * 1e3:8.1f} us ({kernels(lambda: eager(x, y, 0.5))} kernels)"
              f"   fused {tf * 1e3:8.1f} us ({kernels(lambda: fused_ext.scale_add_relu(x, y, 0.5))} kernel)"
              f"   {te / tf:.2f}x")
