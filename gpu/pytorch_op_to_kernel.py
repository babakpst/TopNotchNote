"""pytorch_op_to_kernel.py
Companion example for: topnotchnote.com/gpu/pytorch_op_to_kernel.html

What happens between `c = a + b` in Python and a kernel running on the GPU:
  1. the profiler's view of one add: the aten operator, the runtime launch
     call, and the GPU kernel it started
  2. the CPU cost of one small op, against the GPU time of its kernel
  3. the same op with autograd recording (requires_grad=True), which adds a
     step before the CUDA kernel is reached
  4. launches are asynchronous: a large matmul "returns" long before the GPU
     finishes it

Runs unchanged on ROCm builds of PyTorch, which reuse the torch.cuda API
(the device is still called "cuda"); there the launch call is hipLaunchKernel.

Run:  python pytorch_op_to_kernel.py
"""
import time

import torch
from torch.profiler import ProfilerActivity, profile

assert torch.cuda.is_available()
dev = torch.device("cuda")
backend = f"HIP {torch.version.hip}" if torch.version.hip else f"CUDA {torch.version.cuda}"
print(f"torch {torch.__version__} ({backend}) on {torch.cuda.get_device_name(0)}")

# ---- 1. one add, as the profiler sees it --------------------------------
a = torch.randn(4096, device=dev)
b = torch.randn(4096, device=dev)
for _ in range(10):
    a + b  # warm-up
torch.cuda.synchronize()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    c = a + b
    torch.cuda.synchronize()

print("\n1. one `a + b` on 4,096 floats")
for e in prof.events():
    if e.name.startswith(("aten::", "cuda", "hip")) or e.device_type.name == "CUDA":
        where = "GPU" if e.device_type.name == "CUDA" else "CPU"
        print(f"   {where}  {e.name[:90]}")


# ---- 2 and 3. CPU cost per op --------------------------------------------
def per_op_us(x, y, n=20000):
    for _ in range(100):
        x + y
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        x + y
    t_launch = (time.perf_counter() - t0) / n * 1e6  # CPU side only
    torch.cuda.synchronize()
    t_total = (time.perf_counter() - t0) / n * 1e6
    return t_launch, t_total


def kernel_us(x, y, n=200):
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(n):
            x + y
        torch.cuda.synchronize()
    ks = [e for e in p.events() if e.device_type.name == "CUDA"]
    return sum(e.device_time for e in ks) / len(ks)


print("\n2. CPU time per `a + b` (4,096 floats) against the kernel's GPU time")
launch, total = per_op_us(a, b)
print(f"   CPU, issuing the op:          {launch:6.1f} us")
print(f"   loop time per op incl. sync:  {total:6.1f} us")
print(f"   GPU, the kernel itself:       {kernel_us(a, b):6.1f} us")

print("\n3. the same op with autograd recording")
ag = a.clone().requires_grad_(True)
launch_ag, _ = per_op_us(ag, b)
with torch.no_grad():
    launch_ng, _ = per_op_us(ag, b)
print(f"   requires_grad=True:           {launch_ag:6.1f} us per op")
print(f"   same tensors, torch.no_grad(): {launch_ng:5.1f} us per op")
out = ag + b
print(f"   output carries grad_fn = {type(out.grad_fn).__name__}")

# ---- 4. asynchronous launch ----------------------------------------------
m = torch.randn(2048, 2048, device=dev)
m @ m
torch.cuda.synchronize()
t0 = time.perf_counter()
p = m @ m
t_return = (time.perf_counter() - t0) * 1e3
torch.cuda.synchronize()
t_done = (time.perf_counter() - t0) * 1e3
print("\n4. one 2048 x 2048 matmul")
print(f"   Python got the result tensor after {t_return:7.3f} ms")
print(f"   the GPU finished it after          {t_done:7.3f} ms")
