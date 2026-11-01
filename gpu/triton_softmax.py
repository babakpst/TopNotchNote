"""triton_softmax.py
Companion example for: topnotchnote.com/gpu/triton_softmax.html

A row-wise softmax written as one Triton kernel: each program loads one row,
computes max, exp, sum and divide in registers, and writes the row once.

  1. correctness against torch.softmax, on odd row lengths that need masking
  2. naive softmax (five PyTorch ops) against torch.softmax (one fused
     kernel): what fusing softmax is worth, on any GPU
  3. the Triton kernel against both, on GPUs Triton supports

Triton supports Nvidia GPUs of compute capability 8.0+ and AMD GPUs with
ROCm 6.2+. On anything else, run step 1 in Triton's interpreter, which
executes the kernel on the CPU:

    TRITON_INTERPRET=1 python triton_softmax.py

(With Triton 3.1 the interpreter's stores silently did nothing under NumPy
2.4; with NumPy 1.26 it worked. If every error below is about 1.0, check
your NumPy version.)

Run on a supported GPU:  python triton_softmax.py
"""
import os

import torch
import triton
import triton.language as tl

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"


@triton.jit
def softmax_kernel(out_ptr, in_ptr, in_stride, out_stride, n_cols,
                   BLOCK: tl.constexpr):
    row = tl.program_id(0)                      # one program per row
    cols = tl.arange(0, BLOCK)                  # BLOCK is a power of two >= n_cols
    mask = cols < n_cols
    x = tl.load(in_ptr + row * in_stride + cols, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)                   # subtract the max: no overflow in exp
    num = tl.exp(x)                             # padded lanes: exp(-inf) = 0
    y = num / tl.sum(num, axis=0)
    tl.store(out_ptr + row * out_stride + cols, y, mask=mask)


def softmax(x):
    rows, cols = x.shape
    out = torch.empty_like(x)
    softmax_kernel[(rows,)](out, x, x.stride(0), out.stride(0), cols,
                            BLOCK=triton.next_power_of_2(cols))
    return out


def naive_softmax(x):
    m = x.max(dim=1, keepdim=True).values      # read x, write m
    z = x - m                                  # read x and m, write z
    e = torch.exp(z)                           # read z, write e
    s = e.sum(dim=1, keepdim=True)             # read e, write s
    return e / s                               # read e and s, write the result


def triton_gpu_ok():
    if not torch.cuda.is_available():
        return False
    if torch.version.hip:
        return True
    return torch.cuda.get_device_capability(0) >= (8, 0)


# ---- 1. correctness -------------------------------------------------------
dev_k = "cpu" if INTERPRET else "cuda"
if INTERPRET or triton_gpu_ok():
    print(f"1. Triton kernel vs torch.softmax ({'interpreter, CPU' if INTERPRET else 'GPU'})")
    torch.manual_seed(0)
    for rows, cols in [(4, 7), (64, 100), (33, 781), (8, 1024)]:
        x = torch.randn(rows, cols, device=dev_k) * 10
        err = (softmax(x) - torch.softmax(x, dim=1)).abs().max().item()
        print(f"   {rows:>3} x {cols:<5} max abs error {err:.1e}")
else:
    print("1. skipped: this GPU is below Triton's minimum; rerun with TRITON_INTERPRET=1")

if INTERPRET or not torch.cuda.is_available():
    raise SystemExit

# ---- 2 and 3. speed ------------------------------------------------------
def gbps(fn, x, reps=50):
    for _ in range(5):
        fn(x)
    torch.cuda.synchronize()
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(reps):
        a.record()
        fn(x)
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    ms = sorted(ts)[len(ts) // 2]
    return 2 * x.numel() * x.element_size() / ms / 1e6, ms  # ideal: read once, write once


print(f"\n2./3. softmax of 4096 rows, float32, on {torch.cuda.get_device_name(0)}")
print("   effective GB/s = (bytes of input + output) / time")
print(f"   {'cols':>6} {'naive':>9} {'torch.softmax':>14} {'triton':>9}")
for cols in (256, 1024, 4096):
    x = torch.randn(4096, cols, device="cuda")
    n, _ = gbps(naive_softmax, x)
    t, _ = gbps(lambda v: torch.softmax(v, dim=1), x)
    tr = f"{gbps(softmax, x)[0]:9.1f}" if triton_gpu_ok() else f"{'n/a':>9}"
    print(f"   {cols:>6} {n:9.1f} {t:14.1f} {tr}")
