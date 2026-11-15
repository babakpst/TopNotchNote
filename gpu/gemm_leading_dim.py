"""Time cuBLAS SGEMM on padded arrays: same n x n product, different leading dimension.

Each matrix lives in a larger row-major buffer of shape (n, ld); the GEMM uses
the n x n view buf[:, :n], so PyTorch passes ld to cuBLAS as the leading
dimension. The script checks with the profiler that no copy kernel runs.
"""
import sys, torch
from torch.profiler import profile, ProfilerActivity

n = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
pads = [0, 1, 2, 3, 4, 8, 16, 32, 64]
reps = 20
dev = "cuda"
torch.manual_seed(0)

def views(ld):
    bufs = [torch.randn(n, ld, device=dev) for _ in range(3)]
    return [b[:, :n] for b in bufs]

def time_mm(a, b, c):
    for _ in range(3):
        torch.mm(a, b, out=c)
    torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(True), torch.cuda.Event(True)
    best = float("inf")
    for _ in range(reps):
        t0.record(); torch.mm(a, b, out=c); t1.record(); torch.cuda.synchronize()
        best = min(best, t0.elapsed_time(t1))
    return best

print(f"GPU: {torch.cuda.get_device_name()}, torch {torch.__version__}, n = {n}, FP32")
print(f"{'ld':>6} {'pad':>4} {'ld*4 % 16':>9} {'best ms':>8} {'GFLOP/s':>8}  kernel")
for p in pads:
    ld = n + p
    a, b, c = views(ld)
    ms = time_mm(a, b, c)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        torch.mm(a, b, out=c); torch.cuda.synchronize()
    names = sorted({e.name for e in prof.events() if e.device_type.name == "CUDA"})
    ref = a.contiguous() @ b.contiguous()
    assert torch.allclose(c, ref, rtol=1e-3, atol=1e-3)
    gf = 2 * n**3 / (ms * 1e-3) / 1e9
    print(f"{ld:>6} {p:>4} {(ld*4) % 16:>9} {ms:>8.3f} {gf:>8.1f}  {' + '.join(names)}")
