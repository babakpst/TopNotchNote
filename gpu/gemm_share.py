"""gemm_share.py
Companion example for: topnotchnote.com/gpu/what_is_gemm.html

Builds one decoder layer with Llama 3's proportions, scaled down to fit a small
GPU (model dim 1024 instead of 4096, same FFN ratio of 3.5, same 4:1 ratio of
query heads to key/value heads, head dimension 128), runs a forward pass, and
asks the PyTorch profiler how GPU time splits between matrix-multiply kernels
(GEMMs) and everything else (norms, softmax, RoPE, activations, adds, copies).

Every kernel name is printed with its class, so you can check the split yourself.

Run:  python gemm_share.py            (needs PyTorch with a CUDA or ROCm GPU)
"""
import re
import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity

torch.manual_seed(0)
dev = "cuda"
dtype = torch.float32          # FP32 so older GPUs run it; use bfloat16 on new ones

D, FFN, HEADS, KV_HEADS = 1024, 3584, 8, 2
HD = D // HEADS                # 128, as in Llama 3


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        lin = lambda i, o: torch.nn.Linear(i, o, bias=False)
        self.q, self.k, self.v, self.o = lin(D, D), lin(D, KV_HEADS * HD), lin(D, KV_HEADS * HD), lin(D, D)
        self.gate, self.up, self.down = lin(D, FFN), lin(D, FFN), lin(FFN, D)
        self.n1 = torch.nn.RMSNorm(D)
        self.n2 = torch.nn.RMSNorm(D)

    @staticmethod
    def rope(x, pos):
        half = x.shape[-1] // 2
        freq = 1.0 / (500000 ** (torch.arange(half, device=x.device, dtype=x.dtype) / half))
        ang = pos[:, None] * freq[None, :]
        cos, sin = ang.cos(), ang.sin()
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    def forward(self, x):
        b, s, _ = x.shape
        pos = torch.arange(s, device=x.device, dtype=x.dtype)
        h = self.n1(x)
        q = self.q(h).view(b, s, HEADS, HD).transpose(1, 2)
        k = self.k(h).view(b, s, KV_HEADS, HD).transpose(1, 2)
        v = self.v(h).view(b, s, KV_HEADS, HD).transpose(1, 2)
        q, k = self.rope(q, pos), self.rope(k, pos)
        rep = HEADS // KV_HEADS                      # grouped-query attention
        k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
        scores = (q @ k.transpose(-1, -2)) / HD ** 0.5          # GEMM
        mask = torch.ones(s, s, device=x.device, dtype=torch.bool).triu(1)
        p = scores.masked_fill(mask, float("-inf")).softmax(-1)
        a = (p @ v).transpose(1, 2).reshape(b, s, D)            # GEMM
        x = x + self.o(a)
        h = self.n2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))  # SwiGLU: 3 GEMMs


GEMM_RE = re.compile(r"gemm|gemv|sgemm|hgemm|cutlass|cublas|matmul|mfma|tensile|Cijk", re.I)


def run(seq):
    layer = Layer().to(dev, dtype)
    x = torch.randn(1, seq, D, device=dev, dtype=dtype)
    with torch.no_grad():
        for _ in range(3):
            layer(x)                                  # warm-up: cuBLAS init, autotuning
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(5):
                layer(x)
            torch.cuda.synchronize()
    gemm = other = 0.0
    names = {}
    for e in prof.key_averages():
        t = e.device_time_total if hasattr(e, "device_time_total") else e.cuda_time_total
        if t <= 0 or e.key.startswith(("cuda", "hip", "Memcpy", "Memset")) and "Kernel" not in e.key and "kernel" not in e.key:
            continue
        is_gemm = bool(GEMM_RE.search(e.key))
        names[e.key] = (is_gemm, t)
        if is_gemm:
            gemm += t
        else:
            other += t
    return gemm, other, names


if __name__ == "__main__":
    print(torch.cuda.get_device_name(0), "| torch", torch.__version__, "| dtype", dtype)
    for seq in (256, 1024, 2048):
        gemm, other, names = run(seq)
        total = gemm + other
        print(f"\nsequence length {seq}: GEMM kernels {100 * gemm / total:.1f}% of GPU time "
              f"({gemm / 5 / 1000:.2f} ms of {total / 5 / 1000:.2f} ms per layer)")
        for k, (g, t) in sorted(names.items(), key=lambda kv: -kv[1][1])[:12]:
            print(f"   {'GEMM ' if g else 'other'} {100 * t / total:5.1f}%  {k[:90]}")
