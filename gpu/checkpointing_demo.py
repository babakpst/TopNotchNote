"""checkpointing_demo.py

Companion example for: topnotchnote.com/gpu/gradient_checkpointing.html

Trains a stack of residual MLP blocks for a few steps three ways and prints
the peak GPU memory and the time per training step:
  1. no checkpointing: every activation is kept until the backward pass
  2. checkpoint every block: only each block's input is kept; the inside of
     each block is recomputed during the backward pass
  3. checkpoint_sequential with a few segments: only segment boundaries are
     kept
For each, it also compares one weight gradient with the plain run's, to show
that checkpointing does not change the result.

Run:  python3 checkpointing_demo.py      (needs PyTorch with a CUDA or ROCm GPU)
"""
import time

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

TOKENS, HIDDEN, FFN, BLOCKS = 1024, 1024, 4096, 12


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Linear(HIDDEN, FFN)
        self.act = nn.GELU()
        self.down = nn.Linear(FFN, HIDDEN)

    def forward(self, x):
        return x + self.down(self.act(self.up(x)))


class Model(nn.Module):
    def __init__(self, mode, segments=4):
        super().__init__()
        self.blocks = nn.Sequential(*[Block() for _ in range(BLOCKS)])
        self.mode, self.segments = mode, segments

    def forward(self, x):
        if self.mode == "none":
            return self.blocks(x)
        if self.mode == "every_block":
            for b in self.blocks:
                x = checkpoint(b, x, use_reentrant=False)
            return x
        return checkpoint_sequential(self.blocks, self.segments, x, use_reentrant=False)


def run(mode, segments=4, steps=6):
    torch.manual_seed(0)
    dev = "cuda"
    model = Model(mode, segments).to(dev)
    x = torch.randn(TOKENS, HIDDEN, device=dev, requires_grad=True)
    model(x).square().mean().backward()       # warm-up; also allocates the gradients
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()      # weights + gradients + input
    times = []
    for _ in range(steps):
        model.zero_grad(set_to_none=False)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss = model(x).square().mean()
        loss.backward()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    peak = torch.cuda.max_memory_allocated() - base
    grad = model.blocks[0].up.weight.grad.detach().clone()
    times.sort()
    del model, x
    torch.cuda.empty_cache()
    return peak / 2**20, times[len(times) // 2] * 1e3, grad


def main():
    print(f"{BLOCKS} blocks, {TOKENS} tokens x {HIDDEN} hidden, FFN {FFN}, fp32, "
          f"{torch.cuda.get_device_name()}")
    print("peak = memory used during a step above the weights, gradients and input,")
    print("i.e. mostly the saved activations\n")
    rows = [("no checkpointing", "none", 0),
            ("checkpoint every block", "every_block", 0),
            ("checkpoint_sequential, 4 segments", "seq", 4),
            ("checkpoint_sequential, 2 segments", "seq", 2)]
    ref = None
    for name, mode, seg in rows:
        peak, ms, grad = run(mode, seg or 4)
        if ref is None:
            ref = grad
        diff = (grad - ref).abs().max().item()
        print(f"  {name:36s} peak +{peak:7.0f} MiB   {ms:7.1f} ms/step   "
              f"max |grad diff| {diff:.1e}")


if __name__ == "__main__":
    main()
