"""pytorch_profiler_demo.py

Companion example for: topnotchnote.com/gpu/pytorch_profiler.html

Profiles a small training loop with torch.profiler, using a schedule so only
a few steady-state steps are recorded, record_function labels for each phase,
and a trace file for a timeline viewer. The loop logs the loss with .item()
every step, which makes the CPU wait for the GPU; the second run accumulates
the loss on the GPU and reads it once at the end.

Prints, for each run:
  * the top operators by self device (GPU) time
  * the time spent in each labelled phase
  * the CPU-side synchronization calls and how long they took
and writes trace_<run>.json for the chrome://tracing viewer.

Run:  python3 pytorch_profiler_demo.py       (PyTorch with a CUDA or ROCm GPU)
"""
import time

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile, record_function, schedule

DEV = "cuda"
STEPS = 12


def make():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(1024, 4096), nn.GELU(), nn.Linear(4096, 1024),
                          nn.LayerNorm(1024), nn.Linear(1024, 10)).to(DEV)
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    x = torch.randn(512, 1024, device=DEV)
    y = torch.randint(0, 10, (512,), device=DEV)
    return model, opt, x, y


def train(log_every_step, prof=None):
    model, opt, x, y = make()
    loss_fn = nn.CrossEntropyLoss()
    running = torch.zeros((), device=DEV)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(STEPS):
        with record_function("forward"):
            loss = loss_fn(model(x), y)
        with record_function("backward"):
            opt.zero_grad(set_to_none=True)
            loss.backward()
        with record_function("optimizer"):
            opt.step()
        with record_function("log"):
            if log_every_step:
                _ = loss.item()                  # CPU waits for the GPU here
            else:
                running += loss.detach()         # stays on the GPU
        if prof is not None:
            prof.step()
    if not log_every_step:
        _ = running.item()                       # one wait, at the end
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / STEPS * 1e3


def report(name, ka):
    print(f"\n=== {name}: top 6 operators by self GPU time ===")
    print(ka.table(sort_by="self_cuda_time_total", row_limit=6))
    print(f"=== {name}: labelled phases and synchronization calls (CPU time) ===")
    for e in ka:
        if e.key in ("forward", "backward", "optimizer", "log") or "Synchronize" in e.key:
            print(f"  {e.key:28s} calls {e.count:3d}   CPU total {e.cpu_time_total / 1e3:8.2f} ms")


def main():
    print(torch.cuda.get_device_name())
    train(True)                                   # warm-up, outside the profiler
    for name, every in [("log .item() every step", True), ("accumulate on GPU", False)]:
        plain_ms = train(every)                   # timed without the profiler
        fname = "trace_" + ("item" if every else "accum") + ".json"
        captured = {}

        def on_ready(p):                          # called when the active steps end
            captured["ka"] = p.key_averages()
            p.export_chrome_trace(fname)

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     schedule=schedule(wait=2, warmup=2, active=4, repeat=1),
                     on_trace_ready=on_ready, record_shapes=True) as prof:
            train(every, prof)
        report(name, captured["ka"])
        print(f"  step time without profiler: {plain_ms:.2f} ms   trace: {fname}")


if __name__ == "__main__":
    main()
