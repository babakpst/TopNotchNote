"""straggler_demo.py

Companion example for: topnotchnote.com/gpu/distributed_profiling.html

Four processes run a data-parallel style loop: "compute" for a while, then
all-reduce a 16 MB tensor. One rank is made 30% slower. Each rank times its
own compute and its own all-reduce, and rank 0 prints the table.

The signature of a straggler: the slow rank shows the longest compute and the
SHORTEST all-reduce, while every other rank shows a long all-reduce, because
an all-reduce cannot finish until the last rank arrives. Time spent waiting
for the straggler is reported as communication time on the healthy ranks.

To keep the demo about timing and runnable anywhere, "compute" is a sleep and
the collective uses the gloo backend on the CPU. With GPUs, use backend
"nccl" (NCCL on NVIDIA, RCCL on ROCm) and real work; the pattern is the same.

Run:  python3 straggler_demo.py
"""
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD, STEPS, BASE_MS, SLOW_RANK, SLOW_FACTOR = 4, 10, 40.0, 2, 1.3


def worker(rank, slow_factor, q):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29533")
    dist.init_process_group("gloo", rank=rank, world_size=WORLD)
    grad = torch.ones(4 << 20)                        # 16 MB
    comp, comm = [], []
    for step in range(STEPS + 2):
        t0 = time.perf_counter()
        f = slow_factor if rank == SLOW_RANK else 1.0
        time.sleep(BASE_MS * f / 1e3)                 # stand-in for this rank's compute
        t1 = time.perf_counter()
        dist.all_reduce(grad)
        t2 = time.perf_counter()
        if step >= 2:                                 # skip warm-up steps
            comp.append((t1 - t0) * 1e3)
            comm.append((t2 - t1) * 1e3)
    comp.sort(); comm.sort()
    q.put((rank, comp[len(comp) // 2], comm[len(comm) // 2]))
    dist.destroy_process_group()


def run(slow_factor, title):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, slow_factor, q)) for r in range(WORLD)]
    for p in procs: p.start()
    rows = sorted(q.get() for _ in range(WORLD))
    for p in procs: p.join()
    print(f"\n{title}")
    print("  rank   compute (ms)   all-reduce (ms)   step (ms)")
    for r, c, m in rows:
        tag = "  <- slow rank" if (slow_factor != 1.0 and r == SLOW_RANK) else ""
        print(f"  {r:4d}   {c:12.1f}   {m:15.1f}   {c + m:9.1f}{tag}")


if __name__ == "__main__":
    print(f"{WORLD} ranks, gloo on CPU, 16 MB all-reduce, median of {STEPS} steps")
    run(1.0, "balanced: every rank computes for 40 ms")
    run(SLOW_FACTOR, f"rank {SLOW_RANK} is {int((SLOW_FACTOR - 1) * 100)}% slower")
