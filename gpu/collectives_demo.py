"""collectives_demo.py
Companion example for: topnotchnote.com/gpu/nccl_rccl.html

What the collectives behind multi-GPU training do, and what the ring
all-reduce costs, run with 4 processes on the CPU so it works on any machine:
  1. all_reduce, all_gather, broadcast and reduce_scatter on small tensors,
     printed rank by rank
  2. a ring all-reduce written by hand from send/recv: each rank sends
     2(n-1) chunks of size S/n, so 2(n-1)/n * S bytes, which the script
     counts and compares with the formula nccl-tests uses for "bus bandwidth"

It uses the gloo backend so it needs no GPU. On GPUs, the same calls with
backend="nccl" go through NCCL on Nvidia and RCCL on AMD.

Run:  python collectives_demo.py
"""
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 4


def show(rank, label, t):
    # print in rank order
    for r in range(WORLD):
        dist.barrier()
        if r == rank:
            print(f"   rank {rank} {label}: {t.tolist()}", flush=True)
    dist.barrier()


def ring_all_reduce(x):
    """Sum x across ranks with a ring: reduce-scatter, then all-gather."""
    rank, n = dist.get_rank(), dist.get_world_size()
    chunks = list(x.clone().chunk(n))
    right, left = (rank + 1) % n, (rank - 1) % n
    sent = 0
    # reduce-scatter: after n-1 steps, rank r owns the full sum of chunk r+1
    for step in range(n - 1):
        send_i = (rank - step) % n
        recv_i = (rank - step - 1) % n
        buf = torch.empty_like(chunks[recv_i])
        req = dist.isend(chunks[send_i], right)
        dist.recv(buf, left)
        req.wait()
        sent += chunks[send_i].numel() * chunks[send_i].element_size()
        chunks[recv_i] += buf
    # all-gather: pass the finished chunks around the ring
    for step in range(n - 1):
        send_i = (rank - step + 1) % n
        recv_i = (rank - step) % n
        buf = torch.empty_like(chunks[recv_i])
        req = dist.isend(chunks[send_i], right)
        dist.recv(buf, left)
        req.wait()
        sent += chunks[send_i].numel() * chunks[send_i].element_size()
        chunks[recv_i] = buf
    return torch.cat(chunks), sent


def worker(rank):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29517"
    dist.init_process_group("gloo", rank=rank, world_size=WORLD)
    torch.set_num_threads(1)

    if rank == 0:
        print(f"1. collectives over {WORLD} ranks; rank r starts with [r+1, r+1]")
    x = torch.full((2,), float(rank + 1))

    y = x.clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    show(rank, "all_reduce(sum)   ", y)

    parts = [torch.empty(2) for _ in range(WORLD)]
    dist.all_gather(parts, x)
    show(rank, "all_gather        ", torch.cat(parts))

    b = x.clone()
    dist.broadcast(b, src=0)
    show(rank, "broadcast(from 0) ", b)

    # reduce_scatter: rank r gets the sum of everyone's chunk r
    full = torch.arange(WORLD * 2, dtype=torch.float32) + 10 * rank
    try:
        out = torch.empty(2)
        dist.reduce_scatter(out, list(full.chunk(WORLD)))
        show(rank, "reduce_scatter    ", out)
    except (RuntimeError, NotImplementedError):
        # gloo may not implement it; build it from all_reduce instead
        s = full.clone()
        dist.all_reduce(s)
        show(rank, "reduce_scatter*   ", s.chunk(WORLD)[rank])
        if rank == 0:
            print("   * this gloo build has no reduce_scatter; shown via all_reduce")

    # ---- 2 ----
    size = 1 << 20  # floats per rank: 4 MB
    data = torch.randn(size, generator=torch.Generator().manual_seed(rank))
    ref = data.clone()
    dist.all_reduce(ref)
    got, sent = ring_all_reduce(data)
    err = (got - ref).abs().max().item()
    S = size * 4
    expect = 2 * (WORLD - 1) / WORLD * S
    for r in range(WORLD):
        dist.barrier()
        if r == rank:
            if rank == 0:
                print(f"\n2. ring all-reduce of S = {S / 1e6:.1f} MB per rank, n = {WORLD}")
            print(f"   rank {rank}: sent {sent / 1e6:.2f} MB, 2(n-1)/n * S = "
                  f"{expect / 1e6:.2f} MB, max diff vs all_reduce {err:.1e}",
                  flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    if torch.cuda.is_available():
        nccl = torch.cuda.nccl.version()
        lib = "RCCL" if torch.version.hip else "NCCL"
        print(f"this PyTorch ships {lib} {'.'.join(map(str, nccl)) if isinstance(nccl, tuple) else nccl}\n")
    mp.spawn(worker, nprocs=WORLD)
