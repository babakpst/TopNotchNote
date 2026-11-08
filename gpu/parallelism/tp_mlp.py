# tp_mlp.py
# Companion example for: topnotchnote.com/gpu/parallelism_strategies.html
#
# Tensor parallelism, Megatron-style, on one transformer MLP block:
#   Y = GeLU(X A) B
# A is split by COLUMNS across ranks, so each rank computes its own slice of
# GeLU(X A) with no communication. B is split by ROWS, so each rank produces a
# partial Y; one all-reduce (sum) gives the full Y on every rank.
# That is the "one all-reduce per MLP in the forward pass" from the Megatron
# paper; attention is split the same way, for two per transformer layer.
#
# The script runs N ranks as CPU processes with torch.distributed's gloo
# backend, so no GPU is needed. It checks the sharded result against the
# unsharded layer, and reports how many bytes each all-reduce moved, and the
# per-token message size and all-reduce count for a real model.
#
# Run:  python tp_mlp.py            (2 ranks)
#       python tp_mlp.py 4          (4 ranks)

import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

TOKENS, HIDDEN, FFN = 16, 1024, 4096  # a small MLP block


def run(rank, world, X, A, B, Y_ref):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29512")
    dist.init_process_group("gloo", rank=rank, world_size=world)

    cols = FFN // world
    A_shard = A[:, rank * cols:(rank + 1) * cols]   # column-parallel
    B_shard = B[rank * cols:(rank + 1) * cols, :]   # row-parallel

    H_local = torch.nn.functional.gelu(X @ A_shard)  # no communication
    Y_partial = H_local @ B_shard                     # partial sum of Y
    dist.all_reduce(Y_partial, op=dist.ReduceOp.SUM)  # the one all-reduce

    if rank == 0:
        err = (Y_partial - Y_ref).abs().max().item()
        print(f"{world} ranks: each holds {A_shard.numel() + B_shard.numel():,} of "
              f"{A.numel() + B.numel():,} weights ({100 / world:.0f}%)")
        print(f"  max |sharded - unsharded| = {err:.2e}")
        print(f"  all-reduce payload: {Y_partial.numel() * 4:,} bytes "
              f"({TOKENS} tokens x {HIDDEN} hidden x 4 bytes)")
    dist.destroy_process_group()


if __name__ == "__main__":
    world = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    assert FFN % world == 0
    torch.manual_seed(0)
    X = torch.randn(TOKENS, HIDDEN)
    A = torch.randn(HIDDEN, FFN) / HIDDEN ** 0.5
    B = torch.randn(FFN, HIDDEN) / FFN ** 0.5
    Y_ref = torch.nn.functional.gelu(X @ A) @ B
    mp.spawn(run, args=(world, X, A, B, Y_ref), nprocs=world, join=True)

    # The same arithmetic for Llama 3 70B (80 layers, hidden size 8192), in BF16.
    layers, hidden = 80, 8192
    print(f"\nLlama 3 70B with tensor parallelism, per decode step:")
    print(f"  all-reduces in one forward pass: {layers} layers x 2 = {2 * layers}")
    print(f"  payload per all-reduce, per token in the batch: {hidden} x 2 bytes = "
          f"{hidden * 2 / 1024:.0f} KiB")
