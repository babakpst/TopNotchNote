"""preshuffle_layout.py

Companion example for: topnotchnote.com/gpu/preshuffling.html

Weight preshuffling rearranges a GEMM weight matrix B (N x K, K contiguous)
once, offline, into the order a matrix-instruction kernel reads it. This
script reproduces the index permutation used by two AMD libraries, in NumPy,
and shows what it does to a small matrix of 2-byte elements (as fp16):

  * AITER's shuffle_weight(x, layout=(16, 16))   (aiter/ops/shuffle.py)
  * Composable Kernel's preShuffleWeight          (example/ck_tile/18_flatmm)

Both split N into (N0, NLane=16) and K into (K0, KLane=4, KPack), then store
the data in the order (N0, K0, KLane, NLane, KPack). For fp16, KPack is 8
elements = 16 bytes. The script checks that the two formulas agree, that the
permutation is invertible, and prints where each 16-byte chunk of the first
block came from.

No GPU needed:  python3 preshuffle_layout.py
"""
import numpy as np

NLANE, KLANE = 16, 4


def aiter_shuffle_weight(x: np.ndarray) -> np.ndarray:
    """NumPy transcription of aiter.ops.shuffle.shuffle_weight, layout=(16, 16),
    for 2-byte types: BN = 16, BK = 16 * 2 = 32 elements, K = 16 // 2 = 8."""
    n, k = x.shape
    bn, bk, kpack = 16, 32, 16 // x.itemsize
    v = x.reshape(n // bn, bn, k // bk, bk // kpack, kpack)
    return v.transpose(0, 2, 3, 1, 4).copy().reshape(n, k)


def ck_preshuffle(x: np.ndarray, kpack: int) -> np.ndarray:
    """Composable Kernel's preShuffleWeight index formula (flatmm example),
    written element by element."""
    n_total, k_total = x.shape
    k0_total = k_total // (KLANE * kpack)
    out = np.empty(x.size, dtype=x.dtype)
    for n in range(n_total):
        n0, n1 = divmod(n, NLANE)
        for k in range(k_total):
            k0, rest = divmod(k, KLANE * kpack)
            k1, k2 = divmod(rest, kpack)
            idx = (n0 * kpack * NLANE * KLANE * k0_total + k0 * kpack * NLANE * KLANE
                   + k1 * kpack * NLANE + n1 * kpack + k2)
            out[idx] = x[n, k]
    return out.reshape(n_total, k_total)


def main() -> None:
    n, k = 32, 64                           # two N0 blocks, two K0 blocks
    # A 2-byte type, like fp16. int16 is used so that each value can hold its
    # own (row, col) exactly and we can see where it ended up.
    w = (np.arange(n)[:, None] * 1000 + np.arange(k)[None, :]).astype(np.int16)
    kpack = 16 // w.itemsize                # 8 two-byte elements = 16 bytes

    a = aiter_shuffle_weight(w)
    c = ck_preshuffle(w, kpack)
    print("AITER and CK orders identical:", np.array_equal(a, c))

    flat = a.reshape(-1)
    print("permutation (no value lost or repeated):",
          np.array_equal(np.sort(flat), np.sort(w.reshape(-1))))

    # The first N0 x K0 block is 16 rows x 32 columns of fp16 = 1 KB,
    # stored as 64 consecutive 16-byte chunks.
    print("\nfirst 1 KB of the preshuffled buffer, one 16-byte chunk per line:")
    print(" chunk  source row  source columns")
    for chunk in range(KLANE * NLANE):
        vals = flat[chunk * kpack:(chunk + 1) * kpack].astype(np.int64)
        rows = set(int(v) // 1000 for v in vals)
        cols = [int(v) % 1000 for v in vals]
        if chunk < 18 or chunk >= 62:
            print(f"  {chunk:4d}  {rows.pop():10d}  {cols[0]}..{cols[-1]}")
        elif chunk == 18:
            print("   ...")
    # In the original row-major matrix the same 64 chunks are spread over 16
    # rows, k * 2 = 128 bytes apart.
    print(f"\nin the original layout these 64 chunks sit in {NLANE} rows, "
          f"{k * w.itemsize} bytes apart;")
    print("after preshuffling they are one contiguous 1,024-byte run.")


if __name__ == "__main__":
    main()
