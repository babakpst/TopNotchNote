# jax_vs_eager.py
# Companion example for: topnotchnote.com/gpu/jax_gpu.html
#
# The same small chain of elementwise and reduction ops (a tanh GELU
# followed by a layer norm, over a 4096 x 4096 FP32 array), run three ways:
#   1. JAX without jit: every op is dispatched and launched on its own
#   2. JAX with jax.jit: traced once, compiled by XLA into fused kernels
#   3. PyTorch eager, if torch is installed: the default PyTorch model
# It also shows the three things that surprise people about jit: the
# first call pays for compilation, a new input shape compiles again, and
# timing without block_until_ready() measures only the dispatch.
#
# Run:  pip install -U "jax[cuda13]"        Nvidia, recent driver and GPU
#       pip install -U "jax[cuda12]"        Nvidia, older driver or GPU
#       pip install -U "jax[rocm7-local]"   AMD, with ROCm already installed
#       then  python jax_vs_eager.py
# Either half can be missing: the script skips JAX or PyTorch if it is not
# installed, so you can run it once in each environment.

import time

N = 4096
REPS = 20


def gelu_layernorm(xp, x):
    """tanh GELU followed by a layer norm over the last axis.

    xp is the array module (jax.numpy or torch), so the same arithmetic
    runs on both. Nine elementwise ops and two reductions.
    """
    y = 0.5 * x * (1.0 + xp.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
    mean = y.mean(-1, keepdims=True)
    var = ((y - mean) ** 2).mean(-1, keepdims=True)
    return (y - mean) / xp.sqrt(var + 1e-5)


def timed(fn, sync, reps=REPS):
    """Average milliseconds per call, synchronizing with the device."""
    sync(fn())  # warm-up
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn()
    sync(out)
    return (time.perf_counter() - t0) * 1e3 / reps


def run_jax():
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        print("JAX not installed, skipping\n")
        return None
    print(f"JAX {jax.__version__} on {jax.devices()[0]}")
    x = jax.random.normal(jax.random.PRNGKey(0), (N, N), dtype=jnp.float32)
    sync = lambda a: a.block_until_ready()

    # 1. Eager: no jit, each jnp op runs as its own compiled primitive.
    eager_ms = timed(lambda: gelu_layernorm(jnp, x), sync)

    # 2. jit. The first call traces the Python function and compiles it.
    f = jax.jit(lambda a: gelu_layernorm(jnp, a))
    t0 = time.perf_counter()
    sync(f(x))
    first_ms = (time.perf_counter() - t0) * 1e3
    jit_ms = timed(lambda: f(x), sync)

    # How many kernels did XLA emit? Count the fusions in the compiled HLO.
    hlo = f.lower(x).compile().as_text()
    entry = hlo[hlo.find("ENTRY"):]
    kernels = entry.count(" fusion(") + entry.count(" custom-call(")

    # A new shape is a new program: it compiles again on first use.
    x2 = x[:, : N // 2]
    t0 = time.perf_counter()
    sync(f(x2))
    reshape_ms = (time.perf_counter() - t0) * 1e3

    # Asynchronous dispatch: without block_until_ready() the clock stops
    # when the work is queued, not when it is done.
    t0 = time.perf_counter()
    for _ in range(REPS):
        out = f(x)
    nosync_ms = (time.perf_counter() - t0) * 1e3 / REPS
    sync(out)

    print(f"  eager (no jit)          {eager_ms:8.2f} ms")
    print(f"  jit, first call         {first_ms:8.1f} ms  (trace + compile)")
    print(f"  jit, steady state       {jit_ms:8.2f} ms  ({eager_ms / jit_ms:.1f}x faster than eager)")
    print(f"  jit, new shape          {reshape_ms:8.1f} ms  (compiles again)")
    print(f"  jit, timed without sync {nosync_ms:8.3f} ms  (wrong: dispatch only)")
    print(f"  compiled program: {kernels} kernels for 11 ops\n")
    return jit_ms


def run_torch():
    try:
        import torch
    except ImportError:
        print("PyTorch not installed, skipping\n")
        return None
    if not torch.cuda.is_available():  # ROCm builds also answer to "cuda"
        print("PyTorch has no GPU, skipping\n")
        return None
    print(f"PyTorch {torch.__version__} on {torch.cuda.get_device_name(0)}")
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(N, N, device="cuda", generator=g)
    sync = lambda a: torch.cuda.synchronize()
    eager_ms = timed(lambda: gelu_layernorm(torch, x), sync)

    # Count the kernels eager mode launches for one call.
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        gelu_layernorm(torch, x)
        torch.cuda.synchronize()
    launches = sum(e.count for e in prof.key_averages()
                   if e.device_type.name == "CUDA")
    print(f"  eager                   {eager_ms:8.2f} ms  ({launches} kernel launches)\n")
    return eager_ms


if __name__ == "__main__":
    run_jax()
    run_torch()
