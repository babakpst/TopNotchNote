# matrix_isa_dump.py
# Companion example for: topnotchnote.com/gpu/reading_gpu_assembly.html
#
# Compiles ONE 64x64x64 FP16 tile multiply, written once in Triton, for
# three GPU targets, and counts the matrix instructions in the machine code:
#   * AMD gfx942 (MI300X)  -> v_mfma_* in the AMDGPU ISA
#   * Nvidia sm_80 (A100)  -> HMMA in SASS
#   * Nvidia sm_90 (H100)  -> HGMMA in SASS
# No GPU is needed: Triton compiles ahead of time for any target it knows.
#
# Requirements: triton (tested with 3.1.0), and cuobjdump from the CUDA
# toolkit on PATH for the Nvidia dumps.
# The AMD path normally links the code object with ROCm's ld.lld. Without
# ROCm installed, this script stops after the assembly stage: it points
# TRITON_HIP_LLD_PATH at a dummy file and skips the final link. That reaches
# into Triton internals, so it may need adjusting on other Triton versions.
#
# Run:  python matrix_isa_dump.py

import collections
import os
import re
import shutil
import subprocess
import tempfile

os.environ.setdefault("TRITON_HIP_LLD_PATH", shutil.which("true") or "/bin/true")

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

try:  # skip the final link when ROCm's linker is not installed
    import triton.backends.amd.compiler as amd_compiler

    if not os.path.exists("/opt/rocm/llvm/bin/ld.lld"):
        amd_compiler.HIPBackend.make_hsaco = staticmethod(lambda src, md, opt: b"")
except ImportError:
    amd_compiler = None


@triton.jit
def tile_mm(a_ptr, b_ptr, c_ptr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    m = tl.arange(0, BM)
    n = tl.arange(0, BN)
    k = tl.arange(0, BK)
    a = tl.load(a_ptr + m[:, None] * BK + k[None, :])
    b = tl.load(b_ptr + k[:, None] * BN + n[None, :])
    tl.store(c_ptr + m[:, None] * BN + n[None, :], tl.dot(a, b))


SRC = ASTSource(
    fn=tile_mm,
    signature={"a_ptr": "*fp16", "b_ptr": "*fp16", "c_ptr": "*fp32"},
    constants={"BM": 64, "BN": 64, "BK": 64},
)


def count(lines, pattern):
    c = collections.Counter()
    for line in lines:
        for m in re.finditer(pattern, line):
            c[m.group(0)] += 1
    return c


def show(label, counts, example):
    print(f"\n{label}")
    for name, n in counts.items():
        print(f"  {n:3d} x {name}")
    if example:
        print(f"  e.g. {example.strip()}")


# AMD: the assembly text is one of Triton's compile stages.
if amd_compiler is not None:
    k = triton.compile(SRC, target=GPUTarget("hip", "gfx942", 64))
    asm = k.asm["amdgcn"].splitlines()
    mfma = [l for l in asm if "v_mfma" in l]
    show("gfx942 (MI300X), per wavefront:", count(asm, r"v_mfma_\w+"),
         mfma[0] if mfma else "")

# Nvidia: Triton produces a cubin; cuobjdump turns it into SASS.
for cc, pattern, label in [(80, r"\bHMMA\.\w+(\.\w+)*", "sm_80 (A100), per warp:"),
                           (90, r"\bHGMMA\.\w+(\.\w+)*", "sm_90 (H100), per warpgroup:")]:
    k = triton.compile(SRC, target=GPUTarget("cuda", cc, 32))
    with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
        f.write(k.asm["cubin"])
        f.flush()
        sass = subprocess.run(["cuobjdump", "-sass", f.name], capture_output=True,
                              text=True, check=True).stdout.splitlines()
    hits = [l for l in sass if re.search(pattern, l)]
    show(label, count(sass, pattern), re.sub(r"/\*.*?\*/|;", "", hits[0]) if hits else "")

print("\nThe tile is 64 x 64 x 64 = 262,144 multiply-adds in every case.")
