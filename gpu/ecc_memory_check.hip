// ecc_memory_check.hip
// Companion example for: topnotchnote.com/gpu/ecc_memory.html
//
// Two things:
//   1. Asks the runtime whether ECC is on for this GPU. Datacenter cards
//      report 1; nearly every consumer card reports 0, because it has no
//      ECC to turn on.
//   2. Runs a small memory test of the kind ECC makes unnecessary: fill most
//      of free device memory with address-dependent patterns, read every
//      word back, count mismatches. On a healthy card the count is zero.
//      A non-zero count on a card without ECC is exactly the failure ECC
//      exists to catch, except that here nothing caught it until we looked.
//   3. Flips one bit on purpose and confirms the checker sees it.
// This is a sanity check, not a replacement for a real memory tester: one
// pass will not find a rare, temperature-dependent flip.
//
// Compile:  hipcc -O2 -o main ecc_memory_check.hip
// Nvidia:   hipcc also targets Nvidia (HIP_PLATFORM=nvidia), or swap the hip
//           prefix for cuda in the API calls and build with nvcc.

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define HIP_CHECK(call)                                                   \
  do {                                                                    \
    hipError_t err_ = (call);                                             \
    if (err_ != hipSuccess) {                                             \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,             \
                   hipGetErrorString(err_));                              \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

// Each word's expected value depends on its index and the pass, so a bit
// stuck at 0 or 1, or an address line that maps two words onto one cell,
// shows up as a mismatch.
__device__ __forceinline__ uint32_t pattern(size_t i, uint32_t pass) {
  uint32_t x = (uint32_t)i * 2654435761u ^ (uint32_t)(i >> 32);
  switch (pass) {
    case 0: return 0x00000000u;
    case 1: return 0xFFFFFFFFu;
    case 2: return (i & 1) ? 0xAAAAAAAAu : 0x55555555u;  // checkerboard
    default: return x ^ (pass * 0x9E3779B9u);            // hashed address
  }
}

__global__ void fill(uint32_t *buf, size_t n, uint32_t pass) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x)
    buf[i] = pattern(i, pass);
}

__global__ void verify(const uint32_t *buf, size_t n, uint32_t pass,
                       unsigned long long *badWords,
                       unsigned long long *badBits) {
  unsigned long long words = 0, bits = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x) {
    uint32_t diff = buf[i] ^ pattern(i, pass);
    if (diff) {
      ++words;
      bits += __popc(diff);
    }
  }
  if (words) {
    atomicAdd(badWords, words);
    atomicAdd(badBits, bits);
  }
}

int main() {
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  int ecc = 0;
  HIP_CHECK(hipDeviceGetAttribute(&ecc, hipDeviceAttributeEccEnabled, 0));

  size_t freeB = 0, totalB = 0;
  HIP_CHECK(hipMemGetInfo(&freeB, &totalB));
  std::printf("Device      : %s\n", prop.name);
  std::printf("ECC enabled : %s\n", ecc ? "yes" : "no");
  std::printf("Memory      : %.2f GB total, %.2f GB free\n\n", totalB / 1e9,
              freeB / 1e9);

  // Leave headroom for the display and the runtime itself.
  size_t bytes = (size_t)(freeB * 0.8) & ~(size_t)0xFFFFF;
  size_t n = bytes / sizeof(uint32_t);
  uint32_t *buf = nullptr;
  unsigned long long *dCounts = nullptr;
  HIP_CHECK(hipMalloc(&buf, bytes));
  HIP_CHECK(hipMalloc(&dCounts, 2 * sizeof(unsigned long long)));

  const char *names[] = {"all zeros", "all ones", "checkerboard",
                         "hashed address A", "hashed address B"};
  const int passes = 5;
  int grid = prop.multiProcessorCount * 8, block = 256;
  unsigned long long totalBad = 0;

  std::printf("Testing %.2f GB, %d patterns\n", bytes / 1e9, passes);
  for (int p = 0; p < passes; ++p) {
    unsigned long long counts[2] = {0, 0};
    HIP_CHECK(hipMemset(dCounts, 0, 2 * sizeof(unsigned long long)));
    fill<<<grid, block>>>(buf, n, (uint32_t)p);
    verify<<<grid, block>>>(buf, n, (uint32_t)p, dCounts, dCounts + 1);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(counts, dCounts, sizeof(counts),
                        hipMemcpyDeviceToHost));
    std::printf("  %-17s : %llu bad words, %llu flipped bits\n", names[p],
                counts[0], counts[1]);
    totalBad += counts[0];
  }

  // Self-test: flip one bit on purpose, so a zero above means something.
  {
    fill<<<grid, block>>>(buf, n, 3u);
    uint32_t word = 0;
    size_t victim = n / 3;
    HIP_CHECK(hipMemcpy(&word, buf + victim, sizeof(word),
                        hipMemcpyDeviceToHost));
    word ^= 1u << 7;
    HIP_CHECK(hipMemcpy(buf + victim, &word, sizeof(word),
                        hipMemcpyHostToDevice));
    unsigned long long counts[2] = {0, 0};
    HIP_CHECK(hipMemset(dCounts, 0, 2 * sizeof(unsigned long long)));
    verify<<<grid, block>>>(buf, n, 3u, dCounts, dCounts + 1);
    HIP_CHECK(hipMemcpy(counts, dCounts, sizeof(counts),
                        hipMemcpyDeviceToHost));
    std::printf("  self-test (1 bit flipped on purpose): %llu bad word, "
                "%llu flipped bit %s\n", counts[0], counts[1],
                counts[0] == 1 && counts[1] == 1 ? "-> checker works"
                                                 : "-> CHECKER BROKEN");
  }

  if (totalBad == 0)
    std::printf("\nNo mismatches. %s\n",
                ecc ? "With ECC on, single-bit flips would have been "
                      "corrected before we read them anyway; check the "
                      "driver's ECC counters for those."
                    : "Without ECC, this is the only kind of check you get.");
  else
    std::printf("\n%llu mismatched words. Without ECC nothing reported "
                "this; with ECC, single-bit flips would have been corrected "
                "and counted.\n", totalBad);

  HIP_CHECK(hipFree(buf));
  HIP_CHECK(hipFree(dCounts));
  return 0;
}
