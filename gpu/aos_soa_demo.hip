// aos_soa_demo.hip
// Companion example for: topnotchnote.com/gpu/aos_vs_soa.html
//
// The same particle data stored two ways:
//   AoS  (array of structs):  struct Particle { x, y, z, vx, vy, vz, mass, charge }  particles[n]
//   SoA  (struct of arrays):  x[n], y[n], z[n], vx[n], ..., charge[n]
// and three kernels that touch different amounts of each particle:
//   1. scale one field (charge)            : 1 of 8 floats read and written
//   2. move: x += vx*dt, for x, y, z       : 6 of 8 floats read, 3 written
//   3. reduce all 8 fields to one number   : 8 of 8 floats read
// For AoS there is also a version that loads each 32-byte particle as two
// float4 (16-byte) loads. The program prints the time and the "useful"
// bandwidth: bytes the kernel needs, divided by time.
//
// Compile:  hipcc -O2 -o main aos_soa_demo.hip
// Nvidia:   hipcc also targets Nvidia (HIP_PLATFORM=nvidia), or swap the hip
//           prefix for cuda in the API calls and build with nvcc.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIP_CHECK(call)                                                   \
  do {                                                                    \
    hipError_t err_ = (call);                                             \
    if (err_ != hipSuccess) {                                             \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,             \
                   hipGetErrorString(err_));                              \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

struct __align__(16) Particle { float x, y, z, vx, vy, vz, mass, charge; };  // 32 bytes

struct SoA { float *x, *y, *z, *vx, *vy, *vz, *mass, *charge; };

// ---- 1. one field ----
__global__ void scale_aos(Particle* p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i].charge *= 0.5f;
}
__global__ void scale_soa(SoA s, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) s.charge[i] *= 0.5f;
}

// ---- 2. move ----
__global__ void move_aos(Particle* p, int n, float dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  p[i].x += p[i].vx * dt;
  p[i].y += p[i].vy * dt;
  p[i].z += p[i].vz * dt;
}
__global__ void move_aos_float4(Particle* p, int n, float dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float4* q = reinterpret_cast<float4*>(p + i);
  float4 a = q[0];            // x, y, z, vx
  float4 b = q[1];            // vy, vz, mass, charge
  a.x += a.w * dt;
  a.y += b.x * dt;
  a.z += b.y * dt;
  q[0] = a;                   // write back the first half only
}
__global__ void move_soa(SoA s, int n, float dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  s.x[i] += s.vx[i] * dt;
  s.y[i] += s.vy[i] * dt;
  s.z[i] += s.vz[i] * dt;
}

// ---- 3. all fields ----
__global__ void all_aos(const Particle* p, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  Particle q = p[i];
  out[i] = q.x + q.y + q.z + q.vx + q.vy + q.vz + q.mass + q.charge;
}
__global__ void all_soa(SoA s, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = s.x[i] + s.y[i] + s.z[i] + s.vx[i] + s.vy[i] + s.vz[i] + s.mass[i] + s.charge[i];
}

template <typename F>
static float median_ms(F launch) {
  hipEvent_t a, b;
  HIP_CHECK(hipEventCreate(&a));
  HIP_CHECK(hipEventCreate(&b));
  launch();
  HIP_CHECK(hipGetLastError());
  std::vector<float> t;
  for (int r = 0; r < 11; ++r) {
    HIP_CHECK(hipEventRecord(a));
    launch();
    HIP_CHECK(hipEventRecord(b));
    HIP_CHECK(hipEventSynchronize(b));
    float ms;
    HIP_CHECK(hipEventElapsedTime(&ms, a, b));
    t.push_back(ms);
  }
  HIP_CHECK(hipEventDestroy(a));
  HIP_CHECK(hipEventDestroy(b));
  std::sort(t.begin(), t.end());
  return t[5];
}

int main() {
  const int n = 8 << 20;                       // 8M particles, 256 MB either way
  const unsigned threads = 256, blocks = (n + threads - 1) / threads;
  Particle* aos;
  HIP_CHECK(hipMalloc(&aos, n * sizeof(Particle)));
  HIP_CHECK(hipMemset(aos, 0, n * sizeof(Particle)));
  SoA soa;
  float** f[] = {&soa.x, &soa.y, &soa.z, &soa.vx, &soa.vy, &soa.vz, &soa.mass, &soa.charge};
  for (float** p : f) {
    HIP_CHECK(hipMalloc(p, n * sizeof(float)));
    HIP_CHECK(hipMemset(*p, 0, n * sizeof(float)));
  }
  float* out;
  HIP_CHECK(hipMalloc(&out, n * sizeof(float)));

  auto row = [&](const char* name, double useful_bytes, float ms) {
    std::printf("  %-26s %8.3f ms   %6.1f GB/s useful\n", name, ms, useful_bytes / ms / 1e6);
  };
  const double F = 4.0 * n;                    // bytes in one field of all particles

  std::printf("1. scale one field (needs 1 field read + written)\n");
  row("AoS", 2 * F, median_ms([&] { scale_aos<<<blocks, threads>>>(aos, n); }));
  row("SoA", 2 * F, median_ms([&] { scale_soa<<<blocks, threads>>>(soa, n); }));

  std::printf("2. move (needs 6 fields read, 3 written)\n");
  row("AoS", 9 * F, median_ms([&] { move_aos<<<blocks, threads>>>(aos, n, 0.01f); }));
  row("AoS, two float4 loads", 9 * F, median_ms([&] { move_aos_float4<<<blocks, threads>>>(aos, n, 0.01f); }));
  row("SoA", 9 * F, median_ms([&] { move_soa<<<blocks, threads>>>(soa, n, 0.01f); }));

  std::printf("3. read all 8 fields (needs 8 fields read, 1 written)\n");
  row("AoS", 9 * F, median_ms([&] { all_aos<<<blocks, threads>>>(aos, out, n); }));
  row("SoA", 9 * F, median_ms([&] { all_soa<<<blocks, threads>>>(soa, out, n); }));

  HIP_CHECK(hipFree(aos));
  for (float** p : f) HIP_CHECK(hipFree(*p));
  HIP_CHECK(hipFree(out));
  return 0;
}
