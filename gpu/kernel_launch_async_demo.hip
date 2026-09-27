// kernel_launch_async_demo.hip
// Companion example for: topnotchnote.com/gpu/kernel_launch.html
//
// Three small experiments that make a kernel launch visible:
//   1. What one launch costs: back-to-back launches of an empty kernel.
//   2. Timing it wrong vs right: a host clock around the launch measures
//      only the enqueue; events recorded in the stream measure the kernel.
//   3. Blocks run in no particular order: each block takes a ticket from
//      a global counter when it starts, and we print who started when.
// Compile:  hipcc -O2 -o main kernel_launch_async_demo.hip
// Nvidia:   hipcc also targets Nvidia (HIP_PLATFORM=nvidia), or swap the hip
//           prefix for cuda in the API calls and build with nvcc.

#include <hip/hip_runtime.h>

#include <chrono>
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

using Clock = std::chrono::steady_clock;

static double microsSince(Clock::time_point start) {
  return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
}

__global__ void emptyKernel() {}

// Enough work per element that the kernel runs for milliseconds.
__global__ void busyKernel(float *data, int n, int iters) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float x = data[i];
    for (int k = 0; k < iters; ++k) {
      x = x * 0.999f + 0.001f;
    }
    data[i] = x;
  }
}

// Thread 0 of each block takes the next ticket as the block starts.
__global__ void recordStartOrder(unsigned int *counter, unsigned int *ticketOfBlock) {
  if (threadIdx.x == 0) {
    ticketOfBlock[blockIdx.x] = atomicAdd(counter, 1u);
  }
}

int main() {
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  std::printf("Device: %s, %d SMs/CUs\n\n", prop.name, prop.multiProcessorCount);

  // Warm up: the first launch also loads the code object and is not representative.
  emptyKernel<<<1, 64>>>();
  HIP_CHECK(hipDeviceSynchronize());

  // 1. Launch cost.
  const int launches = 10000;
  auto t0 = Clock::now();
  for (int i = 0; i < launches; ++i) {
    emptyKernel<<<1, 64>>>();
  }
  double enqueueUs = microsSince(t0);
  HIP_CHECK(hipDeviceSynchronize());
  double totalUs = microsSince(t0);
  std::printf("1. %d empty launches\n", launches);
  std::printf("   CPU time to enqueue: %.2f us per launch\n", enqueueUs / launches);
  std::printf("   until the GPU drained them all: %.2f us per launch\n\n", totalUs / launches);

  // 2. Timing wrong vs right.
  const int n = 1 << 22;
  const int iters = 2000;
  const int block = 256;
  const int grid = (n + block - 1) / block;
  float *d_data = nullptr;
  HIP_CHECK(hipMalloc(&d_data, n * sizeof(float)));
  HIP_CHECK(hipMemset(d_data, 0, n * sizeof(float)));

  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  auto t1 = Clock::now();
  HIP_CHECK(hipEventRecord(start));
  busyKernel<<<grid, block>>>(d_data, n, iters);
  HIP_CHECK(hipEventRecord(stop));
  HIP_CHECK(hipGetLastError());  // reports launch errors only, not faults inside the kernel
  double wrongUs = microsSince(t1);  // no synchronization yet: this is the enqueue
  HIP_CHECK(hipEventSynchronize(stop));
  float rightMs = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&rightMs, start, stop));
  std::printf("2. %d blocks of %d threads\n", grid, block);
  std::printf("   host clock around the launch: %.1f us  (the enqueue)\n", wrongUs);
  std::printf("   events in the stream:         %.1f us  (the kernel)\n\n", rightMs * 1000.0f);

  // 3. Block start order.
  const int orderBlocks = 4096;
  unsigned int *d_counter = nullptr, *d_tickets = nullptr;
  HIP_CHECK(hipMalloc(&d_counter, sizeof(unsigned int)));
  HIP_CHECK(hipMalloc(&d_tickets, orderBlocks * sizeof(unsigned int)));
  HIP_CHECK(hipMemset(d_counter, 0, sizeof(unsigned int)));
  recordStartOrder<<<orderBlocks, 64>>>(d_counter, d_tickets);
  HIP_CHECK(hipGetLastError());
  std::vector<unsigned int> tickets(orderBlocks);
  HIP_CHECK(hipMemcpy(tickets.data(), d_tickets, orderBlocks * sizeof(unsigned int),
                      hipMemcpyDeviceToHost));

  int outOfOrder = 0;
  for (int b = 0; b < orderBlocks; ++b) {
    if (tickets[b] != static_cast<unsigned int>(b)) ++outOfOrder;
  }
  std::printf("3. %d blocks, start ticket of blocks 0..15:\n   ", orderBlocks);
  for (int b = 0; b < 16; ++b) std::printf("%u ", tickets[b]);
  std::printf("\n   %d of %d blocks did not start in index order\n", outOfOrder, orderBlocks);

  HIP_CHECK(hipFree(d_tickets));
  HIP_CHECK(hipFree(d_counter));
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  HIP_CHECK(hipFree(d_data));
  return 0;
}
