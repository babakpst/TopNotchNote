// storage_staging_demo.hip
// Companion example for: topnotchnote.com/gpu/gpu_storage_io.html
//
// Without GPUDirect Storage (cuFile) or AMD Infinity Storage (hipFile), a
// file reaches GPU memory through a host "bounce" buffer: the CPU reads the
// file into host memory, then copies it to the GPU. That is also exactly
// what cuFile's compatibility mode and hipFile's fallback backend do.
// This program times the pieces of that path for one file:
//   1. disk -> pinned host buffer, O_DIRECT (bypasses the page cache)
//   2. pinned host buffer -> GPU (hipMemcpy)
//   3. both, one chunk after the other: the plain bounce path from disk
//   4. page cache -> pageable buffer -> GPU (file already cached in RAM)
//   5. page cache -> pinned buffer -> GPU
// Steps 4 and 5 stand in for very fast storage: when the source delivers
// gigabytes per second, the extra host copy starts to matter.
//
// Usage:    ./main <file> [size_MiB]   (creates the file if it does not exist)
// Compile:  hipcc -O2 -o main storage_staging_demo.hip
// Nvidia:   HIP_PLATFORM=nvidia hipcc ..., or rename hip -> cuda and build
//           with nvcc -O2 -x cu -o main storage_staging_demo.hip
// Linux only (O_DIRECT, pread).

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
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

static const size_t CHUNK = size_t(16) << 20;  // 16 MiB per read and per copy

static double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void read_all(int fd, char* buf, size_t size, size_t chunk) {
  for (size_t off = 0; off < size; off += chunk) {
    ssize_t r = pread(fd, buf, chunk, (off_t)off);  // reuse one buffer
    if (r != (ssize_t)chunk) { std::perror("pread"); std::exit(1); }
  }
}

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: %s <file> [size_MiB]\n", argv[0]); return 1; }
  const char* path = argv[1];
  size_t size = (argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 1024) << 20;
  size = size / CHUNK * CHUNK;

  struct stat st;
  if (stat(path, &st) != 0 || (size_t)st.st_size < size) {
    std::printf("creating %s (%zu MiB)...\n", path, size >> 20);
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    std::vector<char> junk(CHUNK);
    for (size_t i = 0; i < CHUNK; ++i) junk[i] = (char)(i * 2654435761u >> 13);
    for (size_t off = 0; off < size; off += CHUNK)
      if (write(fd, junk.data(), CHUNK) != (ssize_t)CHUNK) { std::perror("write"); return 1; }
    fsync(fd);
    close(fd);
  }

  char *pinned, *dev;
  HIP_CHECK(hipHostMalloc((void**)&pinned, CHUNK));  // page aligned, so O_DIRECT accepts it
  HIP_CHECK(hipMalloc((void**)&dev, size));
  std::vector<char> pageable(CHUNK);
  auto gbps = [&](double s) { return size / s / 1e9; };
  std::printf("file %s, %zu MiB, %zu MiB chunks\n\n", path, size >> 20, CHUNK >> 20);

  // 1. disk only
  int dfd = open(path, O_RDONLY | O_DIRECT);
  if (dfd < 0) { std::perror("open O_DIRECT"); return 1; }
  double t0 = now_s();
  read_all(dfd, pinned, size, CHUNK);
  double t_disk = now_s() - t0;

  // 2. host -> GPU only
  t0 = now_s();
  for (size_t off = 0; off < size; off += CHUNK)
    HIP_CHECK(hipMemcpy(dev + off, pinned, CHUNK, hipMemcpyHostToDevice));
  double t_copy = now_s() - t0;

  // 3. the bounce path from disk: read a chunk, copy it, repeat
  t0 = now_s();
  for (size_t off = 0; off < size; off += CHUNK) {
    if (pread(dfd, pinned, CHUNK, (off_t)off) != (ssize_t)CHUNK) { std::perror("pread"); return 1; }
    HIP_CHECK(hipMemcpy(dev + off, pinned, CHUNK, hipMemcpyHostToDevice));
  }
  double t_bounce = now_s() - t0;
  close(dfd);

  // 4 and 5. from the page cache: read the file once (buffered) to cache it
  int bfd = open(path, O_RDONLY);
  read_all(bfd, pageable.data(), size, CHUNK);
  t0 = now_s();
  read_all(bfd, pageable.data(), size, CHUNK);
  double t_cache = now_s() - t0;
  t0 = now_s();
  for (size_t off = 0; off < size; off += CHUNK) {
    if (pread(bfd, pageable.data(), CHUNK, (off_t)off) != (ssize_t)CHUNK) { std::perror("pread"); return 1; }
    HIP_CHECK(hipMemcpy(dev + off, pageable.data(), CHUNK, hipMemcpyHostToDevice));
  }
  double t_cache_pageable = now_s() - t0;
  t0 = now_s();
  for (size_t off = 0; off < size; off += CHUNK) {
    if (pread(bfd, pinned, CHUNK, (off_t)off) != (ssize_t)CHUNK) { std::perror("pread"); return 1; }
    HIP_CHECK(hipMemcpy(dev + off, pinned, CHUNK, hipMemcpyHostToDevice));
  }
  double t_cache_pinned = now_s() - t0;
  close(bfd);

  std::printf("1. disk -> pinned host (O_DIRECT)       %8.0f ms  %6.2f GB/s\n", t_disk * 1e3, gbps(t_disk));
  std::printf("2. pinned host -> GPU                    %8.0f ms  %6.2f GB/s\n", t_copy * 1e3, gbps(t_copy));
  std::printf("3. disk -> pinned -> GPU, chunk by chunk %8.0f ms  %6.2f GB/s\n", t_bounce * 1e3, gbps(t_bounce));
  std::printf("   share of step 3 spent copying to the GPU: about %.0f%%\n\n", 100.0 * t_copy / t_bounce);
  std::printf("4. page cache -> pageable host           %8.0f ms  %6.2f GB/s\n", t_cache * 1e3, gbps(t_cache));
  std::printf("   page cache -> pageable -> GPU         %8.0f ms  %6.2f GB/s\n", t_cache_pageable * 1e3, gbps(t_cache_pageable));
  std::printf("5. page cache -> pinned -> GPU           %8.0f ms  %6.2f GB/s\n", t_cache_pinned * 1e3, gbps(t_cache_pinned));

  HIP_CHECK(hipHostFree(pinned));
  HIP_CHECK(hipFree(dev));
  return 0;
}
