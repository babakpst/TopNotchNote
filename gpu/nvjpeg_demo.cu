// nvjpeg_demo.cu
// Companion example for: topnotchnote.com/gpu/media_libraries.html
//
// Decode a folder of JPEG files into GPU memory with nvJPEG and time it
// per backend:
//   HYBRID      Huffman decoding on the CPU, the rest on the GPU
//   GPU_HYBRID  Huffman decoding on the GPU as well
//   HARDWARE    the dedicated JPEG engine (A100/A30, Hopper, Ada, Blackwell);
//               creation fails on GPUs without one, and the program says so
// first one image at a time (nvjpegDecode), then the whole folder as one
// batch (nvjpegDecodeBatched). Output is interleaved RGB in device memory.
// The AMD counterpart, rocJPEG, needs a VCN JPEG engine and has its own API.
//
// Usage:    ./nvjpeg_demo file1.jpg file2.jpg ...
// Compile:  nvcc -O2 -o nvjpeg_demo nvjpeg_demo.cu -lnvjpeg

#include <cuda_runtime.h>
#include <nvjpeg.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                  \
  do {                                                                    \
    cudaError_t e_ = (call);                                              \
    if (e_ != cudaSuccess) {                                              \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__,             \
                   cudaGetErrorString(e_));                               \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

#define JPEG_CHECK(call)                                                  \
  do {                                                                    \
    nvjpegStatus_t s_ = (call);                                           \
    if (s_ != NVJPEG_STATUS_SUCCESS) {                                    \
      std::fprintf(stderr, "%s:%d: nvJPEG status %d\n", __FILE__,         \
                   __LINE__, (int)s_);                                    \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

static double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: %s file.jpg ...\n", argv[0]); return 1; }
  std::vector<std::vector<unsigned char>> files;
  size_t bytes = 0;
  for (int i = 1; i < argc; ++i) {
    std::ifstream f(argv[i], std::ios::binary);
    files.emplace_back(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    bytes += files.back().size();
  }
  const int n = (int)files.size();

  // Image size from the first file; the demo assumes all files match it.
  nvjpegHandle_t probe;
  JPEG_CHECK(nvjpegCreateSimple(&probe));
  int comps, w[NVJPEG_MAX_COMPONENT], h[NVJPEG_MAX_COMPONENT];
  nvjpegChromaSubsampling_t sub;
  JPEG_CHECK(nvjpegGetImageInfo(probe, files[0].data(), files[0].size(), &comps, &sub, w, h));
  JPEG_CHECK(nvjpegDestroy(probe));
  std::printf("%d files, %.1f MB, %d x %d, %d components\n\n", n, bytes / 1e6, w[0], h[0], comps);

  // One RGBI output buffer per image, in device memory.
  std::vector<nvjpegImage_t> out(n);
  for (auto& img : out) {
    img = nvjpegImage_t{};
    CUDA_CHECK(cudaMalloc(&img.channel[0], (size_t)w[0] * h[0] * 3));
    img.pitch[0] = w[0] * 3;
  }
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  struct { const char* name; nvjpegBackend_t b; } backends[] = {
      {"HYBRID     (Huffman on CPU)", NVJPEG_BACKEND_HYBRID},
      {"GPU_HYBRID (Huffman on GPU)", NVJPEG_BACKEND_GPU_HYBRID},
      {"HARDWARE   (JPEG engine)   ", NVJPEG_BACKEND_HARDWARE},
  };
  for (auto& be : backends) {
    nvjpegHandle_t handle;
    nvjpegStatus_t s = nvjpegCreateEx(be.b, nullptr, nullptr, 0, &handle);
    if (s != NVJPEG_STATUS_SUCCESS) {
      std::printf("%s  not available on this GPU (status %d)\n", be.name, (int)s);
      continue;
    }
    nvjpegJpegState_t state;
    JPEG_CHECK(nvjpegJpegStateCreate(handle, &state));

    // One image at a time. The hardware backend only supports batched decode.
    double single = -1;
    if (be.b != NVJPEG_BACKEND_HARDWARE) {
      for (int rep = 0; rep < 2; ++rep) {  // first pass warms up
        double t0 = now_s();
        for (int i = 0; i < n; ++i)
          JPEG_CHECK(nvjpegDecode(handle, state, files[i].data(), files[i].size(),
                                  NVJPEG_OUTPUT_RGBI, &out[i], stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        single = now_s() - t0;
      }
    }

    // The whole folder as one batch.
    std::vector<const unsigned char*> ptrs(n);
    std::vector<size_t> lens(n);
    for (int i = 0; i < n; ++i) { ptrs[i] = files[i].data(); lens[i] = files[i].size(); }
    double batched = -1;
    s = nvjpegDecodeBatchedInitialize(handle, state, n, 1, NVJPEG_OUTPUT_RGBI);
    if (s == NVJPEG_STATUS_SUCCESS) {
      for (int rep = 0; rep < 2; ++rep) {
        double t0 = now_s();
        JPEG_CHECK(nvjpegDecodeBatched(handle, state, ptrs.data(), lens.data(), out.data(), stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        batched = now_s() - t0;
      }
    }
    std::printf("%s", be.name);
    if (single > 0) std::printf("  one by one %6.2f ms/image (%6.1f images/s)", single / n * 1e3, n / single);
    if (batched > 0) std::printf("  batched %6.2f ms/image (%6.1f images/s)", batched / n * 1e3, n / batched);
    std::printf("\n");
    JPEG_CHECK(nvjpegJpegStateDestroy(state));
    JPEG_CHECK(nvjpegDestroy(handle));
  }
  for (auto& img : out) CUDA_CHECK(cudaFree(img.channel[0]));
  return 0;
}
