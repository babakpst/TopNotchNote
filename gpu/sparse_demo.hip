// sparse_demo.hip
// Companion example for: topnotchnote.com/gpu/sparse_libraries.html
//
// When is a sparse matrix worth storing as sparse? One 8192 x 8192 FP32
// matrix-vector product, y = A x, at densities from 0.1% to 50%:
//   - dense:  hipblasSgemv on the full matrix (256 MB, zeros included)
//   - sparse: hipsparseSpMV on the same matrix in CSR form
// Then the same matrix times a dense 8192 x 128 block, Y = A B:
//   - dense:  hipblasSgemm
//   - sparse: hipsparseSpMM
// For each density it prints the storage each format needs, the time of
// each call, and the largest difference between the dense and sparse
// matrix-vector results. The nonzeros are scattered at random, which is
// the hard case for sparse.
//
// Compile:  hipcc -O2 -o main sparse_demo.hip -lhipsparse -lhipblas
// Nvidia:   hipcc also targets Nvidia (HIP_PLATFORM=nvidia), or swap the
//           hipsparse/hipblas prefixes for cusparse/cublas and build with
//           nvcc -O2 ... -lcusparse -lcublas

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hipsparse/hipsparse.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
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

#define HIPBLAS_CHECK(call)                                               \
  do {                                                                    \
    hipblasStatus_t st_ = (call);                                         \
    if (st_ != HIPBLAS_STATUS_SUCCESS) {                                  \
      std::fprintf(stderr, "%s:%d: hipBLAS error %d\n", __FILE__,         \
                   __LINE__, (int)st_);                                   \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

#define HIPSPARSE_CHECK(call)                                             \
  do {                                                                    \
    hipsparseStatus_t st_ = (call);                                       \
    if (st_ != HIPSPARSE_STATUS_SUCCESS) {                                \
      std::fprintf(stderr, "%s:%d: hipSPARSE error %d\n", __FILE__,       \
                   __LINE__, (int)st_);                                   \
      std::exit(1);                                                       \
    }                                                                     \
  } while (0)

int main() {
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  std::printf("Device: %s\n\n", prop.name);

  const int n = 8192, m = 128;  // m: columns of the dense block B
  const size_t denseBytes = (size_t)n * n * sizeof(float);
  const int reps = 20;
  const float alpha = 1.0f, beta = 0.0f;

  hipblasHandle_t blas;
  HIPBLAS_CHECK(hipblasCreate(&blas));
  hipsparseHandle_t sparse;
  HIPSPARSE_CHECK(hipsparseCreate(&sparse));
  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  std::vector<float> hx(n);
  for (int i = 0; i < n; ++i) hx[i] = (float)((i * 7) % 13) / 13.0f - 0.5f;
  float *x, *yDense, *ySparse, *dA, *B, *C;
  HIP_CHECK(hipMalloc(&x, n * sizeof(float)));
  HIP_CHECK(hipMalloc(&yDense, n * sizeof(float)));
  HIP_CHECK(hipMalloc(&ySparse, n * sizeof(float)));
  HIP_CHECK(hipMalloc(&dA, denseBytes));
  HIP_CHECK(hipMalloc(&B, (size_t)n * m * sizeof(float)));
  HIP_CHECK(hipMalloc(&C, (size_t)n * m * sizeof(float)));
  {
    std::vector<float> hB((size_t)n * m);
    for (size_t i = 0; i < hB.size(); ++i) hB[i] = (float)((i * 5) % 7) / 7.0f - 0.5f;
    HIP_CHECK(hipMemcpy(B, hB.data(), hB.size() * sizeof(float), hipMemcpyHostToDevice));
  }
  HIP_CHECK(hipMemcpy(x, hx.data(), n * sizeof(float), hipMemcpyHostToDevice));

  std::printf("%31s | %-29s | %s\n", "", "y = A x (matrix-vector)",
              "Y = A B, B is 8192 x 128");
  std::printf("%8s %10s %10s | %9s %9s %9s | %9s %9s %7s\n", "density",
              "dense MB", "CSR MB", "gemv ms", "SpMV ms", "|diff|", "gemm ms",
              "SpMM ms", "winner");

  const double densities[] = {0.001, 0.003, 0.01, 0.05, 0.10, 0.20, 0.30, 0.50};
  std::mt19937 rng(42);
  for (double density : densities) {
    // Build the matrix on the host in CSR form: for each row, pick its
    // nonzero columns at random, sorted. Also expand it to dense, row-major.
    const int perRow = std::max(1, (int)std::lround(density * n));
    std::vector<int> rowPtr(n + 1), cols;
    std::vector<float> vals;
    cols.reserve((size_t)perRow * n);
    vals.reserve((size_t)perRow * n);
    std::vector<float> dense((size_t)n * n, 0.0f);
    std::vector<int> pool(n);
    for (int i = 0; i < n; ++i) pool[i] = i;
    for (int r = 0; r < n; ++r) {
      rowPtr[r] = (int)cols.size();
      // Partial shuffle: the first perRow entries become a random subset.
      for (int k = 0; k < perRow; ++k)
        std::swap(pool[k], pool[k + rng() % (n - k)]);
      std::sort(pool.begin(), pool.begin() + perRow);
      for (int k = 0; k < perRow; ++k) {
        float v = (float)((r + pool[k]) % 11) / 11.0f - 0.5f;
        cols.push_back(pool[k]);
        vals.push_back(v);
        dense[(size_t)r * n + pool[k]] = v;
      }
    }
    rowPtr[n] = (int)cols.size();
    const int nnz = rowPtr[n];
    const size_t csrBytes = (size_t)nnz * (sizeof(int) + sizeof(float)) +
                            (n + 1) * sizeof(int);

    int *dRowPtr, *dCols;
    float *dVals;
    HIP_CHECK(hipMalloc(&dRowPtr, (n + 1) * sizeof(int)));
    HIP_CHECK(hipMalloc(&dCols, nnz * sizeof(int)));
    HIP_CHECK(hipMalloc(&dVals, nnz * sizeof(float)));
    HIP_CHECK(hipMemcpy(dRowPtr, rowPtr.data(), (n + 1) * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dCols, cols.data(), nnz * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dVals, vals.data(), nnz * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dA, dense.data(), denseBytes, hipMemcpyHostToDevice));

    // Dense: BLAS is column-major, so the row-major matrix is A^T to it;
    // ask for the transpose to get y = A x.
    HIPBLAS_CHECK(hipblasSgemv(blas, HIPBLAS_OP_T, n, n, &alpha, dA, n, x, 1,
                               &beta, yDense, 1));  // warm-up
    HIP_CHECK(hipEventRecord(start, 0));
    for (int i = 0; i < reps; ++i)
      HIPBLAS_CHECK(hipblasSgemv(blas, HIPBLAS_OP_T, n, n, &alpha, dA, n, x, 1,
                                 &beta, yDense, 1));
    HIP_CHECK(hipEventRecord(stop, 0));
    HIP_CHECK(hipEventSynchronize(stop));
    float denseMs;
    HIP_CHECK(hipEventElapsedTime(&denseMs, start, stop));
    denseMs /= reps;

    // Sparse: describe the matrix and vectors, size the scratch buffer,
    // then call SpMV.
    hipsparseSpMatDescr_t matA;
    hipsparseDnVecDescr_t vecX, vecY;
    HIPSPARSE_CHECK(hipsparseCreateCsr(&matA, n, n, nnz, dRowPtr, dCols, dVals,
                                       HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
                                       HIPSPARSE_INDEX_BASE_ZERO, HIP_R_32F));
    HIPSPARSE_CHECK(hipsparseCreateDnVec(&vecX, n, x, HIP_R_32F));
    HIPSPARSE_CHECK(hipsparseCreateDnVec(&vecY, n, ySparse, HIP_R_32F));
    size_t bufBytes = 0;
    HIPSPARSE_CHECK(hipsparseSpMV_bufferSize(
        sparse, HIPSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta,
        vecY, HIP_R_32F, HIPSPARSE_SPMV_ALG_DEFAULT, &bufBytes));
    void *buf = nullptr;
    if (bufBytes) HIP_CHECK(hipMalloc(&buf, bufBytes));
    HIPSPARSE_CHECK(hipsparseSpMV(sparse, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                  &alpha, matA, vecX, &beta, vecY, HIP_R_32F,
                                  HIPSPARSE_SPMV_ALG_DEFAULT, buf));  // warm-up
    HIP_CHECK(hipEventRecord(start, 0));
    for (int i = 0; i < reps; ++i)
      HIPSPARSE_CHECK(hipsparseSpMV(sparse, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matA, vecX, &beta, vecY, HIP_R_32F,
                                    HIPSPARSE_SPMV_ALG_DEFAULT, buf));
    HIP_CHECK(hipEventRecord(stop, 0));
    HIP_CHECK(hipEventSynchronize(stop));
    float sparseMs;
    HIP_CHECK(hipEventElapsedTime(&sparseMs, start, stop));
    sparseMs /= reps;

    std::vector<float> a(n), b(n);
    HIP_CHECK(hipMemcpy(a.data(), yDense, n * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(b.data(), ySparse, n * sizeof(float), hipMemcpyDeviceToHost));
    double maxDiff = 0.0;
    for (int i = 0; i < n; ++i)
      maxDiff = std::fmax(maxDiff, std::fabs((double)a[i] - b[i]));

    // Matrix times a block of vectors. Dense: the same transpose trick, with
    // B and C column-major. Sparse: SpMM with B and C as dense matrices.
    HIPBLAS_CHECK(hipblasSgemm(blas, HIPBLAS_OP_T, HIPBLAS_OP_N, n, m, n,
                               &alpha, dA, n, B, n, &beta, C, n));  // warm-up
    HIP_CHECK(hipEventRecord(start, 0));
    for (int i = 0; i < 5; ++i)
      HIPBLAS_CHECK(hipblasSgemm(blas, HIPBLAS_OP_T, HIPBLAS_OP_N, n, m, n,
                                 &alpha, dA, n, B, n, &beta, C, n));
    HIP_CHECK(hipEventRecord(stop, 0));
    HIP_CHECK(hipEventSynchronize(stop));
    float gemmMs;
    HIP_CHECK(hipEventElapsedTime(&gemmMs, start, stop));
    gemmMs /= 5;

    hipsparseDnMatDescr_t matB, matC;
    HIPSPARSE_CHECK(hipsparseCreateDnMat(&matB, n, m, n, B, HIP_R_32F,
                                         HIPSPARSE_ORDER_COL));
    HIPSPARSE_CHECK(hipsparseCreateDnMat(&matC, n, m, n, C, HIP_R_32F,
                                         HIPSPARSE_ORDER_COL));
    size_t mmBytes = 0;
    HIPSPARSE_CHECK(hipsparseSpMM_bufferSize(
        sparse, HIPSPARSE_OPERATION_NON_TRANSPOSE,
        HIPSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
        HIP_R_32F, HIPSPARSE_SPMM_ALG_DEFAULT, &mmBytes));
    void *mmBuf = nullptr;
    if (mmBytes) HIP_CHECK(hipMalloc(&mmBuf, mmBytes));
    HIPSPARSE_CHECK(hipsparseSpMM(sparse, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                  HIPSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                                  matA, matB, &beta, matC, HIP_R_32F,
                                  HIPSPARSE_SPMM_ALG_DEFAULT, mmBuf));  // warm-up
    HIP_CHECK(hipEventRecord(start, 0));
    for (int i = 0; i < 5; ++i)
      HIPSPARSE_CHECK(hipsparseSpMM(sparse, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                    HIPSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                                    matA, matB, &beta, matC, HIP_R_32F,
                                    HIPSPARSE_SPMM_ALG_DEFAULT, mmBuf));
    HIP_CHECK(hipEventRecord(stop, 0));
    HIP_CHECK(hipEventSynchronize(stop));
    float spmmMs;
    HIP_CHECK(hipEventElapsedTime(&spmmMs, start, stop));
    spmmMs /= 5;

    std::printf("%7.1f%% %10.1f %10.1f | %9.3f %9.3f %9.1e | %9.2f %9.2f %7s\n",
                density * 100, denseBytes / 1048576.0, csrBytes / 1048576.0,
                denseMs, sparseMs, maxDiff, gemmMs, spmmMs,
                spmmMs < gemmMs ? "SpMM" : "gemm");

    if (mmBuf) HIP_CHECK(hipFree(mmBuf));
    HIPSPARSE_CHECK(hipsparseDestroyDnMat(matB));
    HIPSPARSE_CHECK(hipsparseDestroyDnMat(matC));

    if (buf) HIP_CHECK(hipFree(buf));
    HIPSPARSE_CHECK(hipsparseDestroySpMat(matA));
    HIPSPARSE_CHECK(hipsparseDestroyDnVec(vecX));
    HIPSPARSE_CHECK(hipsparseDestroyDnVec(vecY));
    HIP_CHECK(hipFree(dRowPtr));
    HIP_CHECK(hipFree(dCols));
    HIP_CHECK(hipFree(dVals));
  }

  HIP_CHECK(hipFree(x));
  HIP_CHECK(hipFree(yDense));
  HIP_CHECK(hipFree(ySparse));
  HIP_CHECK(hipFree(dA));
  HIP_CHECK(hipFree(B));
  HIP_CHECK(hipFree(C));
  HIPSPARSE_CHECK(hipsparseDestroy(sparse));
  HIPBLAS_CHECK(hipblasDestroy(blas));
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  return 0;
}
