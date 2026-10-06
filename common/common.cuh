// common.cuh - Day 1 benchmark harness
// Usage: #include "common.cuh" in each .cu file.
// Compile: nvcc -O3 -gencode arch=compute_75,code=sm_75 copy.cu -o copy
#pragma once
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err_ = (call);                                                \
    if (err_ != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),   \
              __FILE__, __LINE__);                                            \
      exit(1);                                                                \
    }                                                                         \
  } while (0)

// Check for errors from the most recent kernel launch.
#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaGetLastError())

// Times `launch` (a lambda that launches one kernel) and returns the
// median time in milliseconds over `iters` runs, after `warmup` runs.
// Each run is timed separately with its own pair of events.
template <typename F>
float bench_ms(F launch, int warmup = 10, int iters = 100) {
  for (int i = 0; i < warmup; ++i) launch();
  CUDA_CHECK_LAUNCH();
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  std::vector<float> times(iters);
  for (int i = 0; i < iters; ++i) {
    CUDA_CHECK(cudaEventRecord(start));
    launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop));
  }
  CUDA_CHECK_LAUNCH();

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  std::sort(times.begin(), times.end());
  return times[iters / 2];
}

// Bandwidth in GB/s given total bytes moved (read + written) and time in ms.
inline double gbps(double bytes, float ms) { return bytes / (ms * 1e-3) / 1e9; }

// Compare two host float arrays exactly (a copy should be bit-identical).
// Returns the number of mismatches and prints the first few.
inline size_t check_equal(const float* a, const float* b, size_t n) {
  size_t bad = 0;
  for (size_t i = 0; i < n; ++i) {
    if (a[i] != b[i]) {
      if (bad < 5) fprintf(stderr, "  mismatch at %zu: %f vs %f\n", i, a[i], b[i]);
      ++bad;
    }
  }
  return bad;
}
