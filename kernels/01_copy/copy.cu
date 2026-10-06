#include "common.cuh" // general include statement, pulls everything i need

// constants

constexpr size_t N = 256ull * 1024 * 1024; // 256mb floats
constexpr size_t BYTES = N * sizeof(float); // 1 gbz
constexpr double = PEAK_GBPS = 320.064; // bandwidth, calculated in devquery

// goal of this file is to check how fast we can move data over float4, scalar, and gridstride float4. we'll use these constant declarations throughout each kernel 

std::vector<float> h_in(N);

for (size_t i = 0; i < N; i++) {
    h_in.[i] = (float) i;
}

// output buf
std::vector<float> h_out(N);

// malloc onto array
float* d_in = nullptr;
float* d_out = nullptr;

CUDA_CHECK(cudaMalloc((void**)&d_in, BYTES));
CUDA_CHECK(cudaMalloc((void**)&d_out, BYTES));
// wrap for protection

CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), BYTES, cudaMemcpyHostToDevice));


