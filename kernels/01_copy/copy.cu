#include "common.cuh" // general include statement, pulls everything i need
#include <iostream>

// constants

constexpr size_t N = 256ull * 1024 * 1024; // 256mb floats
constexpr size_t BYTES = N * sizeof(float); // 1 gbz
constexpr double PEAK_GBPS = 320.064; // bandwidth, calculated in devquery

// goal of this file is to check how fast we can move data over float4, scalar, and gridstride float4. we'll use these constant declarations throughout each kernel 


// verify helper

void verify(const char* name, const float* d_out, std::vector<float> &h_out, const std::vector<float> &h_in) {
    // Inside: copy d_out back to h_out (cudaMemcpyDeviceToHost, wrapped in CUDA_CHECK), call check_equal(h_out.data(), h_in.data(), N), print PASS or FAIL.

    // verification of whether or not something is correct

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, BYTES, cudaMemcpyDeviceToHost)); // dest, source

    size_t bad = check_equal(h_out.data(), h_in.data(), N); // h_in -> answer, h_out -> copied from d_out on GPU to h_out on CPU

    if (bad == 0) {
        std::cout << name << " PASS" << std::endl;
    } else {
        std::cout << name << " FAIL: " << bad << std::endl;
    }
}

void report(const char* name, float ms) {
    // reports time taken for a kernel

    // feed in ms as a float, get back a report of bandwidth and peak bandwidth % comparison

    double bw = gbps(2.0 * BYTES, ms);

    double peakPercent = bw / PEAK_GBPS * 100;

    printf("%-12s  %8.3f ms   %8.3f GB/s  %8.3f%%\n", name, ms, bw, peakPercent);
}


// copy_scalar kernel

__global__ void copy_scalar(const float* __restrict__ in, float* __restrict__ out, size_t n) {

    size_t i = blockIdx.x * blockDim.x + threadIdx.x; // standard practice

    if (i < n) {
        out[i] = in[i];
    }

}

// copy vec with float4 strides

__global__ void copy_vec4(const float4* __restrict__ in, float4* __restrict__ out, size_t n4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n4) {
        out[i] = in[i];
    }
}

int main() {
    std::vector<float> h_in(N);

    for (size_t i = 0; i < N; i++) {
        h_in[i] = (float) i;
    }

    // output buf
    std::vector<float> h_out(N);

    // malloc onto array
    float* d_in = nullptr;
    float* d_out = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&d_in, BYTES));
    CUDA_CHECK(cudaMalloc((void**)&d_out, BYTES));
    // wrap for protection

    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), BYTES, cudaMemcpyHostToDevice)); // source->dest 

    CUDA_CHECK(cudaMemset(d_out, 0, N));

    float ms1 = bench_ms([&] { cudaMemcpy(d_out, d_in, BYTES, cudaMemcpyDeviceToDevice); });

    report("cudaMemcpy test ", ms1);

    verify("cudaMemcpy test ", d_out, h_out, h_in);

    cudaMemset(d_out, 0, BYTES);

    int n4 = N/4;

    int blocks4 = (n4 + 255) / 256;

    float ms3 = bench_ms([&] { copy_vec4<<<blocks4, 256>>>(reinterpret_cast<const float4*>(d_in), reinterpret_cast<float4*>(d_out), N); });

    report("copy_vec4 test", ms3);

    verify("copy_vec4 test", d_out, h_out, h_in);

    cudaMemset(d_out, 0, BYTES);

    int blocks = (N + 255) / 256; // threads = 256

    float ms2 = bench_ms([&] { copy_scalar<<<blocks, 256>>>(d_in, d_out, N); });

    report("copy_scalar test", ms2);

    verify("copy_scalar test", d_out, h_out, h_in);


    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    return 0;
}
