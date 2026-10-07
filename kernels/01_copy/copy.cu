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

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    return 0;
}
