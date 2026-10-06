#include <iostream>
#include <cuda_runtime.h>
#include "common.cuh"


int main() {
    int deviceCount = 0; // return devices (nvidia GPU) via cudaGetDeviceCount

    cudaError_t cudaErr = cudaGetDeviceCount(&deviceCount);

    if (cudaErr != cudaSuccess) {
        std::cerr << "error with finding Cuda Device" << std::endl;
        return 1;
    }

    std::cout << "# CUDA devices found: " << deviceCount << std::endl;


    // print all gpu properties
    
    for (int i = 0; i < deviceCount; i++) {

        
        //registers per SM, L2 size, memory bus width, memory clock. Then compute theoretical bandwidth from the bus width and memory clock, and print it.

        cudaDeviceProp prop;

        cudaGetDeviceProperties(&prop, i);

        // print properties

        std::cout << "GPU name: " << prop.name << std::endl;
        std::cout << "GPU major & minor compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "max threads per GPU SM: " << prop.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "shared mem per gpu block" << prop.sharedMemPerBlock << std::endl;
        std::cout << "num gpu SM's " << prop.multiProcessorCount << std::endl;
        std::cout << "shared mem for each SM " << prop.sharedMemPerMultiprocessor << std::endl;
        std::cout << "registers per SM " << prop.regsPerMultiprocessor << std::endl;
        std::cout << "total l2 cache size " << prop.l2CacheSize << std::endl;
        std::cout << "mem bus width" << prop.memoryBusWidth << std::endl;
        std::cout << "clock rate for memory (in kHz) " << prop.memoryClockRate << std::endl;

        double bandwidth = (((double) prop.memoryClockRate*2000) * (prop.memoryBusWidth / 8) ) / 1e9;

        std::cout << "calculated bandwidth " <<  bandwidth << std::endl;



    }
}