ARCH ?= sm_75
NVCC := nvcc
FLAGS := -O3 -std=c++17 -gencode arch=compute_$(subst sm_,,$(ARCH)),code=$(ARCH) -Icommon

# targets

bin:
	mkdir -p bin

devquery: bin
	$(NVCC) $(FLAGS) hardware/devquery.cu -o bin/devquery

copy: bin
	$(NVCC) $(FLAGS) kernels/01_copy/copy.cu -o bin/copy

peak_flops: bin
	$(NVCC) $(FLAGS) hardware/peak_flops.cu -o bin/peak_flops

.PHONY: devquery copy peak_flops

# so far, these targets are just for performance checking via stress test on CUDA