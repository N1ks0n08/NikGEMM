// device_info.cu  --  build: nvcc -O3 -arch=sm_89 -lineinfo device_info.cu -o device_info
#include <cstdio>
#include "../check.h"

int main() {
    int dev = 0;
    CUDA_CHECK(cudaSetDevice(dev));

    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));

    printf("Name:                  %s\n", p.name);
    printf("Compute capability:    %d.%d\n", p.major, p.minor);
    printf("SM count:              %d\n", p.multiProcessorCount);
    printf("Warp size:             %d\n", p.warpSize);
    printf("Max threads/block:     %d\n", p.maxThreadsPerBlock);
    printf("Shared mem/block:      %zu bytes\n", p.sharedMemPerBlock);
    printf("Registers/block:       %d\n", p.regsPerBlock);
    printf("Memory bus width:      %d bits\n", p.memoryBusWidth);
    printf("Global memory:         %.2f GiB\n", p.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    // On CUDA 13.x the clock fields were removed from cudaDeviceProp,
    // so query them as attributes. Units: kHz.
    int mem_clock_khz = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, dev));
    printf("Memory clock attr:     %d kHz\n", mem_clock_khz);

    // TODO (yours): peak memory bandwidth in GB/s.
    //   Inputs: mem_clock_khz, p.memoryBusWidth.
    //   Figure out: bits -> bytes, and whether the reported clock already
    //   includes the double-data-rate factor. Compare to the published spec.
    double peak_bw_gbs = static_cast<double>(mem_clock_khz / 1000000) * (static_cast<double>(p.memoryBusWidth) / 8.0) * 2.0;  // TODO
    printf("Peak bandwidth (yours): %.1f GB/s\n", peak_bw_gbs);
    printf("L2 cache:              %d bytes\n", p.l2CacheSize);
    return 0;
}
