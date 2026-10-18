#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Wrap every CUDA runtime call: CUDA_CHECK(cudaMalloc(&d_a, bytes));
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s (%d)\n  %s\n  at %s:%d\n  in: %s\n", \
                    cudaGetErrorName(err__), (int)err__,                      \
                    cudaGetErrorString(err__), __FILE__, __LINE__, #call);    \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Launches return nothing, so use this right after one:
//   my_kernel<<<grid, block>>>(...);
//   CUDA_CHECK_LAUNCH();
// cudaGetLastError() catches bad launch configs;
// cudaDeviceSynchronize() surfaces errors that happen during execution.
// (The sync makes the CPU wait, so remove it from timed regions.)
#define CUDA_CHECK_LAUNCH()                                                   \
    do {                                                                      \
        CUDA_CHECK(cudaGetLastError());                                       \
        CUDA_CHECK(cudaDeviceSynchronize());                                  \
    } while (0)
