// failure_tests.cu  --  break things on purpose.
// build: nvcc -O3 -arch=sm_89 -lineinfo failure_tests.cu -o failure_tests
// run:   ./failure_tests 1     (bad launch config)
//        ./failure_tests 2     (out-of-bounds write)
//        compute-sanitizer ./failure_tests 2
#include <cstdio>
#include <cstdlib>
#include "../check.h"

__global__ void write_kernel(float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] = 1.0f;
    }
}

// TODO (yours): copy of write_kernel with a deliberate out-of-bounds write,
// e.g. a[i + 1000000] = 1.0f. Keep the bounds check on i itself, so the
// bug is only the offset.
__global__ void oob_kernel(float* a, int n) {
}

int main(int argc, char** argv) {
    int mode = (argc > 1) ? atoi(argv[1]) : 0;
    const int n = 1 << 20;
    float* d_a;
    CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float)));

    if (mode == 1) {
        // TODO (yours): launch write_kernel with 1025 threads per block.
        // Then call CUDA_CHECK_LAUNCH() and read the error it prints.
        // Try it with only cudaDeviceSynchronize() and no cudaGetLastError()
        // too. What changes?
    } else if (mode == 2) {
        // TODO (yours): launch oob_kernel with a valid config.
        // Run once normally: does it crash? Then run under compute-sanitizer
        // and read the report (which thread, which address, which line).
    } else {
        printf("usage: %s <1|2>\n", argv[0]);
    }

    CUDA_CHECK(cudaFree(d_a));
    return 0;
}
