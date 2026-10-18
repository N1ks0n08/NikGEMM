// vecadd.cu  --  build: nvcc -O3 -arch=sm_89 -lineinfo vecadd.cu -o vecadd
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "../check.h"

// Times a region on the GPU's own timeline (a marker pushed into the stream).
struct GpuTimer {
    cudaEvent_t a, b;
    GpuTimer()  { CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b)); }
    ~GpuTimer() { cudaEventDestroy(a); cudaEventDestroy(b); }
    void start() { CUDA_CHECK(cudaEventRecord(a)); }
    float stop() {  // returns milliseconds
        CUDA_CHECK(cudaEventRecord(b));
        CUDA_CHECK(cudaEventSynchronize(b));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
        return ms;
    }
};

// TODO (yours): global index, bounds check, c[i] = a[i] + b[i].
__global__ void vecadd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int n = 33554435;   // ~2^25, not a multiple of 256
    const size_t bytes = (size_t)n * sizeof(float);

    // Warm-up: the first CUDA call initializes the context and is slow.
    // Time this line once on its own if you want to see how slow.
    CUDA_CHECK(cudaFree(0));

    // Host buffers (h_ = host pointer)
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    for (int i = 0; i < n; i++) {
        h_a[i] = (float)(i % 100) * 0.5f;
        h_b[i] = (float)(i % 37) * 2.0f;
    }

    // Device buffers (d_ = device pointer; never dereference these on the CPU)
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    GpuTimer t;

    // Host -> device
    t.start();
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    float ms_h2d = t.stop();

    // TODO (yours): block size and grid size (round up).
    int block = 256;
    int grid  = (n + block - 1) / block;  // TODO

    // Kernel: warm up first (module load, clock ramp), then time many launches
    for (int w = 0; w < 10; w++) vecadd<<<grid, block>>>(d_a, d_b, d_c, n);
    CUDA_CHECK_LAUNCH();

    const int iters = 100;
    t.start();
    for (int r = 0; r < iters; r++) vecadd<<<grid, block>>>(d_a, d_b, d_c, n);
    float ms_kernel = t.stop() / iters;   // average per launch
    CUDA_CHECK_LAUNCH();

    // A: launch only, no sync
    auto t0 = std::chrono::steady_clock::now();
    vecadd<<<grid, block>>>(d_a, d_b, d_c, n);
    auto t1 = std::chrono::steady_clock::now();
    CUDA_CHECK_LAUNCH();   // after t1, so it isn't timed

    // B: launch + wait for the GPU
    auto t2 = std::chrono::steady_clock::now();
    vecadd<<<grid, block>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t3 = std::chrono::steady_clock::now();

    printf("chrono, no sync:   %.1f us\n", std::chrono::duration<double, std::micro>(t1 - t0).count());
    printf("chrono, with sync: %.1f us\n", std::chrono::duration<double, std::micro>(t3 - t2).count());

    // Device -> host
    t.start();
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    float ms_d2h = t.stop();

    // Verify against a CPU loop
    int bad = 0;
    for (int i = 0; i < n; i++) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            if (bad < 5) printf("mismatch at %d: got %f expected %f\n", i, h_c[i], h_a[i] + h_b[i]);
            bad++;
        }
    }
    printf("verify: %s (%d mismatches)\n", bad ? "FAIL" : "PASS", bad);

    printf("H2D (2 arrays): %.3f ms\n", ms_h2d);
    printf("kernel:         %.3f ms\n", ms_kernel);
    double gbs = (3.0 * bytes) / (ms_kernel * 1e-3) / 1e9;
    printf("achieved:       %.1f GB/s (%.0f%% of 256)\n", gbs, 100.0 * gbs / 256.0);
    printf("D2H (1 array):  %.3f ms\n", ms_d2h);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a); free(h_b); free(h_c);
    return 0;
}
