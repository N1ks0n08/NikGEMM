#include <cstdio>
__global__ void k() { printf("hello from GPU thread %d\n", threadIdx.x); }
int main() { k<<<1,4>>>(); cudaDeviceSynchronize(); return 0; }
