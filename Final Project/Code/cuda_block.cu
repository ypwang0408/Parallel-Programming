#include "config.h"
#include "util.h"
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cuda.h>
#include <string>

#ifndef BLOCK_DIM
#define BLOCK_DIM 16
#endif

void floydWarshall(int *matrix, const int n);
__global__ void floyd_warshall_block_kernel_phase1(int n, int k, int *graph);
__global__ void floyd_warshall_block_kernel_phase2(int n, int k, int *graph);
__global__ void floyd_warshall_block_kernel_phase3(int n, int k, int *graph);

int main(int argc, char *argv[]) {
    int n, density, threadsPerBlock;

    threadsPerBlock = BLOCK_DIM * BLOCK_DIM;
    if (argc <= 2) {
        n = DEFAULT;
        density = 100;
    } else {
        n = atoi(argv[1]);
        density = atoi(argv[2]);
    }

    int *matrix = (int *)malloc(n * n * sizeof(int));
    populateMatrix(matrix, n, density);

    if (PRINTABLE) {
        printf("*** Adjacency matrix:\n");
        showDistances(matrix, n);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    floydWarshall(matrix, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float accum = 0;
    cudaEventElapsedTime(&accum, start, stop);

    if (PRINTABLE) {
        printf("*** The solution is:\n");
        showDistances(matrix, n);
    }

    printf("[GPGPU] Total elapsed time %.2f ms\n", accum);
    free(matrix);

    // calculate theoretical occupancy
    int maxActiveBlocksPerSM, device;
    float occupancy;
    cudaDeviceProp props;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&props, device);

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocksPerSM,
                                                  floyd_warshall_block_kernel_phase1, threadsPerBlock,
                                                  0);
    occupancy = (maxActiveBlocksPerSM * threadsPerBlock / props.warpSize) /
                (float)(props.maxThreadsPerMultiProcessor /
                        props.warpSize);
    printf("Phase: 1\n");
    printf("maxActiveBlocksPerSM: %d, warpSize: %d, maxThreadsPerMultiProcessor: %d\n", maxActiveBlocksPerSM, props.warpSize, props.maxThreadsPerMultiProcessor);
    printf("Launched blocks of size: %d, Theoretical occupancy: %f\n", threadsPerBlock, occupancy);

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocksPerSM,
                                                  floyd_warshall_block_kernel_phase1, threadsPerBlock,
                                                  0);
    occupancy = (maxActiveBlocksPerSM * threadsPerBlock / props.warpSize) /
                (float)(props.maxThreadsPerMultiProcessor /
                        props.warpSize);
    printf("Phase: 2\n");
    printf("maxActiveBlocksPerSM: %d, warpSize: %d, maxThreadsPerMultiProcessor: %d\n", maxActiveBlocksPerSM, props.warpSize, props.maxThreadsPerMultiProcessor);
    printf("Launched blocks of size: %d, Theoretical occupancy: %f\n", threadsPerBlock, occupancy);

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocksPerSM,
                                                  floyd_warshall_block_kernel_phase1, threadsPerBlock,
                                                  0);
    occupancy = (maxActiveBlocksPerSM * threadsPerBlock / props.warpSize) /
                (float)(props.maxThreadsPerMultiProcessor /
                        props.warpSize);
    printf("Phase: 3\n");
    printf("maxActiveBlocksPerSM: %d, warpSize: %d, maxThreadsPerMultiProcessor: %d\n", maxActiveBlocksPerSM, props.warpSize, props.maxThreadsPerMultiProcessor);
    printf("Launched blocks of size: %d, Theoretical occupancy: %f\n", threadsPerBlock, occupancy);

    FILE *pFile;
    pFile = fopen("./result/cuda_block_result.txt", "a");
    if (NULL == pFile) {
        puts("open failure");
        exit(1);
    } else {
        fprintf(pFile, "%d\t%d\t%d\t%.2f\n", threadsPerBlock, n, density, accum);
    }
    fclose(pFile);

    return 0;
}

void floydWarshall(int *matrix, const int n) {
    int *deviceMatrix;
    int size = n * n * sizeof(int);
    cudaMalloc(&deviceMatrix, size);
    cudaMemcpy(deviceMatrix, matrix, size, cudaMemcpyHostToDevice);

    const int blocks = (n + BLOCK_DIM - 1) / BLOCK_DIM;
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM, 1);
    dim3 dimGrid(blocks, blocks, 1);

    for (int k = 0; k < blocks; k++) {
        floyd_warshall_block_kernel_phase1<<<1, dimBlock>>>(n, k, deviceMatrix);
        floyd_warshall_block_kernel_phase2<<<blocks, dimBlock>>>(n, k, deviceMatrix);
        floyd_warshall_block_kernel_phase3<<<dimGrid, dimBlock>>>(n, k, deviceMatrix);
    }

    cudaMemcpy(matrix, deviceMatrix, size, cudaMemcpyDeviceToHost);
    cudaFree(deviceMatrix);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
}

__forceinline__
    __device__ void
    block_calc(int *C, int *A, int *B, int bi, int bj) {
    for (int k = 0; k < BLOCK_DIM; k++) {
        int sum = A[bi * BLOCK_DIM + k] + B[k * BLOCK_DIM + bj];
        if (C[bi * BLOCK_DIM + bj] > sum) {
            C[bi * BLOCK_DIM + bj] = sum;
        }
        __syncthreads();
    }
}

__global__ void floyd_warshall_block_kernel_phase1(int n, int k, int *graph) {
    const int bi = threadIdx.y;
    const int bj = threadIdx.x;
    const int row_idx = k * BLOCK_DIM + bi;
    const int col_idx = k * BLOCK_DIM + bj;

    __shared__ int C[BLOCK_DIM * BLOCK_DIM];

    // Transfer to temp shared arrays
    C[bi * BLOCK_DIM + bj] = graph[row_idx * n + col_idx];

    __syncthreads();

    block_calc(C, C, C, bi, bj);
    // Transfer back to graph
    graph[row_idx * n + col_idx] = C[bi * BLOCK_DIM + bj];
}

__global__ void floyd_warshall_block_kernel_phase2(int n, int k, int *graph) {
    // BlockDim is one dimensional (Straight along diagonal)
    // Blocks themselves are two dimensional
    const int i = blockIdx.x;
    const int bi = threadIdx.y;
    const int bj = threadIdx.x;

    if (i == k)
        return;

    __shared__ int A[BLOCK_DIM * BLOCK_DIM];
    __shared__ int B[BLOCK_DIM * BLOCK_DIM];
    __shared__ int C[BLOCK_DIM * BLOCK_DIM];

    C[bi * BLOCK_DIM + bj] = graph[i * BLOCK_DIM * n + k * BLOCK_DIM + bi * n + bj];
    B[bi * BLOCK_DIM + bj] = graph[k * BLOCK_DIM * n + k * BLOCK_DIM + bi * n + bj];

    __syncthreads();

    block_calc(C, C, B, bi, bj);
    graph[i * BLOCK_DIM * n + k * BLOCK_DIM + bi * n + bj] = C[bi * BLOCK_DIM + bj];

    __syncthreads();

    C[bi * BLOCK_DIM + bj] = graph[k * BLOCK_DIM * n + i * BLOCK_DIM + bi * n + bj];
    A[bi * BLOCK_DIM + bj] = graph[k * BLOCK_DIM * n + k * BLOCK_DIM + bi * n + bj];

    __syncthreads();

    block_calc(C, A, C, bi, bj);
    // Block C is the only one that could be changed
    graph[k * BLOCK_DIM * n + i * BLOCK_DIM + bi * n + bj] = C[bi * BLOCK_DIM + bj];
}

__global__ void floyd_warshall_block_kernel_phase3(int n, int k, int *graph) {
    // BlockDim is one dimensional (Straight along diagonal)
    // Blocks themselves are two dimensional
    const int j = blockIdx.x;
    const int i = blockIdx.y;
    const int bi = threadIdx.y;
    const int bj = threadIdx.x;

    if (i == k && j == k)
        return;

    __shared__ int A[BLOCK_DIM * BLOCK_DIM];
    __shared__ int B[BLOCK_DIM * BLOCK_DIM];
    __shared__ int C[BLOCK_DIM * BLOCK_DIM];

    C[bi * BLOCK_DIM + bj] = graph[i * BLOCK_DIM * n + j * BLOCK_DIM + bi * n + bj];
    A[bi * BLOCK_DIM + bj] = graph[i * BLOCK_DIM * n + k * BLOCK_DIM + bi * n + bj];
    B[bi * BLOCK_DIM + bj] = graph[k * BLOCK_DIM * n + j * BLOCK_DIM + bi * n + bj];

    __syncthreads();

    block_calc(C, A, B, bi, bj);
    graph[i * BLOCK_DIM * n + j * BLOCK_DIM + bi * n + bj] = C[bi * BLOCK_DIM + bj];
}
