// nvcc cuda2.cu -o cuda2.out -gencode=arch=compute_75,code=compute_75 -O3
#include "config.h"
#include "util.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cuda.h>
#include <math.h>
#include <stdlib.h>
#include <string>

#define BLOCK_SIZE 4

__global__ void wakeGPU(int reps);
__global__ void floydWarshallKernel(int k, int *matrix, int n);

void floydWarshall(int *matrix, int n, int threadsPerBlock);

int main(int argc, char *argv[]) {
    int n, density, threadsPerBlock;

    if (argc <= 3) {
        n = DEFAULT;
        density = 100;
        threadsPerBlock = BLOCK_SIZE;
    } else {
        n = atoi(argv[1]);
        density = atoi(argv[2]);
        threadsPerBlock = atoi(argv[3]);
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
    floydWarshall(matrix, n, threadsPerBlock);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float accum;
    cudaEventElapsedTime(&accum, start, stop);

    if (PRINTABLE) {
        printf("*** The solution is:\n");
        showDistances(matrix, n);
    }

    printf("[GPGPU] Total elapsed time %.2f ms\n", accum);

    // calculate theoretical occupancy
    int maxActiveBlocksPerSM;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocksPerSM,
                                                  floydWarshallKernel,
                                                  threadsPerBlock,
                                                  0);

    int device;
    cudaDeviceProp props;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&props, device);

    float occupancy = (maxActiveBlocksPerSM * threadsPerBlock / props.warpSize) /
                      (float)(props.maxThreadsPerMultiProcessor /
                              props.warpSize);

    printf("Launched blocks of size %d. Theoretical occupancy: %f\n",
           threadsPerBlock, occupancy);

    free(matrix);

    FILE *pFile;
    pFile = fopen("./result/cuda2_result.txt", "a");
    if (NULL == pFile) {
        puts("open failure");
        exit(1);
    } else {
        fprintf(pFile, "%d\t%d\t%d\t%.2f\n", threadsPerBlock, n, density, accum);
    }
    fclose(pFile);

    return 0;
}

void floydWarshall(int *matrix, const int n, int threadsPerBlock) {
    int *deviceMatrix;
    int size = n * n * sizeof(int);

    cudaMalloc((int **)&deviceMatrix, size);
    cudaMemcpy(deviceMatrix, matrix, size, cudaMemcpyHostToDevice);

    dim3 dimGrid((n + threadsPerBlock - 1) / threadsPerBlock, n);

    cudaFuncSetCacheConfig(floydWarshallKernel, cudaFuncCachePreferL1);
    for (int k = 0; k < n; k++) {
        floydWarshallKernel<<<dimGrid, threadsPerBlock>>>(k, deviceMatrix, n);
    }
    cudaDeviceSynchronize();

    cudaMemcpy(matrix, deviceMatrix, size, cudaMemcpyDeviceToHost);

    cudaFree(deviceMatrix);

    cudaError err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
}

__global__ void floydWarshallKernel(int k, int *matrix, int n) {
    int i = blockDim.y * blockIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    if (j < n) {
        int newPath = matrix[k * n + j] + matrix[i * n + k];
        int oldPath = matrix[i * n + j];
        if (oldPath > newPath) {
            matrix[i * n + j] = newPath;
        }
    }
}