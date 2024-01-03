// g++ -fopenmp openmp.c -o openmp.out -O3 && ./openmp.out 10 100
#include "config.h"
#include "util.h"
#include <cstdlib>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void floydWarshall(int *matrix, int n, int threads);

int main(int argc, char **argv) {
    int n, density, threads;
    if (argc <= 3) {
        n = DEFAULT;
        density = 100;
        threads = omp_get_max_threads();
    } else {
        n = atoi(argv[1]);
        density = atoi(argv[2]);
        threads = atoi(argv[3]);
    }

    int *matrix;
    matrix = (int *)malloc(n * n * sizeof(int));

    populateMatrix(matrix, n, density);

    if (PRINTABLE) {
        printf("*** Adjacency matrix:\n");
        showDistances(matrix, n);
    }

    struct timespec start, end;
    double accum;

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    floydWarshall(matrix, n, threads);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    accum = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1000000.0;

    if (PRINTABLE) {
        printf("*** The solution is:\n");
        showDistances(matrix, n);
    }

    printf("[OPENMP] Total elapsed time %.2lf ms\n", accum);
    free(matrix);

    FILE *pFile;
    pFile = fopen("./result/omp_result.txt", "a");
    if (NULL == pFile) {
        puts("open failure");
        exit(1);
    } else {
        fprintf(pFile, "%d\t%d\t%d\t%.2lf\n", threads, n, density, accum);
    }
    fclose(pFile);
    return 0;
}

void floydWarshall(int *matrix, int n, int threads) {

    int *rowK = (int *)malloc(sizeof(int) * n);

    for (int k = 0; k < n; k++) {
#pragma omp parallel num_threads(threads)
        {
#pragma omp single
            memcpy(rowK, matrix + (k * n), sizeof(int) * n);

#pragma omp for
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    int newPath = matrix[i * n + k] + rowK[j];
                    if (matrix[i * n + j] > newPath) {
                        matrix[i * n + j] = newPath;
                    }
                }
            }
        }
    }

    free(rowK);
}
