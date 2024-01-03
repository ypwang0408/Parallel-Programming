// g++ -fopenmp openmp.c -o openmp.out -O3 && ./openmp.out 10 100
#include "config.h"
#include "util.h"
#include <cstdlib>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void floyd_warshall_blocked(int *matrix, const int n, const int b);

int main(int argc, char **argv) {
    int n, density, threads, block_sz;
    if (argc <= 4) {
        n = DEFAULT;
        density = 100;
        threads = omp_get_max_threads();
        block_sz = 16;
    } else {
        n = atoi(argv[1]);
        density = atoi(argv[2]);
        threads = atoi(argv[3]);
        block_sz = atoi(argv[4]);
    }

    omp_set_num_threads(threads);

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
    floyd_warshall_blocked(matrix, n, block_sz);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    accum = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1000000.0;

    if (PRINTABLE) {
        printf("*** The solution is:\n");
        showDistances(matrix, n);
    }

    printf("[OPENMP] Total elapsed time %.2lf ms\n", accum);
    free(matrix);

    FILE *pFile;
    pFile = fopen("./result/omp_block_result.txt", "a");
    if (NULL == pFile) {
        puts("open failure");
        exit(1);
    } else {
        fprintf(pFile, "%d\t%d\t%d\t%.2lf\t%d\n", threads, n, density, accum, block_sz);
    }
    fclose(pFile);
    return 0;
}

inline void floyd_warshall_in_place(int *C, int *A, int *B, const int b, const int n) {
    for (int k = 0; k < b; k++) {
        for (int i = 0; i < b; i++) {
            for (int j = 0; j < b; j++) {
                if (C[i * n + j] > A[i * n + k] + B[k * n + j]) {
                    C[i * n + j] = A[i * n + k] + B[k * n + j];
                }
            }
        }
    }
}

void floyd_warshall_blocked(int *matrix, const int n, const int b) {
    // for now, assume b divides n
    const int blocks = n / b;

    // note that [i][j] == [i * input_width * block_width + j * block_width]
    for (int k = 0; k < blocks; k++) {
        floyd_warshall_in_place(&matrix[k * b * n + k * b], &matrix[k * b * n + k * b], &matrix[k * b * n + k * b], b, n);

#pragma omp parallel for
        for (int j = 0; j < blocks; j++) {
            if (j == k)
                continue;
            floyd_warshall_in_place(&matrix[k * b * n + j * b], &matrix[k * b * n + k * b], &matrix[k * b * n + j * b], b, n);
        }

#pragma omp parallel for
        for (int i = 0; i < blocks; i++) {
            if (i == k)
                continue;
            floyd_warshall_in_place(&matrix[i * b * n + k * b], &matrix[i * b * n + k * b], &matrix[k * b * n + k * b], b, n);
            for (int j = 0; j < blocks; j++) {
                if (j == k)
                    continue;
                floyd_warshall_in_place(&matrix[i * b * n + j * b], &matrix[i * b * n + k * b], &matrix[k * b * n + j * b], b, n);
            }
        }
    }
}

// void floyd_warshall_blocked(const int *input, int *output, const int n, const int b) {
//     memcpy(output, input, n * n * sizeof(int));

//     // for now, assume b divides n
//     const int blocks = n / b;

//     // note that [i][j] == [i * input_width * block_width + j * block_width]
//     for (int k = 0; k < blocks; k++) {
//         floyd_warshall_in_place(&output[k * b * n + k * b], &output[k * b * n + k * b], &output[k * b * n + k * b], b, n);

// #pragma omp parallel for
//         for (int j = 0; j < blocks; j++) {
//             if (j == k)
//                 continue;
//             floyd_warshall_in_place(&output[k * b * n + j * b], &output[k * b * n + k * b], &output[k * b * n + j * b], b, n);
//         }

// #pragma omp parallel for
//         for (int i = 0; i < blocks; i++) {
//             if (i == k)
//                 continue;
//             floyd_warshall_in_place(&output[i * b * n + k * b], &output[i * b * n + k * b], &output[k * b * n + k * b], b, n);
//             for (int j = 0; j < blocks; j++) {
//                 if (j == k)
//                     continue;
//                 floyd_warshall_in_place(&output[i * b * n + j * b], &output[i * b * n + k * b], &output[k * b * n + j * b], b, n);
//             }
//         }
//     }
// }