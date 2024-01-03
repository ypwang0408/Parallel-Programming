#include "config.h"
#include "util.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void floydWarshall(int *matrix, int n);

int main(int argc, char **argv) {
    uint n, density;
    if (argc <= 2) {
        n = DEFAULT;
        density = 100;
    } else {
        n = atoi(argv[1]);
        density = atoi(argv[2]);
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
    floydWarshall(matrix, n);
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    accum = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1000000.0;

    if (PRINTABLE) {
        printf("*** The solution is:\n");
        showDistances(matrix, n);
    }

    printf("[SEQUENTIAL] Total elapsed time %.2lf ms\n", accum);
    free(matrix);

    FILE *pFile;
    pFile = fopen("./result/sequential_result.txt", "a");
    if (NULL == pFile) {
        puts("open failure");
        exit(1);
    } else {
        fprintf(pFile, "%d\t%d\t%.2lf\n", n, density, accum);
    }

    fclose(pFile);
    return 0;
}

void floydWarshall(int *matrix, int n) {
    int i, j, k;
    for (k = 0; k < n; k++) {
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                int newPath = matrix[i * n + k] + matrix[k * n + j];
                if (matrix[i * n + j] > newPath) {
                    matrix[i * n + j] = newPath;
                }
            }
        }
    }
}
