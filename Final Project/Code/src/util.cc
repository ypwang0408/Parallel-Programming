#include "util.h"
#include "config.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int iDivUp(int a, int b) {
    int result = ceil(1.0 * a / b);
    if (result < 1) {
        return 1;
    } else {
        return result;
    }
}

void showDistances(int matrix[], int n) {
    int i, j;
    printf("     ");
    for (i = 0; i < n; i++) {
        printf("[%d]  ", i);
    }
    printf("\n");
    for (i = 0; i < n; i++) {
        printf("[%d]", i);
        for (j = 0; j < n; j++) {
            if (matrix[i * n + j] == INF) {
                printf("  inf");
            } else {
                printf("%5d", matrix[i * n + j]);
            }
        }
        printf("\n");
    }
    printf("\n");
}

void populateMatrix(int *matrix, int n, int density) {
    int i, j, value;
    srand(RANDOM_SEED);
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (i == j) {
                matrix[i * n + j] = 0;
            } else {
                value = 1 + rand() % MAX;
                if (value > density) {
                    matrix[i * n + j] = INF;
                } else {
                    matrix[i * n + j] = value;
                }
            }
        }
    }
}