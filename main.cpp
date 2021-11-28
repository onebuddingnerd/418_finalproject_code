#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <cstring>

int main() {
    int N = 30;
    float* x = malloc(sizeof(float) * N);
    float* y = malloc(sizeof(float) * N);

    cudaForwardPassTimer (x, N, );

    // timing stuff

    return 0;
}