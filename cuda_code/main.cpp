#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <cstring>

#define TIMESTEPS 5
#define HSIZE 50
#define VSIZE 8000

double cudaForwardPassTimer (float* x, float* y, float* U_host, float* V_host, float* W_host,
                              float* b_host, float* c_host);

double cudaSequentialForwardPassTimer (float* x, float* y, float* U_host, float* V_host, float* W_host,
                              float* b_host, float* c_host);

void supply_rand_val (float* addr) {
    float ret = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    *addr = ret;
}

int max (int a, int b) {
    if (a > b) {
        return a;
    }
    return b;
}

void init_param_values (float* U, float* V, float* W, float* b,
                    float* c) {
    
    for (int i1 = 0; i1 < max(HSIZE, VSIZE); i1++) {

        for (int i2 = 0; i2 < max(HSIZE, VSIZE); i2++) {
            if (i1 < HSIZE && i2 < VSIZE) {
                supply_rand_val(&U[i1*VSIZE + i2]);
            }
            if (i1 < VSIZE && i2 < HSIZE) {
                supply_rand_val(&V[i1*HSIZE + i2]);
            }
            if (i1 < HSIZE && i2 < HSIZE) {
                supply_rand_val(&W[i1*HSIZE + i2]);
            }
        }

        if (i1 < HSIZE) {
            supply_rand_val(&b[i1]);
        }
        if (i1 < VSIZE) {
            supply_rand_val(&c[i1]);
        }
    }

}

void init_input_values (float* x) {

    for (int i = 0; i < VSIZE * TIMESTEPS; i++) {
        supply_rand_val(&x[i]);
    }

}

float test (int use_sequential) {

    float* x = (float*) calloc(VSIZE * TIMESTEPS, sizeof(float));
    float* y = (float*) calloc(VSIZE * TIMESTEPS, sizeof(float));
    float* U = (float*) calloc(HSIZE * VSIZE, sizeof(float));
    float* V = (float*) calloc(VSIZE * HSIZE, sizeof(float));
    float* W = (float*) calloc(HSIZE * HSIZE, sizeof(float));
    float* b = (float*) calloc(HSIZE, sizeof(float));
    float* c = (float*) calloc(VSIZE, sizeof(float));

    init_param_values(U, V, W, b, c); // random generation of values that
                                                    // could plausibly be computed during training

    init_input_values(x);
    
    float duration = 0;
    if (use_sequential) {
        duration = cudaSequentialForwardPassTimer (x, y, U, V, W, b, c);
    } else { 
        duration = cudaForwardPassTimer (x, y, U, V, W, b, c);
    }

    free(x);
    free(U);
    free(V);
    free(W);
    free(b);
    free(c);

    return duration;

}

int main() {

    fprintf(stdout, "beginning test ... \n");
    fflush(stdout);
    float duration = test(0);
    fprintf(stdout, "result duration (parallel): %f\n", duration);
    fflush(stdout);

    fprintf(stdout, "beginning test ... \n");
    fflush(stdout);
    duration = test(1);
    fprintf(stdout, "result duration (sequential): %f\n", duration);
    fflush(stdout);

    // print
    return 0;
}