#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <cstring>

double cudaForwardPassTimer (float* x, float* y, float* U_host, float* V_host, float* W_host,
                              float* b_host, float* c_host, int vsize, int hsize, int T);

double cudaSequentialForwardPassTimer (float* x, float* y, float* U_host, float* V_host, float* W_host,
                              float* b_host, float* c_host, int vsize, int hsize, int T);

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
                    float* c, int hsize, int vsize) {
    
    for (int i1 = 0; i1 < max(hsize, vsize); i1++) {

        // fprintf(stdout, "%d of %d outerloop: begun\n", i1, max(hsize, vsize));
        // fflush(stdout);

        for (int i2 = 0; i2 < max(hsize, vsize); i2++) {
            if (i1 < hsize && i2 < vsize) {
                supply_rand_val(&U[i1*vsize + i2]);
            }
            if (i1 < vsize && i2 < hsize) {
                supply_rand_val(&V[i1*hsize + i2]);
            }
            if (i1 < hsize && i2 < hsize) {
                supply_rand_val(&W[i1*hsize + i2]);
            }
        }

        // fprintf(stdout, "%d of %d outerloop: completed innerloop\n", i1, max(hsize, vsize));
        // fflush(stdout);

        if (i1 < hsize) {
            supply_rand_val(&b[i1]);
        }
        if (i1 < vsize) {
            supply_rand_val(&c[i1]);
        }
    }

}

void init_input_values (float* x, int vsize, int timesteps) {

    for (int i = 0; i < vsize * timesteps; i++) {
        supply_rand_val(&x[i]);
    }

}

float test (int hsize, int vsize, int timesteps, int use_sequential) {

    float* x = (float*) calloc(vsize * timesteps, sizeof(float));
    float* y = (float*) calloc(vsize * timesteps, sizeof(float));
    float* U = (float*) calloc(hsize * vsize, sizeof(float));
    float* V = (float*) calloc(vsize * hsize, sizeof(float));
    float* W = (float*) calloc(hsize * hsize, sizeof(float));
    float* b = (float*) calloc(hsize, sizeof(float));
    float* c = (float*) calloc(vsize, sizeof(float));

    init_param_values(U, V, W, b, c, hsize, vsize); // random generation of values that
                                                    // could plausibly be computed during training

    init_input_values(x, vsize, timesteps);
    
    float duration  = 0;
    if (use_sequential) {
        duration = cudaSequentialForwardPassTimer (x, y, U, V, W, b, c, 
                                                vsize, hsize, timesteps);
    } else { 
        duration = cudaForwardPassTimer (x, y, U, V, W, b, c, 
                                                vsize, hsize, timesteps);
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

    int hsize = 30;
    int vsize = 8000; // change for testing; these are typical
    int timesteps = 5;

    fprintf(stdout, "beginning test ... \n");
    fflush(stdout);
    float duration = test(hsize, vsize, timesteps, 0);
    fprintf(stdout, "result duration (parallel): %f\n", duration);
    fflush(stdout);

    fprintf(stdout, "beginning test ... \n");
    fflush(stdout);
    duration = test(hsize, vsize, timesteps, 1);
    fprintf(stdout, "result duration (sequential): %f\n", duration);
    fflush(stdout);

    // print
    return 0;
}