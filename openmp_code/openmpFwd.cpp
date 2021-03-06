
#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <cstring>

#include <cmath>
#include <math.h>
#include <chrono>
#include <omp.h>

#define TIMESTEPS 900
#define HSIZE 10
#define VSIZE 800

// three functions copied over from setup code (on host)
// in our CUDA implementation (in the ../cuda_code folder)
int max (int a, int b) {
    if (a > b) {
        return a;
    }
    return b;
}

int min (int a, int b) {
    if (max(a, b) == a) {
        return b;
    }
    
    return a;
}

void supply_rand_val (float* addr) {
    float ret = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    *addr = ret;
}

/* the next three functions are documented in 
  ../cuda_code folder (in the rnnFwd.cu file)
*/
void matmul (float* A, float* B, float* AB, int a, int b, int c) {

    for (int i = 0; i < a; i++) {
        for (int k = 0; k < c; k++) {
            int ab_result_ik = 0;
            for (int j = 0; j < b; j++) {
                ab_result_ik += (A[i*b + j]*B[j*c + k]);
            }
            AB[i*c + k] = ab_result_ik;
        }
    }
    
}

void getExcerpt (float* A, float* dest, int timestep, int dimsize) {
    memcpy(dest, &A[timestep*dimsize], dimsize*sizeof(float));
    return;
}

void setExcerpt (float* A, float* src, int timestep, int dimsize) {
    for (int i = 0; i < dimsize; i++) {
        # pragma omp atomic write
            A[timestep * dimsize + i] = src[i];
    }
}

// a looping setExcerpt, used to propagate the values from a
// particular timestep's hidden layer computation to
// that of index_adv subsequent hidden layers
void forwardFill (float* A, float* src, int timestep, int dimsize,
                  int index_adv) {
    int fill_til = timestep + index_adv;
    for (int t = timestep; t < min(fill_til, TIMESTEPS); t++) {
        setExcerpt(A, src, t, dimsize);
    }
}

/* the next three functions are documented in 
  ../cuda_code folder (in the rnnFwd.cu file)
*/
void vectorAdd (float* A, float* B, float* AplusB, int a, int b) {
    for (int i = 0; i < a; i++) {
        for (int j = 0; j < b; j++) {
            AplusB[i*b + j] = A[i*b + j] + B[i*b + j];
        }
    }
}

void vectorTanh (float* A, float* ret, int a, int b) {
    for (int i = 0; i < a; i++) {
        for (int j = 0; j < b; j++) {
            ret[i] = std::tanh(A[i*a + j]);
        }
    }
}

void vectorSoftmax(float* src, float* dest, int a, int b) {
    float denom = 0;
    
    //PROBLEM if b is not 1 here
    
    for (int i = 0; i < a; i++) {
        denom += exp(src[i]);
    }

    for (int i = 0; i < a; i++) {
        dest[i] = src[i] / denom;
    }

    return;
}

// the function that completes each thread's forward-pass computations,
// called from within a mainloop that computes the output neurons
// for all timesteps' data
// importantly, par_flag distinguishes whether the exeuction should be
// sequential (when it's 0) or parallel (when it's 1)
float* openmp_fwd_pass (float* all_h, int t, float* b, float* U, 
                        float* V, float* W, float* x, float* c,
                        int* timestep_wait_vector, int par_flag) {

    float* h_tminus1 = (float*) calloc(HSIZE, sizeof(float));
    getExcerpt(all_h, h_tminus1, t-1, HSIZE);
    
    int spin_val = 0;
    if (par_flag) {
        # pragma omp atomic read
            spin_val = timestep_wait_vector[t] ;
        while (spin_val == 0) { 
            for (int i = 0; i < 1000000; i++) ;
            # pragma omp atomic read
                spin_val = timestep_wait_vector[t] ;
        } 
    }

    float* a_t = (float*) calloc(HSIZE, sizeof(float));
    float* x_t = x + HSIZE*t;

    float* W_h = (float*) calloc (HSIZE, sizeof(float));
    matmul(W, h_tminus1, W_h, HSIZE, HSIZE, 1);
    float* U_x = (float*) calloc (HSIZE, sizeof(float));
    matmul(U, x_t, U_x, HSIZE, VSIZE, 1);
    float* add1 = (float*) calloc(HSIZE, sizeof(float));
    vectorAdd(U_x, W_h, add1, HSIZE, 1);
    vectorAdd(add1, b, a_t, HSIZE, 1);

    // free(add1);
    // free(h_tminus1);
    // free(U_x);
    // free(W_h);

    float* h_t = (float*) calloc(HSIZE, sizeof(float));
    vectorTanh(a_t, h_t, HSIZE, 1);
    if (par_flag) {
        forwardFill(all_h, h_t, t, HSIZE, 15);
    } else {
        setExcerpt(all_h, h_t, t, HSIZE);
    }
    
    if (par_flag) {
        if (t < TIMESTEPS) {
            for (int i = 0; i < 15; i++) {
                # pragma omp atomic update
                    timestep_wait_vector[t + 1 + i] ++;
            }
        }
    }

    // fprintf(stderr, "timestep %d: completed setting h\n", t);

    float* o_t = (float*) calloc(VSIZE, sizeof(float));
    float* V_h = (float*) calloc(VSIZE, sizeof(float));
    matmul(V, h_t, V_h, VSIZE, HSIZE, 1);
    vectorAdd(c, V_h, o_t, VSIZE, 1);

    float* y_t = (float*) calloc(VSIZE, sizeof(float));
    vectorSoftmax(o_t, y_t, VSIZE, 1);

    // free(o_t);
    // free(V_h);
    // free(h_t);

    return y_t;

}

/*
  next two functions are copied from ../cuda_code/main.cpp
  they initialize random values (random as per the cpp library)
  to mimic plausible inputs and parameter values computed 
  during training time
 */
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

// the mainloop that executes the forward pass both in 
// parallel and sequentially
int main(int argc, const char *argv[]) {

    int num_of_threads = atoi(argv[1]);

    float* x = (float*) calloc(VSIZE * TIMESTEPS, sizeof(float));
    // float* y = (float*) calloc(VSIZE * TIMESTEPS, sizeof(float));
    float* U = (float*) calloc(HSIZE * VSIZE, sizeof(float));
    float* V = (float*) calloc(VSIZE * HSIZE, sizeof(float));
    float* W = (float*) calloc(HSIZE * HSIZE, sizeof(float));
    float* b = (float*) calloc(HSIZE, sizeof(float));
    float* c = (float*) calloc(VSIZE, sizeof(float));

    init_input_values(x);
    init_param_values(U, V, W, b, c); // random generation of values that
                                                    // could plausibly be computed during training


    // alloc h
    float* all_h = (float*) calloc( TIMESTEPS * HSIZE , sizeof(float) );

    int* timestep_wait_vector = (int*) calloc(TIMESTEPS, sizeof(int));
    float* h_0_vals = (float*) calloc(HSIZE, sizeof(float));
    memset(h_0_vals, 1, sizeof(float)*HSIZE);
    setExcerpt(all_h, h_0_vals, 0, HSIZE);
    free(h_0_vals);
    timestep_wait_vector[0] = 1;

    using namespace std::chrono;
    typedef std::chrono::high_resolution_clock Clock;
    typedef std::chrono::duration<double> dsec;

    auto loop_start = Clock::now();
    double loop_time = 0.f;

    // #pragma omp parallel for default(shared) schedule(static)
    for (int t = 0; t < TIMESTEPS; t++) {
        float* y_t = openmp_fwd_pass(all_h, t, b, U, V, W, x, c,
                                      timestep_wait_vector, 0);
        //setExcerpt(y, y_t, t, VSIZE);
        free (y_t);
    }

    loop_time += duration_cast<dsec>(Clock::now() - loop_start).count();
    fprintf(stderr, "seq time: %f\n", loop_time);

    auto loop_start1 = Clock::now();
    double loop_time1 = 0.f;

    omp_set_num_threads(num_of_threads);
    #pragma omp parallel for default(shared) schedule(static)
    for (int t = 0; t < TIMESTEPS; t++) {
        float* y_t = openmp_fwd_pass(all_h, t, b, U, V, W, x, c, 
                                      timestep_wait_vector, 1);
        //setExcerpt(y, y_t, t, VSIZE);
        free (y_t);
    }

    loop_time1 += duration_cast<dsec>(Clock::now() - loop_start1).count();
    fprintf(stderr, "par time: %f\n", loop_time1);

}