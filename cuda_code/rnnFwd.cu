
#include <cmath>
#include <math.h>
#include "CycleTimer.h"

#define DEBUG
#ifdef DEBUG
#define cudaCheckError(ans) cudaAssert((ans), __FILE__, __LINE__);
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n",
        cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
#else
#define cudaCheckError(ans) ans
#endif

#define TIMESTEPS 5
#define HSIZE 30
#define VSIZE 8000

// A is a by b
// B is b by c
// AB is a by c

__device__
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

__device__
void getExcerpt (float* A, float* dest, int timestep, int dimsize) {
    memcpy(dest, &A[timestep*dimsize], dimsize*sizeof(float));
    return;
}

__device__ 
float atomicCAS_f32(float *p, float cmp, float val) {
    return __int_as_float(atomicCAS((int *) p, __float_as_int(cmp), __float_as_int(val)));
}

__device__
void setExcerpt (float* A, float* src, int timestep, int dimsize) {
    for (int i = 0; i < dimsize; i++) {
        atomicExch(&A[timestep * dimsize + i], 5);
    }
    return;
}

__device__
void vectorAdd (float* A, float* B, float* AplusB, int a, int b) {
    for (int i = 0; i < a; i++) {
        for (int j = 0; j < b; j++) {
            AplusB[i*a + j] = A[i*a + j] + B[i*a + j];
        }
    }
}

__device__
void vectorTanh (float* A, float* ret, int a, int b) {
    for (int i = 0; i < a; i++) {
        for (int j = 0; j < b; j++) {
            ret[i] = std::tanh(A[i*a + j]);
        }
    }
}

__device__
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

__device__ 
void initOnes (float* A, int a, int b) {
    for (int i = 0; i < a*b; i++) {
        A[i] = 1;
    }
    return;
}

void host_initOnes (float* A, int a, int b) {
    for (int i = 0; i < a*b; i++) {
        A[i] = 1;
    }
    return;
}

typedef long long cycles_t;
__device__ void kernelSleep(cycles_t sleep_cycles)
{
    cycles_t start = clock64();
    cycles_t cycles_elapsed;
    do { cycles_elapsed = clock64() - start; } 
    while (cycles_elapsed < sleep_cycles);
}

__device__
void print_excerpt(float* A, int start, int til) {
    for ( int i = start; i < til; i++ ) {
        printf("%f ", A[i]);
    }
    printf("\n");

    return;
}

/*
 * reference: https://gist.github.com/PolarNick239/5e370892989800fe42154145911b141f
 * this funciton allows us to use atomicCAS with floats
 */
__device__ 
float atomicCAS_f32_verbose(float *p, float cmp, float val, int thd_index) { 
    if (thd_index == 2) {
        printf("thd 2 calling CAS wrapper-------------------------------------\n");
    } else {
        printf("thd %d calling CAS wrapper\n", thd_index);
    }
	float ret = __int_as_float(atomicCAS((int *) p, __float_as_int(cmp), __float_as_int(val)));
    if (thd_index == 2) {
        printf("thd 2 completed CAS wrapper-------------------------------------\n");
    } else {
        printf("thd %d completed CAS wrapper\n", thd_index);
    }
    return ret;
}

__global__
void kernelComputeForward (float* device_x, float* device_a, float* device_h, float* device_o, 
                            float* device_y, float* U, float* V, float* W, float* b, float* c,
                            float* h_tminus1_all, float* W_h_all, float* U_x_all, float* add1_all,
                            float* a_t_all, float* h_t_all, float* V_h_all, float* o_t_all, float* y_t_all,
                            int sequential_index) {
    int index = 0;
    
    if (sequential_index > -1) {
        index = sequential_index;
    } else {
        index = blockIdx.x * blockDim.x + threadIdx.x; // time index
        if (index >= TIMESTEPS || index == 0) {
            return;
        }
    }

    float* h_tminus1 = &h_tminus1_all[index*HSIZE];
    float* W_h = &W_h_all[index*VSIZE];
    float* U_x = &U_x_all[index*HSIZE];
    float* add1 = &add1_all[index*HSIZE];
    float* a_t = &a_t_all[index*VSIZE];
    float* h_t = &h_t_all[index*HSIZE];
    float* V_h = &a_t_all[index*VSIZE];
    float* o_t = &o_t_all[index*HSIZE];
    float* y_t = &y_t_all[index*VSIZE];

    float* x_t = &device_x[index*VSIZE];

    // note: now threadsPerBlock = TIMESTEPS
    __shared__ float h_shared[HSIZE * TIMESTEPS];

    if (index > 0) {
        while ( atomicCAS_f32(&h_shared[index * (HSIZE - 1)], -1.f, -1.f) == -1 ) ;
    }  // spin til prev timestep's h-level vector has nondefault value (which indicates that the computation is pending

    
     
    // a[t] = b + W * h[t-1] + U * x[t]
    // h[t] = tanh(a[t])
    // o[t] = c + V * h[t]
    // y[t] = softmax(o[t])
    
    // W_h_term =  device_W ** device_h[index * HSIZE : (index+1) * HSIZE]
    // U_x_term = device_U ** device_x[index * VSIZE : (index+1) * VSIZE]
    // device_a[index * HSIZE : (index+1) * HSIZE] = device_b + 
    //                  W_h_term + U_x_term

    // 1a. the W_h term
    if (index >= 1) {
        getExcerpt(device_h, h_tminus1, index-1, HSIZE);
    } else {
        initOnes(h_tminus1, HSIZE, 1);
    }   
    matmul(W, h_tminus1, W_h, HSIZE, HSIZE, 1);
    // 1b. the ux term
    getExcerpt(device_x, x_t, index, VSIZE);
    matmul(U, x_t, U_x, HSIZE, VSIZE, 1);
    // 1: addition of b
    vectorAdd(U_x, b, add1, HSIZE, 1);
    vectorAdd(add1, W_h, a_t, HSIZE, 1);

    // free(h_tminus1);
    // free(W_h);
    // free(U_x);
    // free(add1);
    // setExcerpt(device_a, a_t, index, HSIZE);
    // free(a_t) LATER

    // a_t has the result for the next layer (h)
    // 2a: tanh of vector for a_t
    vectorTanh(a_t, h_t, HSIZE, 1);\
    // 2b: set the excerpt
    for (int i = 0; i < HSIZE; i++) {
        h_shared[TIMESTEPS*index + i] = h_t[i];
    }

    // h_t has the result for the next layer(o)
    // 3: addition of V*h and c
    matmul(V, h_t, V_h, VSIZE, HSIZE, 1);
    vectorAdd(c, V_h, o_t, HSIZE, 1);
    // free(o_t) LATER

    // o_t has the result for the next layer(o)
    // 4: softmax(o_t)
    vectorSoftmax(o_t, y_t, VSIZE, 1);
    setExcerpt(device_y, y_t, index, VSIZE);
    
    // free(a_t);
    // free(h_t);
    // free(o_t);
    // free(y_t);
    return;

}


void forwardPass (float* device_x, float* device_a, float* device_h, float* device_o, 
                  float* device_y, float* U, float* V, float* W, float* b, float* c) {
    const int threadsPerBlock = TIMESTEPS;
    const int blocks = 1;

    // intermediate term
    float* h_tminus1; // = (float*) calloc(HSIZE, sizeof(float));
    cudaMalloc((void**) &h_tminus1, HSIZE * TIMESTEPS * sizeof(float));
    float* W_h; // = (float*) calloc(HSIZE, sizeof(float));
    cudaMalloc((void**) &W_h, HSIZE * sizeof(float));
    //float* x_t; // = (float*) calloc(VSIZE, sizeof(float));
    //cudaMalloc((void**) &x_t, VSIZE * TIMESTEPS * sizeof(float));
    float* U_x; // = (float*) calloc(HSIZE, sizeof(float));
    cudaMalloc((void**) &U_x, HSIZE * TIMESTEPS * sizeof(float));
    float* add1; //= //(float*) calloc(HSIZE, sizeof(float));
    cudaMalloc((void**) &add1, HSIZE * sizeof(float));
    float* a_t; //= (float*) calloc(VSIZE, sizeof(float));
    cudaMalloc((void**) &a_t, VSIZE * TIMESTEPS * sizeof(float));
    float* h_t; // = (float*) calloc(HSIZE, sizeof(float));
    cudaMalloc((void**) &h_t, HSIZE * TIMESTEPS * sizeof(float));
    float* V_h; // = (float*) calloc(VSIZE, sizeof(float));
    cudaMalloc((void**) &V_h, VSIZE * TIMESTEPS * sizeof(float));
    float* o_t;  //= (float*) calloc(HSIZE, sizeof(float));
    cudaMalloc((void**) &o_t, HSIZE * TIMESTEPS * sizeof(float));
    float* y_t; //= (float*) calloc(VSIZE, sizeof(float));
    cudaMalloc((void**) &y_t, VSIZE * TIMESTEPS * sizeof(float));  

    kernelComputeForward<<< blocks, threadsPerBlock >>>(device_x, device_a, device_h, 
                                                        device_o, device_y, U, V, W, b, c, 
                                                        h_tminus1, W_h, U_x,
                                                        add1, a_t, h_t, V_h, o_t, y_t,
                                                        -1);
}

void forwardPassSequential (float* device_x, float* device_a, float* device_h, float* device_o, 
                  float* device_y, float* U, float* V, float* W, float* b, float* c) {
    const int threadsPerBlock = 1;
    const int blocks = 1;

    // intermediate terms
    float* h_tminus1 = (float*) calloc(HSIZE, sizeof(float));
    float* W_h = (float*) calloc(HSIZE, sizeof(float));
    // float* x_t = (float*) calloc(VSIZE, sizeof(float));
    float* U_x = (float*) calloc(HSIZE, sizeof(float));
    float* add1 = (float*) calloc(HSIZE, sizeof(float));
    float* a_t = (float*) calloc(VSIZE, sizeof(float));
    float* h_t = (float*) calloc(HSIZE, sizeof(float));
    float* V_h = (float*) calloc(VSIZE, sizeof(float));
    float* o_t = (float*) calloc(HSIZE, sizeof(float));
    float* y_t = (float*) calloc(VSIZE, sizeof(float));

    for (int i = 0; i < TIMESTEPS; i++) {
        kernelComputeForward<<< blocks, threadsPerBlock >>>(device_x, device_a, device_h, 
                                                            device_o, device_y, U, V, W, b, c, 
                                                            h_tminus1, W_h, U_x,
                                                            add1, a_t, h_t, V_h, o_t, y_t,
                                                            i);
    }
}

__global__
void kernelSetToValAfter (float* A, int n, float val, int offset) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n || index < offset) {
        return;
    }
    A[index] = val;
}

void setToValAfter (float* A, int n, float val, int offset) {
    const int threadsPerBlock = 512;
    const int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    kernelSetToValAfter<<< blocks, threadsPerBlock >>>(A, n, val, offset);

}

double cudaForwardPassTimer (float* x, float* y, float* U_host, float* V_host, float* W_host,
                              float* b_host, float* c_host) {
    float* device_x;
    cudaMalloc((void **)&device_x, sizeof(float) * VSIZE * TIMESTEPS);
    cudaMemcpy(device_x, x, VSIZE * TIMESTEPS * sizeof(float), cudaMemcpyHostToDevice);

    /* 
      shapes:
      inputs/intermediate/outputs:
        x: VSIZE    by        1
        a: HSIZE    by        1
        h: HSIZE    by        1
        o: VSIZE    by        1
        y: VSIZE    by        1

      parameters:
        b: HSIZE    by        1
        c: VSIZE    by        1
        U: HSIZE    by        VSIZE
        V: VSIZE    by        HSIZE
        W: HSIZE    by        HSIZE

    */

    // alloc result (and intermediate result) destinations
    float* device_a;
    // float* device_h;
    float* h = (float*) calloc(HSIZE * TIMESTEPS, sizeof(float));
    host_initOnes(h, HSIZE, 1);
    float* device_o;
    float* device_y;
    cudaMalloc((void**) &device_a, sizeof(float) * HSIZE * TIMESTEPS);
    // cudaMalloc((void**) &device_h, sizeof(float) * HSIZE * TIMESTEPS);
    cudaMalloc((void**) &device_o, sizeof(float) * VSIZE * TIMESTEPS);
    cudaMalloc((void**) &device_y, sizeof(float) * VSIZE * TIMESTEPS);

    // set all hidden layers' values (across all TIMESTEPS) to -1 to enforce waiting
    float* device_h;
    cudaMalloc((void **) &device_h, sizeof(float) * HSIZE * TIMESTEPS);
    cudaMemcpy(device_h, h, sizeof(float) * HSIZE * TIMESTEPS, cudaMemcpyHostToDevice);
    setToValAfter(device_h, HSIZE * TIMESTEPS, -1.f, HSIZE);

    // declare, alloc, and copy onto device the init paramters
    float* b;
    cudaMalloc((void**) &b, HSIZE * sizeof(float));
    cudaMemcpy(b, b_host, HSIZE * sizeof(float), cudaMemcpyHostToDevice);
    float* c;
    cudaMalloc((void**) &c, VSIZE * sizeof(float));
    cudaMemcpy(c, c_host, VSIZE * sizeof(float), cudaMemcpyHostToDevice);
    float* U;
    cudaMalloc((void**) &U, HSIZE * VSIZE * sizeof(float));
    cudaMemcpy(U, U_host, HSIZE * VSIZE * sizeof(float), cudaMemcpyHostToDevice);
    float* W;
    cudaMalloc((void**) &W, HSIZE * HSIZE * sizeof(float));
    cudaMemcpy(W, W_host, HSIZE * HSIZE * sizeof(float), cudaMemcpyHostToDevice);
    float* V;
    cudaMalloc((void**) &V, VSIZE * HSIZE * sizeof(float));
    cudaMemcpy(V, V_host, VSIZE * HSIZE * sizeof(float), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();
    forwardPass(device_x, device_a, device_h, device_o, device_y,
                U, V, W, b, c);
    // cudaThreadSynchronize();
    cudaCheckError( cudaDeviceSynchronize() );
    double endTime = CycleTimer::currentSeconds();
    cudaMemcpy(y, device_y, VSIZE * TIMESTEPS * sizeof(float), cudaMemcpyDeviceToHost);
    double overallDuration = endTime - startTime;
    
    return overallDuration;
}

double cudaSequentialForwardPassTimer (float* x, float* y, float* U_host, float* V_host, float* W_host,
                              float* b_host, float* c_host) {
    float* device_x;
    cudaMalloc((void **)&device_x, sizeof(float) * VSIZE * TIMESTEPS);
    cudaMemcpy(device_x, x, VSIZE * TIMESTEPS * sizeof(float), cudaMemcpyHostToDevice);

    /* 
      shapes:
      inputs/intermediate/outputs:
        x: VSIZE    by        1
        a: HSIZE    by        1
        h: HSIZE    by        1
        o: VSIZE    by        1
        y: VSIZE    by        1

      parameters:
        b: HSIZE    by        1
        c: VSIZE    by        1
        U: HSIZE    by        VSIZE
        V: VSIZE    by        HSIZE
        W: HSIZE    by        HSIZE

    */

    // alloc result (and intermediate result) destinations
    float* device_a;
    float* device_h;
    float* device_o;
    float* device_y;
    cudaMalloc((void**) &device_a, sizeof(float) * HSIZE * TIMESTEPS);
    cudaMalloc((void**) &device_h, sizeof(float) * HSIZE * TIMESTEPS);
    cudaMalloc((void**) &device_o, sizeof(float) * VSIZE * TIMESTEPS);
    cudaMalloc((void**) &device_y, sizeof(float) * VSIZE * TIMESTEPS);

    // set all hidden layers' values (across all TIMESTEPS) to -1 to enforce waiting
    setToValAfter(device_h, HSIZE * TIMESTEPS, -1.f, 0); // CHANGE!!!!

    // declare, alloc, and copy onto device the init paramters
    float* b;
    cudaMalloc((void**) &b, HSIZE * sizeof(float));
    cudaMemcpy(b, b_host, HSIZE * sizeof(float), cudaMemcpyHostToDevice);
    float* c;
    cudaMalloc((void**) &c, VSIZE * sizeof(float));
    cudaMemcpy(c, c_host, VSIZE * sizeof(float), cudaMemcpyHostToDevice);
    float* U;
    cudaMalloc((void**) &U, HSIZE * VSIZE * sizeof(float));
    cudaMemcpy(U, U_host, HSIZE * VSIZE * sizeof(float), cudaMemcpyHostToDevice);
    float* W;
    cudaMalloc((void**) &W, HSIZE * HSIZE * sizeof(float));
    cudaMemcpy(W, W_host, HSIZE * HSIZE * sizeof(float), cudaMemcpyHostToDevice);
    float* V;
    cudaMalloc((void**) &V, VSIZE * HSIZE * sizeof(float));
    cudaMemcpy(V, V_host, VSIZE * HSIZE * sizeof(float), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();
    forwardPassSequential(device_x, device_a, device_h, device_o, device_y,
                U, V, W, b, c);
    cudaThreadSynchronize();
    double endTime = CycleTimer::currentSeconds();
    cudaMemcpy(y, device_y, VSIZE * TIMESTEPS * sizeof(float), cudaMemcpyDeviceToHost);
    double overallDuration = endTime - startTime;
    
    return overallDuration;
}