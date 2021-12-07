
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
    printf("excerptgetting attempt, at index %d (max is %d for array)\n", timestep*dimsize, dimsize*5);
    memcpy(dest, &A[timestep*dimsize], dimsize*sizeof(float));
    printf("%d index excerptgetting attempt complete\n", timestep*dimsize);
    return;
}

__device__ 
float atomicCAS_f32(float *p, float cmp, float val) {
    return __int_as_float(atomicCAS((int *) p, __float_as_int(cmp), __float_as_int(val)));
}

__device__
void setExcerpt (float* A, float* src, int timestep, int dimsize) {
    printf("starting setexcerpt with %d vals to be set\n", dimsize);
    for (int i = 0; i < dimsize; i++) {
        // A[timestep * dimsize + i] = src[i];
        // while (atomicCAS_f32(&A[timestep * dimsize + i], -1, 5) != -1) ;
        atomicExch(&A[timestep * dimsize + i], 5);
    }
    // memcpy(& A[timestep * dimsize], src, dimsize * sizeof(float));
    printf("excerptsetting of %d vals complete\n", dimsize);
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
                            float* h_tminus1, float* W_h, float* x_t, float* U_x, float* add1,
                            float* a_t, float* h_t, float* V_h, float* o_t, float* y_t,
                            int hsize, int vsize, int timesteps, int threadsPerBlock, 
                            int sequential_index) {
    int index = 0;
    
    if (sequential_index > -1) {
        index = sequential_index;
    } else {
        index = blockIdx.x * blockDim.x + threadIdx.x; // time index
        if(index == 0){
            printf("Thread %d executing, timesteps: %d\n", index, timesteps);
        }
        if (index >= timesteps || index == 0) {
            return;
        }
    }

    // __shared__ float h_shared[hsize * threadsPerBlock];
    // if (index % threadsPerBlock == 0) {
    //     for (int i = 0; i < min(threadsPerBlock, timesteps - index)) {
    //         h_shared[i]
    //     }
    // }

    // __syncthreads();

    printf("Thread %d right before if statement\n", index);
    if (index > 0) {
        printf("At time index %d, (hsize - 1) * index = %d, device_h is: %f\n", index, index * (hsize - 1), device_h[index * (hsize - 1)]);
        // int loopcount = 0;
        // while (device_h[index * (hsize - 1)] == -1.f) {
        while ( atomicCAS_f32(&device_h[index * (hsize - 1)], -1.f, -1.f) == -1 ) {
            // printf("thd %d waiting for %f to change to naything other than -1\n", index, device_h[index * (hsize - 1)]);
            //spin_little();
            kernelSleep(10000);
        }
        // while ( atomicCAS(&device_h[index * (hsize - 1)], -1.f, -1.f) == -1 ) ; 
        printf("thd %d: exiting waitloop\n ", index);
        printf("thd %d: %f is checkval\n", index, device_h[index * (hsize - 1)] );
    }  // spin til prev timestep's h-level vector has nondefault value (which indicates that the computation is pending
    printf("Thread %d completed spin\n", index);
    
     
    // a[t] = b + W * h[t-1] + U * x[t]
    // h[t] = tanh(a[t])
    // o[t] = c + V * h[t]
    // y[t] = softmax(o[t])
    
    // W_h_term =  device_W ** device_h[index * hsize : (index+1) * hsize]
    // U_x_term = device_U ** device_x[index * vsize : (index+1) * vsize]
    // device_a[index * hsize : (index+1) * hsize] = device_b + 
    //                  W_h_term + U_x_term

    // 1a. the W_h term
    if (index >= 1) {
        getExcerpt(device_h, h_tminus1, index-1, hsize);
    } else {
        initOnes(h_tminus1, hsize, 1);
    }   
    printf("%d-indexed thd completed getExcerpt (of step 1 for global h)\n", index); 
    matmul(W, h_tminus1, W_h, hsize, hsize, 1);
    printf("%d-indexed thd completed matmul of W and h_tminus1\n", index);
    // 1b. the ux term
    getExcerpt(device_x, x_t, index, vsize);
    matmul(U, x_t, U_x, hsize, vsize, 1);
    // 1: addition of b
    vectorAdd(U_x, b, add1, hsize, 1);
    vectorAdd(add1, W_h, a_t, hsize, 1);
    printf("%d-indexed thd completed comp1\n", index);

    // free(h_tminus1);
    // free(W_h);
    // free(U_x);
    // free(add1);
    // setExcerpt(device_a, a_t, index, hsize);
    // free(a_t) LATER

    // a_t has the result for the next layer (h)
    // 2: tanh of vector for a_t
    vectorTanh(a_t, h_t, hsize, 1);
    setExcerpt(device_h, h_t, index, hsize);
    if (index == 1) {
        printf("printing setting of thd 1\n");
        print_excerpt(device_h, index*hsize, (index+1)*hsize);
    }
    printf("%d-indexed thd completed comp2\n", index);
    // free(h_t) LATER

    // h_t has the result for the next layer(o)
    // 3: addition of V*h and c
    matmul(V, h_t, V_h, vsize, hsize, 1);
    vectorAdd(c, V_h, o_t, hsize, 1);
    // setExcerpt(device_o, o_t, index, hsize);
    printf("%d-indexed thd completed comp3\n", index);
    // free(o_t) LATER

    // o_t has the result for the next layer(o)
    // 4: softmax(o_t)
    vectorSoftmax(o_t, y_t, vsize, 1);
    setExcerpt(device_y, y_t, index, vsize);
    printf("%d-indexed thd completed comp4\n", index);
    
    // free(a_t);
    // free(h_t);
    // free(o_t);
    // free(y_t);
    return;

}


void forwardPass (float* device_x, float* device_a, float* device_h, float* device_o, 
                  float* device_y, float* U, float* V, float* W, float* b, float* c,
                  int hsize, int vsize, int T) {
    const int threadsPerBlock = 2;
    const int blocks = (T + threadsPerBlock - 1) / threadsPerBlock;

    // intermediate term
    float* h_tminus1; // = (float*) calloc(hsize, sizeof(float));
    cudaMalloc((void**) &h_tminus1, hsize * sizeof(float));
    float* W_h; // = (float*) calloc(hsize, sizeof(float));
    cudaMalloc((void**) &W_h, hsize * sizeof(float));
    float* x_t; // = (float*) calloc(vsize, sizeof(float));
    cudaMalloc((void**) &x_t, vsize * sizeof(float));
    float* U_x; // = (float*) calloc(hsize, sizeof(float));
    cudaMalloc((void**) &U_x, hsize * sizeof(float));
    float* add1; //= //(float*) calloc(hsize, sizeof(float));
    cudaMalloc((void**) &add1, hsize * sizeof(float));
    float* a_t; //= (float*) calloc(vsize, sizeof(float));
    cudaMalloc((void**) &a_t, vsize * sizeof(float));
    float* h_t; // = (float*) calloc(hsize, sizeof(float));
    cudaMalloc((void**) &h_t, hsize * sizeof(float));
    float* V_h; // = (float*) calloc(vsize, sizeof(float));
    cudaMalloc((void**) &V_h, vsize * sizeof(float));
    float* o_t;  //= (float*) calloc(hsize, sizeof(float));
    cudaMalloc((void**) &o_t, hsize * sizeof(float));
    float* y_t; //= (float*) calloc(vsize, sizeof(float));
    cudaMalloc((void**) &y_t, vsize * sizeof(float)); 

    kernelComputeForward<<< blocks, threadsPerBlock >>>(device_x, device_a, device_h, 
                                                        device_o, device_y, U, V, W, b, c, 
                                                        h_tminus1, W_h, x_t, U_x,
                                                        add1, a_t, h_t, V_h, o_t, y_t,
                                                        hsize, vsize, T, threadsPerBlock,
                                                        -1);
}

void forwardPassSequential (float* device_x, float* device_a, float* device_h, float* device_o, 
                  float* device_y, float* U, float* V, float* W, float* b, float* c,
                  int hsize, int vsize, int T) {
    const int threadsPerBlock = 1;
    const int blocks = 1;

    // intermediate terms
    float* h_tminus1 = (float*) calloc(hsize, sizeof(float));
    float* W_h = (float*) calloc(hsize, sizeof(float));
    float* x_t = (float*) calloc(vsize, sizeof(float));
    float* U_x = (float*) calloc(hsize, sizeof(float));
    float* add1 = (float*) calloc(hsize, sizeof(float));
    float* a_t = (float*) calloc(vsize, sizeof(float));
    float* h_t = (float*) calloc(hsize, sizeof(float));
    float* V_h = (float*) calloc(vsize, sizeof(float));
    float* o_t = (float*) calloc(hsize, sizeof(float));
    float* y_t = (float*) calloc(vsize, sizeof(float));

    for (int i = 0; i < T; i++) {
        kernelComputeForward<<< blocks, threadsPerBlock >>>(device_x, device_a, device_h, 
                                                            device_o, device_y, U, V, W, b, c, 
                                                            h_tminus1, W_h, x_t, U_x,
                                                            add1, a_t, h_t, V_h, o_t, y_t,
                                                            hsize, vsize, T, threadsPerBlock,
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
                              float* b_host, float* c_host, int vsize, int hsize, int T) {
    float* device_x;
    cudaMalloc((void **)&device_x, sizeof(float) * vsize * T);
    cudaMemcpy(device_x, x, vsize * T * sizeof(float), cudaMemcpyHostToDevice);

    /* 
      shapes:
      inputs/intermediate/outputs:
        x: vsize    by        1
        a: hsize    by        1
        h: hsize    by        1
        o: vsize    by        1
        y: vsize    by        1

      parameters:
        b: hsize    by        1
        c: vsize    by        1
        U: hsize    by        vsize
        V: vsize    by        hsize
        W: hsize    by        hsize

    */

    // alloc result (and intermediate result) destinations
    float* device_a;
    // float* device_h;
    float* h = (float*) calloc(hsize * T, sizeof(float));
    host_initOnes(h, hsize, 1);
    float* device_o;
    float* device_y;
    cudaMalloc((void**) &device_a, sizeof(float) * hsize * T);
    // cudaMalloc((void**) &device_h, sizeof(float) * hsize * T);
    cudaMalloc((void**) &device_o, sizeof(float) * vsize * T);
    cudaMalloc((void**) &device_y, sizeof(float) * vsize * T);

    // set all hidden layers' values (across all timesteps) to -1 to enforce waiting
    float* device_h;
    cudaMalloc((void **) &device_h, sizeof(float) * hsize * T);
    cudaMemcpy(device_h, h, sizeof(float) * hsize * T, cudaMemcpyHostToDevice);
    setToValAfter(device_h, hsize * T, -1.f, hsize);

    // declare, alloc, and copy onto device the init paramters
    float* b;
    cudaMalloc((void**) &b, hsize * sizeof(float));
    cudaMemcpy(b, b_host, hsize * sizeof(float), cudaMemcpyHostToDevice);
    float* c;
    cudaMalloc((void**) &c, vsize * sizeof(float));
    cudaMemcpy(c, c_host, vsize * sizeof(float), cudaMemcpyHostToDevice);
    float* U;
    cudaMalloc((void**) &U, hsize * vsize * sizeof(float));
    cudaMemcpy(U, U_host, hsize * vsize * sizeof(float), cudaMemcpyHostToDevice);
    float* W;
    cudaMalloc((void**) &W, hsize * hsize * sizeof(float));
    cudaMemcpy(W, W_host, hsize * hsize * sizeof(float), cudaMemcpyHostToDevice);
    float* V;
    cudaMalloc((void**) &V, vsize * hsize * sizeof(float));
    cudaMemcpy(V, V_host, vsize * hsize * sizeof(float), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();
    forwardPass(device_x, device_a, device_h, device_o, device_y,
                U, V, W, b, c,
                hsize, vsize, T);
    // cudaThreadSynchronize();
    cudaCheckError( cudaDeviceSynchronize() );
    double endTime = CycleTimer::currentSeconds();
    cudaMemcpy(y, device_y, vsize * T * sizeof(float), cudaMemcpyDeviceToHost);
    double overallDuration = endTime - startTime;
    
    return overallDuration;
}

double cudaSequentialForwardPassTimer (float* x, float* y, float* U_host, float* V_host, float* W_host,
                              float* b_host, float* c_host, int vsize, int hsize, int T) {
    float* device_x;
    cudaMalloc((void **)&device_x, sizeof(float) * vsize * T);
    cudaMemcpy(device_x, x, vsize * T * sizeof(float), cudaMemcpyHostToDevice);

    /* 
      shapes:
      inputs/intermediate/outputs:
        x: vsize    by        1
        a: hsize    by        1
        h: hsize    by        1
        o: vsize    by        1
        y: vsize    by        1

      parameters:
        b: hsize    by        1
        c: vsize    by        1
        U: hsize    by        vsize
        V: vsize    by        hsize
        W: hsize    by        hsize

    */

    // alloc result (and intermediate result) destinations
    float* device_a;
    float* device_h;
    float* device_o;
    float* device_y;
    cudaMalloc((void**) &device_a, sizeof(float) * hsize * T);
    cudaMalloc((void**) &device_h, sizeof(float) * hsize * T);
    cudaMalloc((void**) &device_o, sizeof(float) * vsize * T);
    cudaMalloc((void**) &device_y, sizeof(float) * vsize * T);

    // set all hidden layers' values (across all timesteps) to -1 to enforce waiting
    setToValAfter(device_h, hsize * T, -1.f, 0); // CHANGE!!!!

    // declare, alloc, and copy onto device the init paramters
    float* b;
    cudaMalloc((void**) &b, hsize * sizeof(float));
    cudaMemcpy(b, b_host, hsize * sizeof(float), cudaMemcpyHostToDevice);
    float* c;
    cudaMalloc((void**) &c, vsize * sizeof(float));
    cudaMemcpy(c, c_host, vsize * sizeof(float), cudaMemcpyHostToDevice);
    float* U;
    cudaMalloc((void**) &U, hsize * vsize * sizeof(float));
    cudaMemcpy(U, U_host, hsize * vsize * sizeof(float), cudaMemcpyHostToDevice);
    float* W;
    cudaMalloc((void**) &W, hsize * hsize * sizeof(float));
    cudaMemcpy(W, W_host, hsize * hsize * sizeof(float), cudaMemcpyHostToDevice);
    float* V;
    cudaMalloc((void**) &V, vsize * hsize * sizeof(float));
    cudaMemcpy(V, V_host, vsize * hsize * sizeof(float), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();
    forwardPassSequential(device_x, device_a, device_h, device_o, device_y,
                U, V, W, b, c,
                vsize, hsize, T);
    cudaThreadSynchronize();
    double endTime = CycleTimer::currentSeconds();
    cudaMemcpy(y, device_y, vsize * T * sizeof(float), cudaMemcpyDeviceToHost);
    double overallDuration = endTime - startTime;
    
    return overallDuration;
}