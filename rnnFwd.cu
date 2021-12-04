
#include <cmath>
#include <math.h>

// A is a by b
// B is b by c
// AB is a by c
void matmul (float* A, float* B, float* AB, int a, int b, int c) {

    for (int i = 0; i < a; i++) {
        for (int k = 0; k < c; k++) {
            int ab_result_ik = 0;
            for (int j = 0; j < b; j++){
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
        A[timestep * dimsize + i] = src[i];
    }
}

void vectorAdd (float* A, float* B, float* AplusB, int a, int b) {
    for (int i = 0; i < a; i++) {
        for (int j = 0; j < b; j++) {
            AplusB[i*a + j] = A[i*a + j] + B[i*a + j];
        }
    }
}

void vectorTanh (float* A, float* ret, int a, int b) {
    for (int i = 0; i < a; i++) {
        for (int j = 0; j < b; j++) {
            ret[i] = std::tanh(A[i*a + j])
        }
    }
}

void vectorSoftmax(float* src, float* dest, int a, int b) {
    float denom = 0;
    
    //PROBLEM if b is not 1 here
    
    for (int = 0; i < a; i++) {
        denom += exp(src[i]);
    }

    for (int i = 0; i < a; i++) {
        dest[i] = src[i] / denom;
    }

    return;
}

__global__
void kernelComputeForward (float* device_x, float* device_a, float* device_h, float* device_o, 
                            float* device_y, int hsize, int vsize, float* W, float* U, float* V,
                            float* c) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; // time index

    if (index > 0) {
        while (device_h[index * (hsize - 1)] == -1) ;
    } // spin til prev timestep's h-level vector lacks nondefault value (which indicates that the computation is pending)
    
    
    // a[t] = b + W * h[t-1] + U * x[t]
    // h[t] = tanh(a[t])
    // o[t] = c + V * h[t]
    // y[t] = softmax(o[t])
    
    // W_h_term =  device_W ** device_h[index * hsize : (index+1) * hsize]
    // U_x_term = device_U ** device_x[index * vsize : (index+1) * vsize]
    // device_a[index * hsize : (index+1) * hsize] = device_b + 
    //                  W_h_term + U_x_term

    // 1a. the W_h term
    float* h_tminus1 = calloc(hsize, sizeof(float));
    getExcerpt(device_h, h_tminus1, index-1, hsize);    
    float* W_h = calloc(hsize, sizeof(float));
    matmul(W, h_tminus1, W_h, hsize, hsize, 1);
    // 1b. the ux term
    float* x_t = calloc(vsize, sizeof(float));
    getExcerpt(device_x, x_t, index, vsize);
    float* U_x = calloc(hsize, sizeof(float));
    matmul(U, x_t, U_x, hsize, vsize, 1);
    // 1: addition of b
    float* add1 = calloc(hsize, sizeof(float));
    vectorAdd(U_x, b, add1, hsize, 1);
    float* a_t = calloc(vsize, sizeof(float));
    vectorAdd(add1, W_h, a_t, hsize, 1);

    free(h_tminus1);
    free(W_h);
    free(U_x);
    free(add1);
    setExcerpt(device_a, a_t, index, hsize);
    // free(a_t) LATER

    // a_t has the result for the next layer (h)
    // 2: tanh of vector for a_t
    float* h_t = calloc(hsize, sizeof(float));
    vectorTanh(a_t, h_t, hsize, 1);
    setExcerpt(device_h, h_t, index, hsize);
    // free(h_t) LATER

    // h_t has the result for the next layer(o)
    // 3: addition of V*h and c
    float* V_h = calloc(vsize, sizeof(float));
    matmul(V, h_t, V_h, vsize, hsize, 1);
    float* o_t = calloc(hsize, sizeof(float));
    vectorAdd(c, V_h, o_t, hsize, 1);
    setExcerpt(device_o, o_t, index, hsize);
    // free(o_t) LATER

    // o_t has the result for the next layer(o)
    // 4: softmax(o_t)
    float* y_t = calloc(vsize, sizeof(float));
    vectorSoftmax(o_t, y_t, vsize, 1);
    setExcerpt(device_y, y_y, index, vsize);
    
    free(a_t);
    free(h_t);
    free(o_t);
    free(y_t);
    return;

}


void forwardPass (float* device_x, float* device_a, float* device_h, float* device_o, 
                  float* device_y, int vsize, int hsize, int T) {
    const int threadsPerBlock = 512;
    const int blocks = (T + threadsPerBlock - 1) / threadsPerBlock;

    kernelComputeForward<<< blocks, threadsPerBlock >>>(device_x, device_a, device_h, 
                                                        device_o, device_y);
}

double cudaForwardPassTimer (float* x, float* y, float* W, float* V, float* W,
                              int vsize, int hsize, int T) {
    float* device_x;
    float* device_y;
    
    cudaMalloc((void **)&device_x, sizeof(int) * length);
    cudaMemcpy(device_x, x, length, cudaMemcpyHostToDevice);

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

    cudaMalloc((void**) &device_x, sizeof(int) * vsize * T);
    cudaMalloc((void**) &device_a, sizeof(int) * hsize * T);
    cudaMalloc((void**) &device_h, sizeof(int) * hsize * T);
    cudaMalloc((void**) &device_o, sizeof(int) * vsize * T);
    cudaMalloc((void**) &device_y, sizeof(int) * vsize * T);

    double startTime = CycleTimer::currentSeconds();
    forwardPass(device_x, devica_a, device_h, device_o, device_y, 
                          vsize, hsize, T);
    cudaThreadSynchronize();
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;
    
    return overallDuration;
}