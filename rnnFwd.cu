__global__
void kernelComputeForward () {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
}

__global__
void forwardPass (float* device_x, int length, float* device_y) {
    const int threadsPerBlock = 512;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    kernelComputeForward<<< blocks, threadsPerBlock >>>(device_x);
}

double cudaForwardPassTimer (float* x, int* length, float* y) {
    float* device_x;
    float* device_y;
    

}

int main() {
    int N = 30;
    int* x = malloc(sizeof(int) * N);

    cudaForwardPassTimer ();


    return 0;
}