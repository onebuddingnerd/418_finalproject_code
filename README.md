# Title

Source code for "Parallelization of Inference on Recurrent Neural Networks"

# Website

https://lmishr.github.io/15418_project.html

# File Structure and Usage

## The CUDA Code

- It's within the ./cuda_code directory
- There's a building script `build.sh` that will build the binary simply by running `sh build.sh` (on unix machines). Note the binary will not work unless it is run on a machine with a GPU that supports the CUDA API
- Problem parameters need to be changed at *compile-time* by modifying the TIMESTEPS, HSIZE, and VSIZE constants in the source code (this only entails changing the corresponding `#define` statements at the top of the `rnnFwd.cu` and `main.cpp` files)

## The OpenMP Code

- It's within the ./openmp_code directory
- There's a building script `build.sh` that will build the binary simply by running `sh build.sh` (on unix machines). No GPU is required.
- Problem paramters again must be changed in the same fashion as with the CUDA code (but, for this folder's code, only one file, `openmpFwd.cpp` has to be changed)
- The binary that is created takes one command line argument, which is the number of processors, i.e. its usage is given by:
`./openmpFwd <num_processors>`