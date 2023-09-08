#pragma once


#define RELEARN_CUDA_FOUND CUDA_FOUND

#ifdef __CUDACC__
    // File is compiled with Cuda compiler
    #define CUDA_COMPILER
#endif

#ifndef __CUDACC__
    //File is compiled wiith host compiler e.g. g++
    #define HOST_COMPILER
#endif

//Macro that enable execution of a function on gpu and devivr
#if __CUDACC__
    #define GPU_AND_HOST __device__ __host__
#else
    #define GPU_AND_HOST
#endif