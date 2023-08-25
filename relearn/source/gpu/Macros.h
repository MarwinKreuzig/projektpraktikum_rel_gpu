#pragma once

#define cuda_available true

#ifdef __CUDACC__
    #define CUDA_COMPILER
#endif

#ifndef __CUDACC__
    #define HOST_COMPILER
#endif

#if __CUDACC__
    #define GPU_AND_HOST __device__
#else
    #define GPU_AND_HOST
#endif