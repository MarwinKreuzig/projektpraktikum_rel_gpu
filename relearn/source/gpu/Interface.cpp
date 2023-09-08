#if !CUDA_FOUND

// Macros the replace the methods of our interface to cuda with empty methods when compiled without cuda
#define CUDA_DEFINITION \
    { }
#define CUDA_PTR_DEFINITION \
    { return 0; }
#include "gpu/Interface.h"

#endif