#if !CUDA_FOUND

// Macros the replace the methods of our interface to utils with empty methods when compiled without utils
#define CUDA_DEFINITION \
    { }
#define CUDA_PTR_DEFINITION \
    { return 0; }
#include "gpu/utils/Interface.h"

#endif