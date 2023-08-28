#if !CUDA_FOUND

#define CUDA_DEFINITION {}
#define CUDA_PTR_DEFINITION {return 0;}
#include "gpu/Interface.h"

#endif