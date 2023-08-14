#pragma once

#include "gpu/Commons.cuh"
#include "gpu/GpuTypes.h"

#include <cuda.h>

namespace gpu::neurons::NeuronsExtraInfos {

 size_t number_local_neurons_host;

 __device__ __constant__ size_t number_local_neurons_device;

     __device__ char* disable_flags;

 void init(const RelearnTypes::number_neurons_type num_neurons) {
    cuda_calloc_symbol(disable_flags, sizeof(char) * num_neurons, 1);

    cuda_copy_to_device(number_local_neurons_device, num_neurons);
    gpu_check_last_error();

    number_local_neurons_host = num_neurons;
}
};