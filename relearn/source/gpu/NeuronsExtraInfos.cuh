#pragma once

#include "gpu/Commons.cuh"
#include "gpu/GpuTypes.h"
#include "gpu/CudaVector.cuh"

#include <cuda.h>

namespace gpu::neurons::NeuronsExtraInfos {

 size_t number_local_neurons_host;
 __device__ __constant__ size_t number_local_neurons_device;

     __device__ gpu::Vector::CudaArray<char> disable_flags;
     gpu::Vector::CudaArrayDeviceHandle<char> handle_disable_flags(disable_flags);

 void init(const RelearnTypes::number_neurons_type num_neurons) {
    handle_disable_flags.resize(num_neurons, 1);

    cuda_copy_to_device(number_local_neurons_device, num_neurons);

    number_local_neurons_host = num_neurons;
}
};