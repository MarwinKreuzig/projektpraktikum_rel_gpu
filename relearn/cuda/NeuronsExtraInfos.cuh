#pragma once

#include "Commons.cuh"
#include "gpu/GpuTypes.h"
#include "CudaVector.cuh"

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

void create_neurons(size_t creation_count) {
    const auto old_size = number_local_neurons_host;
    const auto new_size = old_size + creation_count;
    cuda_copy_to_device(number_local_neurons_device, new_size);
    number_local_neurons_host = new_size;
}

void disable_neurons(const size_t* neuron_ids, size_t num_disabled_neurons)  {
    if(num_disabled_neurons == 0) {
        return;
    }
    cuda_set_for_indices<char>(handle_disable_flags.data(), neuron_ids, num_disabled_neurons, 0);
}

void enable_neurons(const size_t* neuron_ids, size_t num_enabled_neurons)  {
    if (num_enabled_neurons == 0) {
        return;
    }
    cuda_set_for_indices<char>(handle_disable_flags.data(), neuron_ids, num_enabled_neurons, 1);
}

};