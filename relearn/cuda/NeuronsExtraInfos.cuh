#pragma once

#include "Commons.cuh"
#include "enums/UpdateStatus.h"
#include "gpu/GpuTypes.h"
#include "CudaVector.cuh"

#include <cuda.h>

namespace gpu::neurons::NeuronsExtraInfos {

    class NeuronsExtraInfos {

        public:
 size_t number_local_neurons_device;

     gpu::Vector::CudaVector<UpdateStatus> disable_flags;

    

 __device__ void init(const RelearnTypes::number_neurons_type num_neurons) {
   disable_flags.resize(num_neurons, UpdateStatus::Enabled);

    number_local_neurons_device = num_neurons;
}

public:

 __device__ NeuronsExtraInfos(size_t num_neurons) : number_local_neurons_device(num_neurons) {
        init(num_neurons);
     }


__device__ void create_neurons(size_t creation_count) {
    const auto old_size = number_local_neurons_device;
    const auto new_size = old_size + creation_count;
    number_local_neurons_device = new_size;
    disable_flags.resize(new_size, UpdateStatus::Enabled);
}

__device__ void disable_neurons(const size_t* neuron_ids, size_t num_disabled_neurons)  {
    if(num_disabled_neurons == 0) {
        return;
    }
    disable_flags.set(neuron_ids, num_disabled_neurons, UpdateStatus::Disabled);
}

__device__ void enable_neurons(const size_t* neuron_ids, size_t num_enabled_neurons)  {
    if (num_enabled_neurons == 0) {
        return;
    }
    disable_flags.set(neuron_ids, num_enabled_neurons, UpdateStatus::Enabled);
}

 inline __device__ size_t get_number_local_neurons() {
    return number_local_neurons_device;
 }

    };

     size_t number_local_neurons_host;

     __device__ NeuronsExtraInfos* extra_infos;


  void init(const RelearnTypes::number_neurons_type num_neurons) {
    void* extra_infos_dev_ptr = init_class_on_device<NeuronsExtraInfos>(num_neurons);
    cuda_copy_to_device(extra_infos, extra_infos_dev_ptr);
  // cuda_generic_kernel<<<1,1>>>([=]__device__(size_t n) {device_init(n);}, num_neurons);

    cudaDeviceSynchronize();
    gpu_check_last_error();
       number_local_neurons_host = num_neurons;
}

void disable_neurons(const size_t* neuron_ids, size_t num_disabled_neurons)  {
    void* dev_ptr = cuda_malloc(sizeof(size_t)*num_disabled_neurons);
    cuda_memcpy_to_device(dev_ptr, (void*)neuron_ids, sizeof(size_t), num_disabled_neurons);

    cuda_generic_kernel<<<1,1>>>([=]__device__(size_t* ids, size_t num) {extra_infos->disable_neurons(ids,num);}, (size_t*)dev_ptr, num_disabled_neurons);

    cudaDeviceSynchronize();
    gpu_check_last_error();

    cudaFree(dev_ptr);
    gpu_check_last_error();
}

void enable_neurons(const size_t* neuron_ids, size_t num_enabled_neurons)  {
    void* dev_ptr = cuda_malloc(sizeof(size_t)*num_enabled_neurons);
    cuda_memcpy_to_device(dev_ptr, (void*) neuron_ids, sizeof(size_t), num_enabled_neurons);

    cuda_generic_kernel<<<1,1>>>([=]__device__(size_t* ids, size_t num) {extra_infos->enable_neurons(ids,num);},(size_t*)dev_ptr, num_enabled_neurons);

    cudaDeviceSynchronize();
    gpu_check_last_error();

    cudaFree(dev_ptr);
    gpu_check_last_error();
}

};