#include "cuda_neuron_model.h"

#include "gpu/Commons.cuh"
#include "gpu/RelearnGPUException.h"

#include <thrust/device_vector.h>

namespace gpu::models::izhekevich {

void init_gpu(RelearnTypes::number_neurons_type number_neurons) {
    RelearnGPUException::fail("No gpu support");
}

void init_neurons_gpu(const RelearnTypes::number_neurons_type start_id, const RelearnTypes::number_neurons_type end_id) {
    RelearnGPUException::fail("No gpu support");
}

void update_activity_gpu() {
    RelearnGPUException::fail("No gpu support");
}


void create_neurons_gpu(const RelearnTypes::number_neurons_type creation_count) {
    RelearnGPUException::fail("No gpu support");
}
};

namespace gpu::models::poisson {

thrust::device_vector<double> x{};
thrust::device_vector<double> u{};
__device__ RelearnTypes::number_neurons_type num_neurons{};
//__device__ thrust::device_vector<UpdateStatus> disable_flags{};


void init_gpu(RelearnTypes::number_neurons_type number_neurons) {
   cudaMemcpy(&num_neurons, &number_neurons, sizeof(RelearnTypes::number_neurons_type), cudaMemcpyKind::cudaMemcpyHostToDevice);
   gpu_check_last_error();

   /*x.resize(num_neurons, 0.0);
   u.resize(num_neurons, 0.0);
   //disable_flags.resize(number_neurons, UpdateStatus::Enabled);
   num_neurons = number_neurons;*/
}

void init_neurons_gpu(const RelearnTypes::number_neurons_type start_id, const RelearnTypes::number_neurons_type end_id) {
    RelearnGPUException::fail("No gpu support");
}

void update_activity_gpu() {
    RelearnGPUException::fail("No gpu support");
}


void create_neurons_gpu(const RelearnTypes::number_neurons_type creation_count) {
    RelearnGPUException::fail("No gpu support");
}
};