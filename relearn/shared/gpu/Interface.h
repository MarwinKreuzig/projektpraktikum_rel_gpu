#pragma once

#include "gpu/Macros.h"
#include "gpu/GpuTypes.h"
#include "enums/FiredStatus.h"

#include <span>


#ifndef CUDA_DEFINITION
#define CUDA_DEFINITION ;
#define CUDA_PTR_DEFINITION CUDA_DEFINITION
#endif



namespace gpu::models::izhekevich {

void construct_gpu(const unsigned int h, double V_spike, double a, double b, double c, double d, double k1, double k2, double k3) CUDA_DEFINITION

void init_gpu(RelearnTypes::number_neurons_type number_neurons) CUDA_DEFINITION

void init_neurons_gpu(const RelearnTypes::number_neurons_type start_id, const RelearnTypes::number_neurons_type end_id)  CUDA_DEFINITION

void update_activity_gpu(const size_t step,  const double* stimulus, const double* background, const double* syn_input, size_t num_neurons)  CUDA_DEFINITION

void create_neurons_gpu(const RelearnTypes::number_neurons_type creation_count)  CUDA_DEFINITION
};

namespace gpu::models::poisson {

void construct_gpu( const unsigned int h, const double x_0,
    const double tau_x,
    const unsigned int refractory_period) CUDA_DEFINITION

void init_gpu(const RelearnTypes::number_neurons_type number_neurons)  CUDA_DEFINITION

void init_neurons_gpu(const RelearnTypes::number_neurons_type start_id, const RelearnTypes::number_neurons_type end_id) CUDA_DEFINITION

void update_activity_gpu(const size_t step, const double* stimulus, const double* background, const double* syn_input, size_t num_neurons) CUDA_DEFINITION

void create_neurons_gpu(const RelearnTypes::number_neurons_type creation_count) CUDA_DEFINITION
};

namespace gpu::models::aeif {

void construct_gpu( const unsigned int _h, double _C, double _g_L, double _E_L, double _V_T, double _d_T, double _tau_w, double _a, double _b, double _V_spike) CUDA_DEFINITION

void init_gpu(const RelearnTypes::number_neurons_type number_neurons)  CUDA_DEFINITION

void init_neurons_gpu(const RelearnTypes::number_neurons_type start_id, const RelearnTypes::number_neurons_type end_id) CUDA_DEFINITION

void update_activity_gpu(const size_t step, const double* stimulus, const double* background, const double* syn_input, size_t num_neurons) CUDA_DEFINITION

void create_neurons_gpu(const RelearnTypes::number_neurons_type creation_count) CUDA_DEFINITION
};

namespace gpu::models::fitz_hugh_nagumo {

void construct_gpu(const unsigned int _h, double _a, double _b, double _phi, double _init_w, double _init_x) CUDA_DEFINITION

void init_gpu(const RelearnTypes::number_neurons_type number_neurons)  CUDA_DEFINITION

void init_neurons_gpu(const RelearnTypes::number_neurons_type start_id, const RelearnTypes::number_neurons_type end_id) CUDA_DEFINITION

void update_activity_gpu(const size_t step, const double* stimulus, const double* background, const double* syn_input, size_t num_neurons) CUDA_DEFINITION

void create_neurons_gpu(const RelearnTypes::number_neurons_type creation_count) CUDA_DEFINITION
};


namespace gpu::models::NeuronModel {
    FiredStatus* get_fired() CUDA_PTR_DEFINITION
    void disable_neurons(const size_t* neuron_ids, size_t num_disabled_neurons) CUDA_DEFINITION
    void enable_neurons(const size_t* neuron_ids, size_t num_disabled_neurons) CUDA_DEFINITION
};

namespace gpu::neurons::NeuronsExtraInfos {
    void disable_neurons(const size_t* neuron_ids, size_t num_disabled_neurons) CUDA_DEFINITION
    void enable_neurons(const size_t* neuron_ids, size_t num_disabled_neurons) CUDA_DEFINITION
    void init(const RelearnTypes::number_neurons_type num_neurons) CUDA_DEFINITION
};