#pragma once

#include "gpu/GpuTypes.h"
#include "neurons/enums/FiredStatus.h"

#include <span>

namespace gpu::models::izhekevich {

void construct_gpu(const unsigned int h, double V_spike, double a, double b, double c, double d, double k1, double k2, double k3);

void init_gpu(RelearnTypes::number_neurons_type number_neurons);

void init_neurons_gpu(const RelearnTypes::number_neurons_type start_id, const RelearnTypes::number_neurons_type end_id) ;

void update_activity_gpu(const size_t step,  const double* stimulus, const double* background, const double* syn_input, size_t num_neurons) ;

void create_neurons_gpu(const RelearnTypes::number_neurons_type creation_count) ;
};

namespace gpu::models::poisson {

void construct_gpu( const unsigned int h, const double x_0,
    const double tau_x,
    const unsigned int refractory_period);

void init_gpu(const RelearnTypes::number_neurons_type number_neurons) ;

void init_neurons_gpu(const RelearnTypes::number_neurons_type start_id, const RelearnTypes::number_neurons_type end_id);

void update_activity_gpu(const size_t step, const double* stimulus, const double* background, const double* syn_input, size_t num_neurons);

void create_neurons_gpu(const RelearnTypes::number_neurons_type creation_count);
};

namespace gpu::models::aeif {

void construct_gpu( const unsigned int _h, double _C, double _g_L, double _E_L, double _V_T, double _d_T, double _tau_w, double _a, double _b, double _V_spike);

void init_gpu(const RelearnTypes::number_neurons_type number_neurons) ;

void init_neurons_gpu(const RelearnTypes::number_neurons_type start_id, const RelearnTypes::number_neurons_type end_id);

void update_activity_gpu(const size_t step, const double* stimulus, const double* background, const double* syn_input, size_t num_neurons);

void create_neurons_gpu(const RelearnTypes::number_neurons_type creation_count);
};

namespace gpu::models::fitz_hugh_nagumo {

void construct_gpu(const unsigned int _h, double _a, double _b, double _phi, double _init_w, double _init_x);

void init_gpu(const RelearnTypes::number_neurons_type number_neurons) ;

void init_neurons_gpu(const RelearnTypes::number_neurons_type start_id, const RelearnTypes::number_neurons_type end_id);

void update_activity_gpu(const size_t step, const double* stimulus, const double* background, const double* syn_input, size_t num_neurons);

void create_neurons_gpu(const RelearnTypes::number_neurons_type creation_count);
};


namespace gpu::models::NeuronModel {
    FiredStatus* get_fired();
}

namespace gpu::neurons::NeuronsExtraInfos {
    void init(const RelearnTypes::number_neurons_type num_neurons);
};