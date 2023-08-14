#pragma once

#include "gpu/GpuTypes.h"
#include "neurons/enums/FiredStatus.h"

namespace gpu::models::izhekevich {

void construct_gpu(const unsigned int h);

void init_gpu(RelearnTypes::number_neurons_type number_neurons);

void init_neurons_gpu(const RelearnTypes::number_neurons_type start_id, const RelearnTypes::number_neurons_type end_id) ;

void update_activity_gpu(const size_t step) ;


void create_neurons_gpu(const RelearnTypes::number_neurons_type creation_count) ;
};

namespace gpu::models::poisson {

void construct_gpu( const unsigned int h, const double x_0,
    const double tau_x,
    const unsigned int refractory_period);

void init_gpu(const RelearnTypes::number_neurons_type number_neurons) ;

void init_neurons_gpu(const RelearnTypes::number_neurons_type start_id, const RelearnTypes::number_neurons_type end_id);

void update_activity_gpu(const size_t step);


void create_neurons_gpu(const RelearnTypes::number_neurons_type creation_count);
};

namespace gpu::models::NeuronModel {
    FiredStatus* get_fired();
}

namespace gpu::neurons::NeuronsExtraInfos {
    void init(const RelearnTypes::number_neurons_type num_neurons);
};