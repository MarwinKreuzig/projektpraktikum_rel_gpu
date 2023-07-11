#pragma once

#include "gpu/GpuTypes.h"

namespace gpu::models::izhekevich {

void init_gpu(RelearnTypes::number_neurons_type number_neurons);

void init_neurons_gpu(const RelearnTypes::number_neurons_type start_id, const RelearnTypes::number_neurons_type end_id) ;

void update_activity_gpu() ;


void create_neurons_gpu(const RelearnTypes::number_neurons_type creation_count) ;
};

namespace gpu::models::poisson {


void init_gpu(RelearnTypes::number_neurons_type number_neurons) ;

void init_neurons_gpu(const RelearnTypes::number_neurons_type start_id, const RelearnTypes::number_neurons_type end_id);

void update_activity_gpu();


void create_neurons_gpu(const RelearnTypes::number_neurons_type creation_count);
};