#pragma once

#include "../../utils/GpuTypes.h"
#include "NeuronModelDataHandle.h"

#if CUDA_FOUND
#define IF_CUDA_NOT
#else
#define IF_CUDA_NOT \
    { }
#endif

namespace gpu {
/**
 * @brief Calls the kernel for the AEIF neuron model.
 */
void init_aeif_kernel(RelearnGPUTypes::step_type step, gpu::NeuronModelDataHandle* gpu_data_handle, RelearnGPUTypes::number_neurons_type number_neurons, gpu::models::ModelDataHandle* model_data_handle, double C, double g_L, double E_L, double V_T, double d_T, double tau_w, double a, double b, double V_spike, double d_T_inverse, double tau_w_inverse, double C_inverse, double scale, unsigned int h) IF_CUDA_NOT;

/**
 * @brief Calls the kernel for the FitzHughNagumo neuron model.
 */
void init_fitzhughnagumo_kernel(RelearnGPUTypes::step_type step, gpu::NeuronModelDataHandle* gpu_data_handle, RelearnGPUTypes::number_neurons_type number_neurons, gpu::models::ModelDataHandle* model_data_handle, double a, double b, double phi, double init_w, double init_x, double scale, unsigned int h) IF_CUDA_NOT;

/**
 * @brief Calls the kernel for the Izhikevich neuron model.
 */
void init_izhikevich_kernel(RelearnGPUTypes::step_type step, gpu::NeuronModelDataHandle* gpu_data_handle, RelearnGPUTypes::number_neurons_type number_neurons, gpu::models::ModelDataHandle* model_data_handle, double V_spike, double a, double b, double c, double d, double k1, double k2, double k3, double scale, unsigned int h) IF_CUDA_NOT;

/**
 * @brief Calls the kernel for the Poisson neuron model.
 */
void init_poisson_kernel(RelearnGPUTypes::step_type step, gpu::NeuronModelDataHandle* gpu_data_handle, RelearnGPUTypes::number_neurons_type number_neurons, gpu::models::ModelDataHandle* model_data_handle, double x_0, double tau_x, unsigned int refractory_period) IF_CUDA_NOT;
}
