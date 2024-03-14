#include "ModelKernels.h"
#include "NeuronModelData.cuh"
#include "../../../shared/calculations/NeuronModelCalculations.h"

namespace gpu {
__global__ void aeif_model_kernel(RelearnGPUTypes::step_type step, gpu::NeuronModelData* gpu_data_handle, models::AEIFModelData* model_data_handle, RelearnGPUTypes::number_neurons_type number_neurons, double C, double g_L, double E_L, double V_T, double d_T, double tau_w, double a, double b, double V_spike, double d_T_inverse, double tau_w_inverse, double C_inverse, double scale, unsigned int h);

__global__ void fitzhughnagumo_model_kernel(RelearnGPUTypes::step_type step, gpu::NeuronModelData* gpu_data_handle, models::FitzHughNagumoModelData* model_data_handle, RelearnGPUTypes::number_neurons_type number_neurons, double a, double b, double phi, double init_w, double init_x, double scale, unsigned int h);

__global__ void izhikevich_model_kernel(RelearnGPUTypes::step_type step, gpu::NeuronModelData* gpu_data_handle, models::IzhikevichModelData* model_data_handle, double V_spike, double a, double b, double c, double d, double k1, double k2, double k3, double scale, unsigned int h);

__global__ void poisson_model_kernel(RelearnGPUTypes::step_type step, gpu::NeuronModelData* gpu_data_handle, models::PoissonModelData* model_data_handle, double x_0, double tau_x, unsigned int refractory_period);

void init_aeif_kernel(RelearnGPUTypes::step_type step, gpu::NeuronModelDataHandle* gpu_data_handle, RelearnGPUTypes::number_neurons_type number_neurons, gpu::models::ModelDataHandle* model_data_handle, double C, double g_L, double E_L, double V_T, double d_T, double tau_w, double a, double b, double V_spike, double d_T_inverse, double tau_w_inverse, double C_inverse, double scale, unsigned int h) {
    const auto num_threads = get_number_threads(aeif_model_kernel, number_neurons);
    const auto num_blocks = get_number_blocks(num_threads, number_neurons);

    aeif_model_kernel<<<num_threads, num_blocks>>>(step, (NeuronModelData*)((NeuronModelDataHandleImpl*)gpu_data_handle)->get_device_ptr(), ((models::AEIFModelDataHandleImpl*)model_data_handle)->device_ptr, number_neurons, C, g_L, E_L, V_T, d_T, tau_w, a, b, V_spike, d_T_inverse, tau_w_inverse, C_inverse, scale, h);

    cudaDeviceSynchronize();
    gpu_check_last_error();
}

void init_fitzhughnagumo_kernel(RelearnGPUTypes::step_type step, gpu::NeuronModelDataHandle* gpu_data_handle, RelearnGPUTypes::number_neurons_type number_neurons, gpu::models::ModelDataHandle* model_data_handle, double a, double b, double phi, double init_w, double init_x, double scale, unsigned int h) {
    const auto num_threads = get_number_threads(fitzhughnagumo_model_kernel, number_neurons);
    const auto num_blocks = get_number_blocks(num_threads, number_neurons);

    fitzhughnagumo_model_kernel<<<num_threads, num_blocks>>>(step, (NeuronModelData*)((NeuronModelDataHandleImpl*)gpu_data_handle)->get_device_ptr(), ((models::FitzHughNagumoModelDataHandleImpl*)model_data_handle)->device_ptr, number_neurons, a, b, phi, init_w, init_x, scale, h);

    cudaDeviceSynchronize();
    gpu_check_last_error();
}

void init_izhikevich_kernel(RelearnGPUTypes::step_type step, gpu::NeuronModelDataHandle* gpu_data_handle, RelearnGPUTypes::number_neurons_type number_neurons, gpu::models::ModelDataHandle* model_data_handle, double V_spike, double a, double b, double c, double d, double k1, double k2, double k3, double scale, unsigned int h) {
    const auto num_threads = get_number_threads(izhikevich_model_kernel, number_neurons);
    const auto num_blocks = get_number_blocks(num_threads, number_neurons);

    izhikevich_model_kernel<<<num_threads, num_blocks>>>(step, (NeuronModelData*)((NeuronModelDataHandleImpl*)gpu_data_handle)->get_device_ptr(), ((models::IzhikevichModelDataHandleImpl*)model_data_handle)->device_ptr, V_spike, a, b, c, d, k1, k2, k3, scale, h);

    cudaDeviceSynchronize();
    gpu_check_last_error();
}

void init_poisson_kernel(RelearnGPUTypes::step_type step, gpu::NeuronModelDataHandle* gpu_data_handle, RelearnGPUTypes::number_neurons_type number_neurons, gpu::models::ModelDataHandle* model_data_handle, double x_0, double tau_x, unsigned int refractory_period) {
    const auto num_threads = get_number_threads(poisson_model_kernel, number_neurons);
    const auto num_blocks = get_number_blocks(num_threads, number_neurons);
    poisson_model_kernel<<<num_threads, num_blocks>>>(step, (NeuronModelData*)((NeuronModelDataHandleImpl*)gpu_data_handle)->get_device_ptr(), ((models::PoissonModelDataHandleImpl*)model_data_handle)->device_ptr, x_0, tau_x, refractory_period);

    cudaDeviceSynchronize();
    gpu_check_last_error();
}

__global__ void aeif_model_kernel(RelearnGPUTypes::step_type step, gpu::NeuronModelData* gpu_data_handle, models::AEIFModelData* model_data_handle, RelearnGPUTypes::number_neurons_type number_neurons, double C, double g_L, double E_L, double V_T, double d_T, double tau_w, double a, double b, double V_spike, double d_T_inverse, double tau_w_inverse, double C_inverse, double scale, unsigned int h) {
    const auto neuron_id = block_thread_to_neuron_id(blockIdx.x, threadIdx.x, blockDim.x);

    if (neuron_id >= gpu_data_handle->extra_infos->get_number_local_neurons()) {
        return;
    }

    if (gpu_data_handle->extra_infos->disable_flags[neuron_id] == UpdateStatus::Disabled) {
        return;
    }
    const auto synaptic_input = gpu_data_handle->get_synaptic_input(step, neuron_id);
    const auto background_activity = gpu_data_handle->get_background_activity(step, neuron_id);
    const auto stimulus = gpu_data_handle->get_stimulus(step, neuron_id);

    const auto _x = gpu_data_handle->get_x(neuron_id);

    const auto _w = model_data_handle->w[neuron_id];

    const auto& [x_val, this_fired, w_val] = Calculations::aeif(_x, synaptic_input, background_activity, stimulus, _w, h, scale, V_spike, g_L, E_L, V_T, d_T, d_T_inverse, a, b, C_inverse, tau_w_inverse);

    model_data_handle->w[neuron_id] = w_val;
    gpu_data_handle->set_x(neuron_id, x_val);
    gpu_data_handle->set_fired(neuron_id, this_fired);
}

__global__ void fitzhughnagumo_model_kernel(RelearnGPUTypes::step_type step, gpu::NeuronModelData* gpu_data_handle, models::FitzHughNagumoModelData* model_data_handle, RelearnGPUTypes::number_neurons_type number_neurons, double a, double b, double phi, double init_w, double init_x, double scale, unsigned int h) {
    const auto neuron_id = block_thread_to_neuron_id(blockIdx.x, threadIdx.x, blockDim.x);

    if (neuron_id >= gpu_data_handle->extra_infos->get_number_local_neurons()) {
        return;
    }

    if (gpu_data_handle->extra_infos->disable_flags[neuron_id] == UpdateStatus::Disabled) {
        return;
    }
    const auto synaptic_input = gpu_data_handle->get_synaptic_input(step, neuron_id);
    const auto background_activity = gpu_data_handle->get_background_activity(step, neuron_id);
    const auto stimulus = gpu_data_handle->get_stimulus(step, neuron_id);

    const auto _x = gpu_data_handle->get_x(neuron_id);

    const auto _w = model_data_handle->w[neuron_id];

    const auto& [x_val, this_fired, w_val] = Calculations::fitz_hugh_nagumo(_x, synaptic_input, background_activity, stimulus, _w, h, scale, phi, a, b);

    model_data_handle->w[neuron_id] = w_val;
    gpu_data_handle->set_x(neuron_id, x_val);
    gpu_data_handle->set_fired(neuron_id, this_fired);
}

__global__ void izhikevich_model_kernel(RelearnGPUTypes::step_type step, gpu::NeuronModelData* gpu_data_handle, models::IzhikevichModelData* model_data_handle, double V_spike, double a, double b, double c, double d, double k1, double k2, double k3, double scale, unsigned int h) {
    const auto neuron_id = block_thread_to_neuron_id(blockIdx.x, threadIdx.x, blockDim.x);

    if (neuron_id >= gpu_data_handle->extra_infos->get_number_local_neurons()) {
        return;
    }

    if (gpu_data_handle->extra_infos->disable_flags[neuron_id] == UpdateStatus::Disabled) {
        return;
    }
    const auto synaptic_input = gpu_data_handle->get_synaptic_input(step, neuron_id);
    const auto background_activity = gpu_data_handle->get_background_activity(step, neuron_id);
    const auto stimulus = gpu_data_handle->get_stimulus(step, neuron_id);
    // bg__[neuron_id] = background_activity;

    const auto _x = gpu_data_handle->get_x(neuron_id);

    const auto _u = model_data_handle->u[neuron_id];

    const auto& [x_val, this_fired, u_val] = Calculations::izhikevich(_x, synaptic_input, background_activity, stimulus, _u, h, scale, V_spike, a, b, c, d, k1, k2, k3);

    model_data_handle->u[neuron_id] = u_val;
    gpu_data_handle->set_x(neuron_id, x_val);
    gpu_data_handle->set_fired(neuron_id, this_fired);
}

__global__ void poisson_model_kernel(RelearnGPUTypes::step_type step, gpu::NeuronModelData* gpu_data_handle, models::PoissonModelData* model_data_handle, double x_0, double tau_x, unsigned int refractory_period) {
    const auto refractory_time = &model_data_handle->refractory_time;

    const auto tau_x_inverse = 1.0 / tau_x;

    const auto neuron_id = block_thread_to_neuron_id(blockIdx.x, threadIdx.x, blockDim.x);

    if (neuron_id >= gpu_data_handle->extra_infos->get_number_local_neurons()) {
        return;
    }

    // Init
    auto curand_state = gpu::RandomHolder::init(step, gpu_data_handle->extra_infos->get_number_local_neurons(), gpu::RandomHolder::POISSON, neuron_id);
    const auto random_value = gpu::RandomHolder::get_percentage(&curand_state);

    if (gpu_data_handle->extra_infos->disable_flags[neuron_id] == UpdateStatus::Disabled) {
        return;
    }
    const auto synaptic_input = gpu_data_handle->get_synaptic_input(step, neuron_id);
    const auto background_activity = gpu_data_handle->get_background_activity(step, neuron_id);
    const auto stimulus = gpu_data_handle->get_stimulus(step, neuron_id);

    const auto x_ = gpu_data_handle->x[neuron_id];

    const auto& [x_val, this_fired, this_refractory_time] = Calculations::poisson(x_, synaptic_input, background_activity, stimulus, (*refractory_time)[neuron_id], random_value, x_0, refractory_period, gpu_data_handle->h, gpu_data_handle->scale, tau_x_inverse);

    (*refractory_time)[neuron_id] = this_refractory_time;
    gpu_data_handle->set_x(neuron_id, x_val);
    gpu_data_handle->set_fired(neuron_id, this_fired);
}
}
