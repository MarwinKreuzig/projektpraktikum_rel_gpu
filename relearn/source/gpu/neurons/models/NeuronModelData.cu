#include "NeuronModelData.cuh"

#include "../../utils/Interface.h"
#include "../NeuronsExtraInfos.cuh"

namespace gpu {
__device__ double NeuronModelData::get_synaptic_input(RelearnGPUTypes::step_type step, const RelearnGPUTypes::neuron_id_type neuron_id) {
    return syn_input[neuron_id];
}

__device__ double NeuronModelData::get_background_activity(RelearnGPUTypes::step_type step, const RelearnGPUTypes::neuron_id_type neuron_id) {
    return background_calculator->get(step, neuron_id);
}

__device__ double NeuronModelData::get_stimulus(RelearnGPUTypes::step_type step, const RelearnGPUTypes::neuron_id_type neuron_id) {
    return stimulus[neuron_id];
}

__device__ void NeuronModelData::set_x(const RelearnGPUTypes::neuron_id_type neuron_id, double new_value) {
    x[neuron_id] = new_value;
}
__device__ double NeuronModelData::get_x(const RelearnGPUTypes::neuron_id_type neuron_id) {
    return x[neuron_id];
}
__device__ void NeuronModelData::set_fired(const RelearnGPUTypes::neuron_id_type neuron_id, FiredStatus new_value) {
    fired[neuron_id] = new_value;
}
__device__ void NeuronModelData::set_fired(gpu::Vector::CudaArray<FiredStatus> new_values) {
    fired = new_values;
}

NeuronModelDataHandleImpl::~NeuronModelDataHandleImpl() {
    cudaFree(device_ptr);
}

void NeuronModelDataHandleImpl::set_x(const RelearnGPUTypes::neuron_id_type neuron_id, double new_value) {
    x_handle.set(&neuron_id, 1, new_value);
}

void NeuronModelDataHandleImpl::fill_x(RelearnGPUTypes::neuron_id_type start_id, RelearnGPUTypes::neuron_id_type end_id, double new_value) {
    x_handle.fill(start_id, end_id, new_value);
}

void NeuronModelDataHandleImpl::set_fired(const RelearnGPUTypes::neuron_id_type neuron_id, const FiredStatus new_value) {
    fired_handle.set(&neuron_id, 1, new_value);
}

void NeuronModelDataHandleImpl::set_fired(std::vector<FiredStatus>* new_values) {
    if (new_values->size() != fired_handle.get_size()) {
        fired_handle.resize(new_values->size());
    }
    fired_handle.copy_to_device(&new_values->front(), new_values->size());
}

bool NeuronModelDataHandleImpl::get_fired(const RelearnGPUTypes::neuron_id_type neuron_id) {
    FiredStatus* result = (FiredStatus*)malloc(sizeof(FiredStatus));
    cuda_memcpy_to_host(result, fired_handle.data() + neuron_id, sizeof(FiredStatus), 1);
    bool has_fired = *result == FiredStatus::Fired;
    free(result);
    return has_fired;
}

void NeuronModelDataHandleImpl::set_extra_infos(const std::unique_ptr<gpu::neurons::NeuronsExtraInfosHandle>& _extra_infos_handle) {
    extra_infos_handle = (gpu::neurons::NeuronsExtraInfos*)(static_cast<neurons::NeuronsExtraInfosHandleImpl*>(_extra_infos_handle.get())->get_device_pointer());
    cuda_generic_kernel<<<1, 1>>>([=] __device__(NeuronModelData * neuron_model, gpu::neurons::NeuronsExtraInfos * extra_infos) { neuron_model->extra_infos = extra_infos; }, (NeuronModelData*)device_ptr, extra_infos_handle);
    gpu_check_last_error();
}

std::vector<FiredStatus> NeuronModelDataHandleImpl::get_fired() const noexcept {
    std::vector<FiredStatus> fired_data;
    fired_handle.copy_to_host(fired_data);
    return fired_data;
}

NeuronModelData* NeuronModelDataHandleImpl::get_device_ptr() {
    return device_ptr;
}

RelearnGPUTypes::number_neurons_type NeuronModelDataHandleImpl::get_extra_infos_number_local_neurons() {
    auto result = (RelearnGPUTypes::number_neurons_type*)cuda_malloc(sizeof(RelearnGPUTypes::number_neurons_type));

    cuda_generic_kernel<<<1, 1>>>([=] __device__(gpu::neurons::NeuronsExtraInfos * extra_infos_handle, RelearnGPUTypes::number_neurons_type * number_neurons) { *number_neurons = extra_infos_handle->get_number_local_neurons(); }, extra_infos_handle, result);
    cudaDeviceSynchronize();
    gpu_check_last_error();

    RelearnGPUTypes::number_neurons_type host_result;
    cuda_memcpy_to_host(&host_result, result, sizeof(RelearnGPUTypes::number_neurons_type), 1);

    return host_result;
}

namespace models {
    void AEIFModelDataHandleImpl::init(const RelearnGPUTypes::number_neurons_type number_neurons) {
        w_handle.resize(number_neurons, 0);
    }

    void AEIFModelDataHandleImpl::create_neurons(RelearnGPUTypes::number_neurons_type number_neurons) {
        w_handle.resize(number_neurons);
    }

    void AEIFModelDataHandleImpl::init_neurons(NeuronModelDataHandle* gpu_handle, RelearnGPUTypes::number_neurons_type start_id, RelearnGPUTypes::number_neurons_type end_id) {
        gpu_handle->fill_x(start_id, end_id, E_L);
    }

    double AEIFModelDataHandleImpl::get_secondary_variable(const RelearnGPUTypes::neuron_id_type neuron_id) const {
        double* result = (double*)malloc(sizeof(double));
        cuda_memcpy_to_host(result, w_handle.data() + neuron_id, sizeof(double), 1);
        double host_result = *result;
        free(result);
        return host_result;
    }

    void FitzHughNagumoModelDataHandleImpl::init(const RelearnGPUTypes::number_neurons_type number_neurons) {
        w_handle.resize(number_neurons, 0);
    }

    void FitzHughNagumoModelDataHandleImpl::create_neurons(RelearnGPUTypes::number_neurons_type number_neurons) {
        w_handle.resize(number_neurons);
    }

    void FitzHughNagumoModelDataHandleImpl::init_neurons(NeuronModelDataHandle* gpu_handle, RelearnGPUTypes::number_neurons_type start_id, RelearnGPUTypes::number_neurons_type end_id) {
        w_handle.fill(start_id, end_id, init_w);
        gpu_handle->fill_x(start_id, end_id, init_x);
    }

    double FitzHughNagumoModelDataHandleImpl::get_secondary_variable(const RelearnGPUTypes::neuron_id_type neuron_id) const {
        double* result = (double*)malloc(sizeof(double));
        cuda_memcpy_to_host(result, w_handle.data() + neuron_id, sizeof(double), 1);
        double host_result = *result;
        free(result);
        return host_result;
    }

    void IzhikevichModelDataHandleImpl::init(const RelearnGPUTypes::number_neurons_type number_neurons) {
        u_handle.resize(number_neurons, 0);
    }

    void IzhikevichModelDataHandleImpl::create_neurons(RelearnGPUTypes::number_neurons_type number_neurons) {
        u_handle.resize(number_neurons);
    }

    void IzhikevichModelDataHandleImpl::init_neurons(NeuronModelDataHandle* gpu_handle, RelearnGPUTypes::number_neurons_type start_id, RelearnGPUTypes::number_neurons_type end_id) {
        gpu_handle->fill_x(start_id, end_id, c);
    }

    double IzhikevichModelDataHandleImpl::get_secondary_variable(const RelearnGPUTypes::neuron_id_type neuron_id) const {
        double* result = (double*)malloc(sizeof(double));
        cuda_memcpy_to_host(result, u_handle.data() + neuron_id, sizeof(double), 1);
        double host_result = *result;
        free(result);
        return host_result;
    }

    void PoissonModelDataHandleImpl::init(const RelearnGPUTypes::number_neurons_type number_neurons) {
        refractory_time_handle.resize(number_neurons);
    }

    void PoissonModelDataHandleImpl::create_neurons(RelearnGPUTypes::number_neurons_type number_neurons) {
        refractory_time_handle.resize(number_neurons);
    }

    void PoissonModelDataHandleImpl::init_neurons(NeuronModelDataHandle* gpu_handle, RelearnGPUTypes::number_neurons_type start_id, RelearnGPUTypes::number_neurons_type end_id) { }

    double PoissonModelDataHandleImpl::get_secondary_variable(const RelearnGPUTypes::neuron_id_type neuron_id) const {
        double* result = (double*)malloc(sizeof(double));
        cuda_memcpy_to_host(result, refractory_time_handle.data() + neuron_id, sizeof(double), 1);
        double host_result = *result;
        free(result);
        return host_result;
    }
}
}
