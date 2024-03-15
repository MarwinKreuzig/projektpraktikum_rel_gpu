#include "NeuronModelData.cuh"

#include "../../utils/Interface.h"
#include "../NeuronsExtraInfos.cuh"
#include "../input/BackgroundActivity.cuh"

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

NeuronModelDataHandleImpl::NeuronModelDataHandleImpl(NeuronModelData* dev_ptr, std::vector<double>* x, gpu::neurons::NeuronsExtraInfos* extra_infos, unsigned int h, double scale, size_t cur_step, gpu::background::BackgroundActivity* background_calculator, std::vector<double>* stimulus, std::vector<double>* syn_input, std::vector<FiredStatus>* fired)
    : device_ptr{ dev_ptr }
    , extra_infos_handle{ extra_infos }
    , background_calculator_handle{ background_calculator } {
    _init(x, extra_infos, h, scale, cur_step, background_calculator, stimulus, syn_input, fired);
}

void NeuronModelDataHandleImpl::_init(std::vector<double>* x, gpu::neurons::NeuronsExtraInfos* extra_infos, unsigned int h, double scale, size_t cur_step, gpu::background::BackgroundActivity* background_calculator, std::vector<double>* stimulus, std::vector<double>* syn_input, std::vector<FiredStatus>* fired) {
    h_handle = (unsigned int*)(execute_and_copy<void*>([=] __device__(NeuronModelData * neuron_model_data) { return (void*)&neuron_model_data->h; }, device_ptr));
    scale_handle = (double*)execute_and_copy<void*>([=] __device__(NeuronModelData * neuron_model_data) { return (void*)&neuron_model_data->scale; }, device_ptr);
    cur_step_handle = (size_t*)execute_and_copy<void*>([=] __device__(NeuronModelData * neuron_model_data) { return (void*)&neuron_model_data->cur_step; }, device_ptr);

    void* x_ptr = execute_and_copy<void*>([=] __device__(NeuronModelData * neuron_model_data, unsigned int h, double scale, size_t cur_step, gpu::neurons::NeuronsExtraInfos* extra_infos, gpu::background::BackgroundActivity* background_calculator) {
        neuron_model_data->h = h;
        neuron_model_data->scale = scale;
        neuron_model_data->cur_step = cur_step;
        neuron_model_data->extra_infos = extra_infos;
        neuron_model_data->background_calculator = background_calculator;

        return (void*)&neuron_model_data->x;
    },
        device_ptr, h, scale, cur_step, extra_infos, background_calculator);

    // void* x_ptr = execute_and_copy<void*>([=] __device__(NeuronModelData * neuron_model_data) { return (void*)&neuron_model_data->x; }, device_ptr);
    x_handle = gpu::Vector::CudaArrayDeviceHandle<double>(x_ptr);
    x_handle.copy_to_device(*x);

    void* stimulus_ptr = execute_and_copy<void*>([=] __device__(NeuronModelData * neuron_model_data) { return (void*)&neuron_model_data->stimulus; }, device_ptr);
    stimulus_handle = gpu::Vector::CudaArrayDeviceHandle<double>(stimulus_ptr);
    stimulus_handle.copy_to_device(*stimulus);

    void* syn_input_ptr = execute_and_copy<void*>([=] __device__(NeuronModelData * neuron_model_data) { return (void*)&neuron_model_data->syn_input; }, device_ptr);
    syn_input_handle = gpu::Vector::CudaArrayDeviceHandle<double>(syn_input_ptr);
    syn_input_handle.copy_to_device(*syn_input);

    void* fired_ptr = execute_and_copy<void*>([=] __device__(NeuronModelData * neuron_model_data) { return (void*)&neuron_model_data->fired; }, device_ptr);
    fired_handle = gpu::Vector::CudaArrayDeviceHandle<FiredStatus>(fired_ptr);
    fired_handle.copy_to_device(*fired);
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

void NeuronModelDataHandleImpl::init(RelearnGPUTypes::number_neurons_type number_neurons) {
    x_handle.resize(number_neurons, 0.0);
    fired_handle.resize(number_neurons, FiredStatus::Inactive);
    stimulus_handle.resize(number_neurons);
    syn_input_handle.resize(number_neurons);
}

std::vector<FiredStatus> NeuronModelDataHandleImpl::get_fired() const noexcept {
    std::vector<FiredStatus> fired_data;
    fired_handle.copy_to_host(fired_data);
    return fired_data;
}

RelearnGPUTypes::number_neurons_type NeuronModelDataHandleImpl::get_extra_infos_number_local_neurons() {
    auto result = (RelearnGPUTypes::number_neurons_type*)cuda_malloc(sizeof(RelearnGPUTypes::number_neurons_type));

    cuda_generic_kernel<<<1, 1>>>([=] __device__(gpu::neurons::NeuronsExtraInfos * extra_infos_handle, RelearnGPUTypes::number_neurons_type * number_neurons) { *number_neurons = extra_infos_handle->num_neurons; }, extra_infos_handle, result);
    cudaDeviceSynchronize();
    gpu_check_last_error();

    RelearnGPUTypes::number_neurons_type host_result;
    cuda_memcpy_to_host(&host_result, result, sizeof(RelearnGPUTypes::number_neurons_type), 1);

    return host_result;
}

void* NeuronModelDataHandleImpl::get_device_ptr() {
    return (void*)device_ptr;
}

std::shared_ptr<NeuronModelDataHandle> create_neuron_model_data(std::vector<double>* x, gpu::neurons::NeuronsExtraInfos* extra_infos, unsigned int h, double scale, size_t cur_step, gpu::background::BackgroundHandle* background_calculator, std::vector<double>* stimulus, std::vector<double>* syn_input, std::vector<FiredStatus>* fired) {
    NeuronModelData* neuron_model_data_dev_ptr = init_class_on_device<NeuronModelData>();

    auto a = std::make_shared<NeuronModelDataHandleImpl>(neuron_model_data_dev_ptr, x, extra_infos, h, scale, cur_step, (background::BackgroundActivity*)((background::BackgroundActivityHandleImpl*)background_calculator)->get_device_pointer(), stimulus, syn_input, fired);
    return std::move(a);
}

namespace models {
    std::unique_ptr<ModelDataHandle> create_aeif_model_data(double E_L) {
        AEIFModelData* data_dev_ptr = init_class_on_device<AEIFModelData>();

        std::vector<double> w_data{};
        auto a = std::make_unique<AEIFModelDataHandleImpl>(data_dev_ptr, &w_data, E_L);
        return std::move(a);
    }
    std::unique_ptr<ModelDataHandle> create_fitzhughnagumo_model_data(double init_w, double init_x) {
        FitzHughNagumoModelData* data_dev_ptr = init_class_on_device<FitzHughNagumoModelData>();

        std::vector<double> w_data{};
        auto a = std::make_unique<FitzHughNagumoModelDataHandleImpl>(data_dev_ptr, &w_data, init_w, init_x);
        return std::move(a);
    }
    std::unique_ptr<ModelDataHandle> create_izhikevich_model_data(double c) {
        IzhikevichModelData* data_dev_ptr = init_class_on_device<IzhikevichModelData>();

        std::vector<double> u_data{};
        auto a = std::make_unique<IzhikevichModelDataHandleImpl>(data_dev_ptr, &u_data, c);
        return std::move(a);
    }
    std::unique_ptr<ModelDataHandle> create_poisson_model_data() {
        PoissonModelData* data_dev_ptr = init_class_on_device<PoissonModelData>();

        std::vector<double> refractory_time_data{};
        auto a = std::make_unique<PoissonModelDataHandleImpl>(data_dev_ptr, &refractory_time_data);
        return std::move(a);
    }

    AEIFModelDataHandleImpl::AEIFModelDataHandleImpl(AEIFModelData* dev_ptr, std::vector<double>* w_data, double E_L_)
        : device_ptr{ dev_ptr }
        , E_L{ E_L_ } {
        _init(w_data);
    }
    void AEIFModelDataHandleImpl::_init(std::vector<double>* w_data) {
        void* w_ptr = execute_and_copy<void*>([=] __device__(AEIFModelData * aeif_model_data) { return (void*)&aeif_model_data->w; }, device_ptr);
        w_handle = gpu::Vector::CudaArrayDeviceHandle<double>(w_ptr);
        w_handle.copy_to_device(*w_data);
    }

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

    FitzHughNagumoModelDataHandleImpl::FitzHughNagumoModelDataHandleImpl(FitzHughNagumoModelData* dev_ptr, std::vector<double>* w_data, double init_w_, double init_x_)
        : device_ptr{ dev_ptr }
        , init_w{ init_w_ }
        , init_x{ init_x_ } {
        _init(w_data);
    }

    void FitzHughNagumoModelDataHandleImpl::_init(std::vector<double>* w_data) {
        void* w_ptr = execute_and_copy<void*>([=] __device__(FitzHughNagumoModelData * fitz_hugh_nagumo_model_data) { return (void*)&fitz_hugh_nagumo_model_data->w; }, device_ptr);
        w_handle = gpu::Vector::CudaArrayDeviceHandle<double>(w_ptr);
        w_handle.copy_to_device(*w_data);
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

    IzhikevichModelDataHandleImpl::IzhikevichModelDataHandleImpl(IzhikevichModelData* dev_ptr, std::vector<double>* u_data, double c_)
        : device_ptr{ dev_ptr }
        , c{ c_ } {
        _init(u_data);
    }

    void IzhikevichModelDataHandleImpl::_init(std::vector<double>* u_data) {

        void* u_ptr = execute_and_copy<void*>([=] __device__(IzhikevichModelData * izhikevich_model_data) { return (void*)&izhikevich_model_data->u; }, device_ptr);
        u_handle = gpu::Vector::CudaArrayDeviceHandle<double>(u_ptr);
        u_handle.copy_to_device(*u_data);
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

    PoissonModelDataHandleImpl::PoissonModelDataHandleImpl(PoissonModelData* dev_ptr, std::vector<double>* refractory_time_data)
        : device_ptr(dev_ptr) {
        _init(refractory_time_data);
    }

    void PoissonModelDataHandleImpl::_init(std::vector<double>* refractory_time_data) {
        void* refractory_time_ptr = execute_and_copy<void*>([=] __device__(PoissonModelData * poisson_model_data) { return (void*)&poisson_model_data->refractory_time; }, device_ptr);
        refractory_time_handle = gpu::Vector::CudaArrayDeviceHandle<double>(refractory_time_ptr);
        refractory_time_handle.copy_to_device(*refractory_time_data);
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
