#include "SynapticElements.cuh"

#include "../../Commons.cuh"
#include <cuda.h>

namespace gpu::models {

std::unique_ptr<SynapticElementsHandle> create_synaptic_elements(const ElementType type) {
    SynapticElements* synaptic_elements_dev_ptr = init_class_on_device<SynapticElements>(type);

    auto a = std::make_unique<SynapticElementsHandleImpl>(synaptic_elements_dev_ptr, type);
    return std::move(a);
}

SynapticElementsHandleImpl::SynapticElementsHandleImpl(SynapticElements* _dev_ptr, const ElementType type)
    : device_ptr(_dev_ptr)
    , type(type) {
    _init();
}

void SynapticElementsHandleImpl::_init() {
    void* grown_elements_ptr = execute_and_copy<void*>([=] __device__(SynapticElements * synaptic_elements) { return (void*)&synaptic_elements->grown_elements; }, /*(neurons::NeuronsExtraInfos*)*/ device_ptr);
    handle_grown_elements = gpu::Vector::CudaArrayDeviceHandle<double>(grown_elements_ptr);

    void* connected_elements_ptr = execute_and_copy<void*>([=] __device__(SynapticElements * synaptic_elements) { return (void*)&synaptic_elements->connected_elements; }, /*(neurons::NeuronsExtraInfos*)*/ device_ptr);
    handle_connected_elements = gpu::Vector::CudaArrayDeviceHandle<unsigned int>(connected_elements_ptr);

    void* signal_types_ptr = execute_and_copy<void*>([=] __device__(SynapticElements * synaptic_elements) { return (void*)&synaptic_elements->signal_types; }, /*(neurons::NeuronsExtraInfos*)*/ device_ptr);
    handle_signal_types = gpu::Vector::CudaArrayDeviceHandle<SignalType>(signal_types_ptr);

    RelearnGPUTypes::number_neurons_type* size_ptr = execute_and_copy<RelearnGPUTypes::number_neurons_type*>([=] __device__(SynapticElements * synaptic_elements) { return &synaptic_elements->size; }, /*(neurons::NeuronsExtraInfos*)*/ device_ptr);
    handle_size = size_ptr;
}

void SynapticElementsHandleImpl::init(RelearnGPUTypes::number_neurons_type number_neurons, const std::vector<double>& grown_elements) {
    cuda_memcpy_to_device((void*)handle_size, (void*)&number_neurons, sizeof(RelearnGPUTypes::number_neurons_type), 1);
    handle_grown_elements.copy_to_device(grown_elements);
    handle_connected_elements.resize(number_neurons, 0);
    handle_signal_types.resize(number_neurons);
}

void SynapticElementsHandleImpl::create_neurons(const RelearnGPUTypes::number_neurons_type new_size, const std::vector<double>& grown_elements) {
    cuda_memcpy_to_device((void*)handle_size, (void*)&new_size, sizeof(RelearnGPUTypes::number_neurons_type), 1);
    handle_grown_elements.copy_to_device(grown_elements);
    handle_connected_elements.resize(new_size, 0);
    handle_signal_types.resize(new_size);
}

[[nodiscard]] void* SynapticElementsHandleImpl::get_device_pointer() {
    return device_ptr;
}

// Once everything is on GPU, this should not need to be called
void SynapticElementsHandleImpl::update_grown_elements(const RelearnGPUTypes::neuron_id_type neuron_id, const double delta) {
    double* grown_elements_gpu = handle_grown_elements.data();
    double to_set;
    cuda_memcpy_to_host((void*)(grown_elements_gpu + neuron_id), (void*)&to_set, sizeof(double), 1);
    to_set += delta;
    cuda_memcpy_to_device((void*)(grown_elements_gpu + neuron_id), (void*)&to_set, sizeof(double), 1);
}

void SynapticElementsHandleImpl::update_connected_elements(const RelearnGPUTypes::neuron_id_type neuron_id, const int delta) {
    unsigned int* connected_elements_gpu = handle_connected_elements.data();
    unsigned int to_set;
    cuda_memcpy_to_host((void*)(connected_elements_gpu + neuron_id), (void*)&to_set, sizeof(unsigned int), 1);
    to_set += delta;
    cuda_memcpy_to_device((void*)(connected_elements_gpu + neuron_id), (void*)&to_set, sizeof(unsigned int), 1);
}

void SynapticElementsHandleImpl::set_signal_types(const std::vector<SignalType>& types) {
    handle_signal_types.copy_to_device(types);
}

};