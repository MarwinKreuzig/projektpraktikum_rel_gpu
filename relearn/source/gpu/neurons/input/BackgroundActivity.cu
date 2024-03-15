#include "BackgroundActivity.cuh"
#include "../../Commons.cuh"
#include <memory.h>

namespace gpu::background {
__device__ BackgroundActivity::BackgroundActivity() { }

__device__ double BackgroundActivity::get(size_t step, size_t neuron_id) {
    double b;
    if (extra_infos->disable_flags[neuron_id] == UpdateStatus::Disabled) {
        b = 0.0;
    } else {
        b = get_internal(step, neuron_id);
    }
    background_cache[neuron_id] = b;
    return b;
}

__device__ void BackgroundActivity::set_extra_infos(gpu::neurons::NeuronsExtraInfos* _extra_infos) {
    extra_infos = _extra_infos;
}

__device__ gpu::neurons::NeuronsExtraInfos* BackgroundActivity::get_extra_infos() {
    RelearnGPUException::device_check(extra_infos != nullptr, "BackgroundActivity::get_extra_infos: Pointer is null");
    return extra_infos;
}

__device__ Constant::Constant(double c)
    : BackgroundActivity()
    , constant(c) {
}

__device__ double Constant::get_internal(size_t step, size_t neuron_id) const {
    return constant;
}

__device__ Normal::Normal(double _mean, double _stddev)
    : BackgroundActivity()
    , mean(_mean)
    , stddev(_stddev) {
}

__global__ void update_input_for_all_neurons_kernel(gpu::background::BackgroundActivity* calculator, size_t step) {
    const auto neuron_id = block_thread_to_neuron_id(blockIdx.x, threadIdx.x, blockDim.x);

    if (neuron_id >= calculator->get_extra_infos()->num_neurons) {
        return;
    }

    const auto v = calculator->get(step, neuron_id);
}

BackgroundActivityHandleImpl::BackgroundActivityHandleImpl(void* calculator)
    : background_calculator(calculator) {
    _init();
}
void BackgroundActivityHandleImpl::init(size_t num_neurons) {
    background_cache.resize(num_neurons);
}

void BackgroundActivityHandleImpl::create_neurons(size_t num_neurons) {
    background_cache.resize(background_cache.get_size() + num_neurons);
}

std::vector<double> BackgroundActivityHandleImpl::get_background_activity() {
    std::vector<double> v;
    background_cache.copy_to_host(v);
    return v;
}

void BackgroundActivityHandleImpl::set_extra_infos(const std::unique_ptr<gpu::neurons::NeuronsExtraInfosHandle>& extra_infos_handle) {
    cuda_generic_kernel<<<1, 1>>>([=] __device__(BackgroundActivity * calculator, gpu::neurons::NeuronsExtraInfos * extra_infos) { calculator->set_extra_infos(extra_infos); }, (BackgroundActivity*)background_calculator, (neurons::NeuronsExtraInfos*)(extra_infos_handle.get()->get_device_pointer()));
}

void BackgroundActivityHandleImpl::update_input_for_all_neurons_on_gpu(RelearnGPUTypes::step_type step, RelearnGPUTypes::number_neurons_type number_local_neurons) {
    RelearnGPUException::check(number_local_neurons > 0, "BackgroundActivity::update_input_for_all_neurons_on_gpu: Number neurons is 0");
    RelearnGPUException::check(background_calculator != nullptr, "BackgroundActivity::update_input_for_all_neurons_on_gpu: Device pointer is null");

    const auto num_threads = get_number_threads(update_input_for_all_neurons_kernel, number_local_neurons);
    const auto num_blocks = get_number_blocks(num_threads, number_local_neurons);

    update_input_for_all_neurons_kernel<<<num_blocks, num_threads>>>((gpu::background::BackgroundActivity*)background_calculator, step);
    cudaDeviceSynchronize();
    gpu_check_last_error();
}

void* BackgroundActivityHandleImpl::get_device_pointer() {
    return background_calculator;
}

void BackgroundActivityHandleImpl::_init() {
    void* background_cache_ptr = (void*)execute_and_copy<void*>([=] __device__(void* calculator) -> void* { return &((gpu::background::BackgroundActivity*)calculator)->background_cache; }, background_calculator);
    background_cache = gpu::Vector::CudaArrayDeviceHandle<double>(background_cache_ptr);
}

std::shared_ptr<BackgroundHandle> set_constant_background(double c) {
    void* background_calculator = init_class_on_device<gpu::background::Constant>(c);
    return std::make_shared<BackgroundActivityHandleImpl>(background_calculator);
}

std::shared_ptr<BackgroundHandle> set_normal_background(double mean, double stddev) {
    void* background_calculator = init_class_on_device<gpu::background::Normal>(mean, stddev);
    return std::make_shared<BackgroundActivityHandleImpl>(background_calculator);
}

std::shared_ptr<BackgroundHandle> set_fast_normal_background(double mean, double stddev, size_t multiplier) {
    void* background_calculator = init_class_on_device<gpu::background::Normal>(mean, stddev);
    return std::make_shared<BackgroundActivityHandleImpl>(background_calculator);
}

};