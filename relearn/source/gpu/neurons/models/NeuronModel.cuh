#pragma once

#include "enums/FiredStatus.h"

#include "Commons.cuh"
#include "background/BackgroundActivity.cuh"
#include "cuda/gpu/GpuTypes.h"
#include "cuda/gpu/Interface.h"
#include "cuda/NeuronsExtraInfos.cuh"

#include "cuda/CudaArray.cuh"
#include "cuda/CudaVector.cuh"

#include <numeric>

namespace gpu::models {

class NeuronModel {

protected:
    gpu::Vector::CudaVector<double> x;

    gpu::neurons::NeuronsExtraInfos* extra_infos;

    unsigned int h;
    double scale;

    size_t cur_step;

    gpu::background::BackgroundActivity* background_calculator;

public:
    gpu::Vector::CudaArray<double> stimulus;

    gpu::Vector::CudaArray<double> syn_input;

    gpu::Vector::CudaArray<FiredStatus> fired;

    __device__ NeuronModel(const unsigned int _h, void* gpu_background_calculator) {
        h = _h;
        const auto _scale = 1.0 / _h;
        scale = _scale;
        background_calculator = (gpu::background::BackgroundActivity*)gpu_background_calculator;
    }

    /**
     * @brief Initializes the model to include number_neurons many local neurons.
     *      Sets the initial membrane potential and initial synaptic inputs to 0.0 and fired to false
     * @param number_neurons The number of local neurons to store in this class
     */
    __device__ virtual void init(const RelearnGPUTypes::number_neurons_type number_neurons) {
        x.resize(number_neurons);
    }

    /**
     * @brief Creates new neurons and adds those to the local portion.
     * @param creation_count The number of local neurons that should be added
     */
    __device__ virtual void create_neurons(RelearnGPUTypes::number_neurons_type creation_count) {
    }

    /**
     * @brief Provides a hook to initialize all neurons with local id in [start_id, end_id)
     *      This method exists because of the order of operations when creating neurons
     * @param start_id The first local neuron id to initialize
     * @param end_id The next to last local neuron id to initialize
     */
    __device__ virtual void init_neurons(const RelearnGPUTypes::number_neurons_type start_id, const RelearnGPUTypes::number_neurons_type end_id) { }

    /**
     * Returns the membrane potential of the neuron
     * @param neuron_id The neuron id
     * @return Membrane potential of the neuron
     */
    __device__ inline double get_x(RelearnGPUTypes::neuron_id_type neuron_id) {
        return x[neuron_id];
    }

    /**
     * Set the membrane potential of the neuron
     * @param neuron_id The neuron id
     * @param _x New membrane potential of the neuron
     */
    __device__ inline void set_x(const RelearnGPUTypes::neuron_id_type neuron_id, double _x) {
        x[neuron_id] = _x;
    }

    /**
     * Sets the fire status of the neuron
     * @param neuron_id The neuron id
     * @param _fired Fire status of the neuron
     */
    __device__ inline void set_fired(const RelearnGPUTypes::neuron_id_type neuron_id, FiredStatus _fired) {
        fired[neuron_id] = _fired;
    }

    /**
     * @brief Returns the neurons' respective stimulation in the current simulation step
     * @return External stimulation
     */
    __device__ inline double get_stimulus(RelearnGPUTypes::step_type step, const RelearnGPUTypes::neuron_id_type neuron_id) {
        return stimulus[neuron_id];
    }

    /**
     * @brief Returns the neurons' respective background activity in the current simulation step
     * @return Background activity
     */
    __device__ inline double get_background_activity(RelearnGPUTypes::step_type step, const RelearnGPUTypes::neuron_id_type neuron_id) {
        return background_calculator->get(step, neuron_id);
    }

    /**
     * @brief Returns the neurons' respective synaptic input in the current simulation step
     * @return Synaptic input
     */
    __device__ inline double get_synaptic_input(RelearnGPUTypes::step_type step, const RelearnGPUTypes::neuron_id_type neuron_id) {
        return syn_input[neuron_id];
    }

    /**
     * Sets the NeuronsExtraInfos
     * @param _extra_infos Pointer to the NeuronsExtraInfos instance on the utils
     */
    __device__ void set_extra_infos(neurons::NeuronsExtraInfos* _extra_infos) {
        extra_infos = _extra_infos;
    }

    /**
     * Virtual method that does the actual activity update
     * @param step Current step
     */
    __device__ virtual void update_activity(RelearnGPUTypes::step_type step) = 0;
};

/**
 * The kernel to update the activity of the neuron model
 * @param Pointer to the neuron model on the utils
 * @param step The current step
 */
__global__ void update_activity_kernel(NeuronModel* neuron_model, RelearnGPUTypes::step_type step) {
    neuron_model->update_activity(step);
}

class NeuronModelHandleImpl : public gpu::models::NeuronModelHandle {
public:
    NeuronModelHandleImpl(void* _dev_ptr)
        : device_ptr(_dev_ptr) {
        _init();
    }

    void* get_device_pointer() const {
        return device_ptr;
    }

    void _init() {
        void* stimulus_ptr = (void*)execute_and_copy<void*>([=] __device__(void* neuron_model) { return &((NeuronModel*)neuron_model)->stimulus; }, device_ptr);
        void* syn_input_ptr = (void*)execute_and_copy<void*>([=] __device__(void* neuron_model) { return &((NeuronModel*)neuron_model)->syn_input; }, device_ptr);

        handle_stimulation = gpu::Vector::CudaArrayDeviceHandle<double>(stimulus_ptr);
        handle_syn_input = gpu::Vector::CudaArrayDeviceHandle<double>(syn_input_ptr);
    }

    gpu::Vector::CudaArrayDeviceHandle<FiredStatus> handle_fired;

    void set_extra_infos(const std::unique_ptr<gpu::neurons::NeuronsExtraInfosHandle>& extra_infos_handle) override {
        cuda_generic_kernel<<<1, 1>>>([=] __device__(NeuronModel * neuron_model, gpu::neurons::NeuronsExtraInfos * extra_infos) { neuron_model->set_extra_infos(extra_infos); }, (NeuronModel*)device_ptr,
            (gpu::neurons::NeuronsExtraInfos*)(static_cast<neurons::NeuronsExtraInfosHandleImpl*>(extra_infos_handle.get())->get_device_pointer()));
    }

    void init_neuron_model(const RelearnGPUTypes::number_neurons_type number_neurons) {
        gpu_check_last_error();
        cudaDeviceSynchronize();
        cuda_generic_kernel<<<1, 1>>>([=] __device__(NeuronModel * neuron_model, size_t number_neurons) {
            neuron_model->init(number_neurons);
        },
            (NeuronModel*)device_ptr, number_neurons);
        gpu_check_last_error();
        cudaDeviceSynchronize();
        gpu_check_last_error();

        void* fired_ptr = execute_and_copy<void*>([=] __device__(NeuronModel * neuron_model) -> void* { return (void*)&(neuron_model->fired); }, (NeuronModel*)device_ptr);
        handle_fired = gpu::Vector::CudaArrayDeviceHandle<FiredStatus>(fired_ptr);
        handle_fired.resize(number_neurons);
    }

    void init_neurons(const RelearnGPUTypes::number_neurons_type start_id, const RelearnGPUTypes::number_neurons_type end_id) {
        cuda_generic_kernel<<<1, 1>>>([] __device__(NeuronModel * neuron_model, size_t start_id, size_t end_id) { neuron_model->init_neurons(start_id, end_id); }, (NeuronModel*)device_ptr, start_id, end_id);
        gpu_check_last_error();
        cudaDeviceSynchronize();
    }

    void create_neurons(RelearnGPUTypes::neuron_id_type creation_count) {
        cuda_generic_kernel<<<1, 1>>>([] __device__(NeuronModel * neuron_model, size_t number_neurons) { neuron_model->create_neurons(number_neurons); }, (NeuronModel*)device_ptr, creation_count);
        gpu_check_last_error();
        cudaDeviceSynchronize();
    }

    void disable_neurons(const std::vector<RelearnGPUTypes::neuron_id_type>& neuron_ids) {
    }

    void enable_neurons(const std::vector<RelearnGPUTypes::neuron_id_type>& neuron_ids) override {
    }

    std::vector<FiredStatus> get_fired() override {
        handle_fired.copy_to_host(vec_f);

        /*size_t fired = 0;
        for(const auto e:vec_f) {
            if(e==FiredStatus::Fired) {
                fired++;
            }
        }

        std::cout << "Fired " << fired << "\n";*/

        return vec_f;
    }

    void update_activity(RelearnGPUTypes::step_type step, const std::vector<double>& syn_input, const std::vector<double>& stimulation) override {
        const auto number_local_neurons = syn_input.size();
        handle_stimulation.copy_to_device(stimulation.data(), number_local_neurons);
        handle_syn_input.copy_to_device(syn_input.data(), number_local_neurons);

        const auto num_threads = get_number_threads(update_activity_kernel, number_local_neurons);
        const auto num_blocks = get_number_blocks(num_threads, number_local_neurons);

        update_activity_kernel<<<num_blocks, num_threads>>>((NeuronModel*)device_ptr, step);
    }

private:
    void* device_ptr;
    gpu::Vector::CudaArrayDeviceHandle<double> handle_syn_input;
    gpu::Vector::CudaArrayDeviceHandle<double> handle_stimulation;

    std::vector<FiredStatus> vec_f{};
};

template <typename T, typename... Args>
std::shared_ptr<NeuronModelHandle> construct(std::shared_ptr<gpu::background::BackgroundHandle> background_handle, double _h, Args... args) {
    RelearnGPUException::check(background_handle != nullptr, "NeuronModel::construct: Background activity not set");

    void* model = (void*)init_class_on_device<T>(_h, (gpu::background::BackgroundActivity*)(static_cast<gpu::background::BackgroundActivityHandleImpl*>(background_handle.get())->get_device_pointer()), args...);
    return std::make_shared<NeuronModelHandleImpl>((void*)model);
}

};