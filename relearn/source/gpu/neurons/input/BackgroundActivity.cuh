#pragma once

#include "../../utils/Interface.h"
#include "../../structure/CudaArray.cuh"
#include "../NeuronsExtraInfos.cuh"
#include "../../utils/Random.cuh"
#include <vector>

namespace gpu::background {

class BackgroundActivity {
    /**
     * BackgroundActivityCalculator on the gpu
     */

public:
    __device__ BackgroundActivity();

    /**
     * Returns the background activity for a certain neuron id at a certain step
     * @param step Current step
     * @param neuron_id Neuron id
     * @return The background activity for the neuron at the step
     */
    __device__ double get(size_t step, size_t neuron_id);

    /**
     * Virtual method to implement the actual background activitiy calculator
     * @param step Current step
     * @param neuron_id Neuron id
     * @return The background activity for the neuron at the step
     */
    __device__ virtual double get_internal(size_t step, size_t neuron_id) const = 0;

    /**
     * Caches the last values of background activity for every neuron to test the implementation
     */
    gpu::Vector::CudaArray<double> background_cache;

    gpu::neurons::NeuronsExtraInfos* extra_infos;

    /**
     * Sets the NeuronsExtraInfos
     * @param _extra_infos Pointer to the NeuronsExtraInfos instance on the gpu
     */
    __device__ void set_extra_infos(gpu::neurons::NeuronsExtraInfos* _extra_infos);

    /**
     * Returns the NeuronsExtraInfos
     * @return Pointer to the NeuronsExtraInfos instance on the gpu
     */
    __device__ gpu::neurons::NeuronsExtraInfos* get_extra_infos();
};

class Constant : public BackgroundActivity {

public:
    __device__ Constant(double c);

    __device__ double get_internal(size_t step, size_t neuron_id) const override;

private:
    double constant;
};

class Normal : public BackgroundActivity {

public:
    __device__ Normal(double _mean, double _stddev);

    __device__ inline double get_internal(size_t step, size_t neuron_id) const override {
        auto curand_state = gpu::RandomHolder::init(step, extra_infos->num_neurons, gpu::RandomHolder::BACKGROUND, neuron_id);
        const auto random_value = gpu::RandomHolder::get_normal(&curand_state, mean, stddev);
        return random_value;
    }

private:
    double mean;
    double stddev;
};

__global__ void update_input_for_all_neurons_kernel(gpu::background::BackgroundActivity* calculator, size_t step);

class BackgroundActivityHandleImpl : public gpu::background::BackgroundHandle {

public:
    BackgroundActivityHandleImpl(void* calculator);

    void init(size_t num_neurons) override;

    void create_neurons(size_t num_neurons) override;

    std::vector<double> get_background_activity() override;

    void set_extra_infos(const std::unique_ptr<gpu::neurons::NeuronsExtraInfosHandle>& extra_infos_handle) override;

    void update_input_for_all_neurons_on_gpu(RelearnGPUTypes::step_type step, RelearnGPUTypes::number_neurons_type number_local_neurons) override;

    void* get_device_pointer();

    void _init();

private:
    void* background_calculator;
    gpu::Vector::CudaArrayDeviceHandle<double> background_cache;
};

std::shared_ptr<BackgroundHandle> set_constant_background(double c);

std::shared_ptr<BackgroundHandle> set_normal_background(double mean, double stddev);

std::shared_ptr<BackgroundHandle> set_fast_normal_background(double mean, double stddev, size_t multiplier);

};
