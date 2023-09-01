#pragma once

#include "enums/FiredStatus.h"

#include "Commons.cuh"
#include "background/BackgroundActivity.cuh"
#include "gpu/GpuTypes.h"
#include "gpu/Interface.h"
#include "NeuronsExtraInfos.cuh"

#include "CudaArray.cuh"
#include "CudaVector.cuh"

#include <numeric>

namespace gpu::models {

    __device__ gpu::Vector::CudaArray<double> stimulus;
gpu::Vector::CudaArrayDeviceHandle<double> handle_stimulation;

__device__ gpu::Vector::CudaArray<double> syn_input;
gpu::Vector::CudaArrayDeviceHandle<double> handle_syn_input;


class NeuronModel {
    protected:

gpu::Vector::CudaVector<double> x;

unsigned int h;
double scale;

size_t cur_step;

gpu::background::BackgroundActivity* background_calculator;

public:

gpu::Vector::CudaArray<FiredStatus> fired;

__device__ NeuronModel(const unsigned int _h, void* gpu_background_calculator) {
    h =  _h;
    const auto _scale = 1.0/_h;
    scale =  _scale;
    background_calculator =  (gpu::background::BackgroundActivity*) gpu_background_calculator;
}

__device__ virtual void init(const RelearnTypes::number_neurons_type number_neurons) {
    printf("Neuron model init1\n");
    x.resize(number_neurons);
}

__device__ virtual void create_neurons(size_t creation_count) {
    gpu::neurons::NeuronsExtraInfos::extra_infos->create_neurons(creation_count);        
}

__device__ virtual void init_neurons(const RelearnTypes::number_neurons_type start_id, const RelearnTypes::number_neurons_type end_id) {}
__device__ inline double get_x(size_t neuron_id) {
    return x[neuron_id];
}

__device__ inline void set_x(const size_t neuron_id, double _x) {
    x[neuron_id] = _x;
}

__device__ inline void set_fired(const size_t neuron_id, FiredStatus _fired) {
    fired[neuron_id] = _fired;
}

__device__ inline double get_stimulus(size_t step,const size_t neuron_id) {
    return stimulus[neuron_id];
}

__device__ inline double get_background_activity(size_t step,const size_t neuron_id) {
    return background_calculator->get(step,neuron_id);

}
__device__ inline double get_synaptic_input(size_t step,const size_t neuron_id) {
    return syn_input[neuron_id];
}

__device__ virtual void update_activity(size_t step) =0;

};

__device__ NeuronModel* neuron_model;

void* neuron_model_on_device;
gpu::Vector::CudaArrayDeviceHandle<FiredStatus> handle_fired;

void init_neuron_model(const RelearnTypes::number_neurons_type number_neurons) {
    gpu_check_last_error();
    cudaDeviceSynchronize();
    cuda_generic_kernel<<<1,1>>>([=]__device__(size_t number_neurons){
        neuron_model->init(number_neurons);
        }, number_neurons);
    gpu_check_last_error();
    cudaDeviceSynchronize();
    gpu_check_last_error();

    void* fired_ptr = execute_and_copy<void*>([=] __device__ () -> void* {return (void*)&(neuron_model->fired);});
    handle_fired = gpu::Vector::CudaArrayDeviceHandle<FiredStatus>(fired_ptr);
    handle_fired.resize(number_neurons);
}

void init_neurons(const RelearnTypes::number_neurons_type start_id, const RelearnTypes::number_neurons_type end_id) {
    cuda_generic_kernel<<<1,1>>>([]__device__(size_t start_id, size_t end_id){neuron_model->init_neurons(start_id, end_id);}, start_id, end_id);
    gpu_check_last_error();
    cudaDeviceSynchronize();
}

void create_neurons(size_t creation_count) {
    cuda_generic_kernel<<<1,1>>>([]__device__(size_t number_neurons){neuron_model->create_neurons(number_neurons);}, creation_count);  
    gpu_check_last_error();
    cudaDeviceSynchronize();
}

void disable_neurons(const size_t* neuron_ids, size_t num_disabled_neurons) {
    if(num_disabled_neurons == 0) {
        return;
    }
    //cuda_generic_kernel<<<1,1>>>([]__device__(const size_t* neuron_ids, size_t num_disabled_neurons){neuron_model->disable_neurons(neuron_ids, num_disabled_neurons);},neuron_ids, num_disabled_neurons);
    gpu_check_last_error();
    cudaDeviceSynchronize();
}

void enable_neurons(const size_t* neuron_ids, size_t num_enabled_neurons) {
    if(num_enabled_neurons == 0) {
        return;
    }
}

std::vector<FiredStatus> vec_f{};

FiredStatus* get_fired() {
    handle_fired.copy_to_host(vec_f);

    size_t fired = 0;
    for(const auto e:vec_f) {
        if(e==FiredStatus::Fired) {
            fired++;
        }
    }

    std::cout << "Fired " << fired << "\n";

    return vec_f.data();
}


__global__ void update_activity_kernel(size_t step) {
    neuron_model->update_activity(step);
}

void update_activity(size_t step, const double* syn_input, const double* stimulation) {
    const auto number_local_neurons = gpu::neurons::NeuronsExtraInfos::number_local_neurons_host;

    handle_stimulation.copy_to_device(stimulation, number_local_neurons);
    handle_syn_input.copy_to_device(syn_input, number_local_neurons);

        const auto num_threads = get_number_threads(gpu::models::update_activity_kernel, number_local_neurons);
        const auto num_blocks = get_number_blocks(num_threads, number_local_neurons);

        update_activity_kernel<<<num_blocks, num_threads>>>(step);
}

gpu::background::BackgroundActivity* background_calculator;

template<typename T,typename... Args>
void construct(double _h, Args...args) {
    RelearnGPUException::check(background_calculator != nullptr, "NeuronModel::construct: Background activity not set");

    gpu_get_handle_for_device_symbol(double,handle_stimulation, stimulus);
    gpu_get_handle_for_device_symbol(double,handle_syn_input, syn_input);

    void* model = (void*)init_class_on_device<T>(_h,background_calculator,args...);
    cuda_copy_to_device(gpu::models::neuron_model, model);
    gpu::models::neuron_model_on_device = (void*)model;
}

void set_constant_background(double c) {
    background_calculator = init_class_on_device<gpu::background::Constant>(c);
}

void set_normal_background(double mean, double stddev) {
    background_calculator= init_class_on_device<gpu::background::Normal>(mean, stddev);
}
};