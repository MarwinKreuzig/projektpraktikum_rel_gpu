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

   



class NeuronModel {

    protected:

gpu::Vector::CudaVector<double> x;

gpu::neurons::NeuronsExtraInfos::NeuronsExtraInfos* extra_infos;



unsigned int h;
double scale;

size_t cur_step;

gpu::background::BackgroundActivity* background_calculator;

public:

gpu::Vector::CudaArray<double> stimulus;

gpu::Vector::CudaArray<double> syn_input;

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
    //extra_infos->create_neurons(creation_count);        
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

__device__ void set_extra_infos(neurons::NeuronsExtraInfos::NeuronsExtraInfos* _extra_infos) {
    extra_infos = _extra_infos;
}

__device__ virtual void update_activity(size_t step) =0;

};

__global__ void update_activity_kernel(NeuronModel* neuron_model, size_t step) {
    neuron_model->update_activity(step);
}


class  NeuronModelHandleImpl : public  gpu::models::NeuronModelHandle {
    public:
    NeuronModelHandleImpl(void* _dev_ptr) : device_ptr(_dev_ptr) {
        _init();
    }

    void* get_device_pointer() const {
        return device_ptr;
    }

    void _init() {
        void* stimulus_ptr = (void*) execute_and_copy<void*>([=] __device__ (void* neuron_model) {return &((NeuronModel*)neuron_model)->stimulus;}, device_ptr);
        void* syn_input_ptr = (void*) execute_and_copy<void*>([=] __device__ (void* neuron_model) {return &((NeuronModel*)neuron_model)->syn_input;}, device_ptr);

        handle_stimulation = gpu::Vector::CudaArrayDeviceHandle<double>(stimulus_ptr);
        handle_syn_input = gpu::Vector::CudaArrayDeviceHandle<double>(syn_input_ptr);
    }

    gpu::Vector::CudaArrayDeviceHandle<FiredStatus> handle_fired;

    void set_extra_infos(const std::unique_ptr<gpu::neurons::NeuronsExtraInfos::NeuronsExtraInfosHandle>& extra_infos_handle) override {
        cuda_generic_kernel<<<1,1>>>([=]__device__(NeuronModel* neuron_model,gpu::neurons::NeuronsExtraInfos::NeuronsExtraInfos* extra_infos){ neuron_model->set_extra_infos(extra_infos);}, (NeuronModel*) device_ptr, (neurons::NeuronsExtraInfos::NeuronsExtraInfos*)static_cast<neurons::NeuronsExtraInfos::NeuronsExtraInfosHandleImpl*>(extra_infos_handle.get())->get_device_pointer());
    }

void init_neuron_model(const RelearnTypes::number_neurons_type number_neurons) {
    gpu_check_last_error();
    cudaDeviceSynchronize();
    cuda_generic_kernel<<<1,1>>>([=]__device__(NeuronModel* neuron_model,size_t number_neurons){
        neuron_model->init(number_neurons);
        }, (NeuronModel*)device_ptr, number_neurons);
    gpu_check_last_error();
    cudaDeviceSynchronize();
    gpu_check_last_error();

    void* fired_ptr = execute_and_copy<void*>([=] __device__ (NeuronModel* neuron_model) -> void* {return (void*)&(neuron_model->fired);}, (NeuronModel*)device_ptr);
    handle_fired = gpu::Vector::CudaArrayDeviceHandle<FiredStatus>(fired_ptr);
    handle_fired.resize(number_neurons);
}

void init_neurons(const RelearnTypes::number_neurons_type start_id, const RelearnTypes::number_neurons_type end_id) {
    cuda_generic_kernel<<<1,1>>>([]__device__(NeuronModel* neuron_model,size_t start_id, size_t end_id){neuron_model->init_neurons(start_id, end_id);},(NeuronModel*)device_ptr, start_id, end_id);
    gpu_check_last_error();
    cudaDeviceSynchronize();
}

void create_neurons(size_t creation_count) {
    cuda_generic_kernel<<<1,1>>>([]__device__(NeuronModel* neuron_model,size_t number_neurons){neuron_model->create_neurons(number_neurons);}, (NeuronModel*)device_ptr,creation_count);  
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


void update_activity(size_t step, const double* syn_input, const double* stimulation, size_t number_local_neurons) {

    handle_stimulation.copy_to_device(stimulation, number_local_neurons);
    handle_syn_input.copy_to_device(syn_input, number_local_neurons);

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


template<typename T,typename... Args>
std::shared_ptr<NeuronModelHandle> construct(std::shared_ptr<gpu::background::BackgroundHandle> background_handle, double _h, Args...args) {
    RelearnGPUException::check(background_handle != nullptr, "NeuronModel::construct: Background activity not set");


    void* model = (void*)init_class_on_device<T>(_h,(gpu::background::BackgroundActivity*) (static_cast<gpu::background::BackgroundActivityHandleImpl*>(background_handle.get())->get_device_pointer()),args...);
    return std::make_shared<NeuronModelHandleImpl>((void*)model);
}

};