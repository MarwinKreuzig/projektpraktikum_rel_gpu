#pragma once

#include "gpu/Macros.h"
#include "gpu/GpuTypes.h"
#include "enums/FiredStatus.h"

#include <memory>
#include <span>
#include <vector>


#ifndef CUDA_DEFINITION
#define CUDA_DEFINITION ;
#define CUDA_PTR_DEFINITION CUDA_DEFINITION
#endif

namespace gpu::neurons::NeuronsExtraInfos {
    class NeuronsExtraInfosHandle {
        public:
        virtual void disable_neurons(const size_t* neuron_ids, size_t num_disabled_neurons) =0;
    virtual void enable_neurons(const size_t* neuron_ids, size_t num_disabled_neurons) =0;
    virtual void init(size_t number_neurons)=0;
    virtual void* get_device_pointer()=0;
    virtual void create_neurons(size_t num_created_neurons)=0;
    };

    std::unique_ptr<NeuronsExtraInfosHandle> create() CUDA_PTR_DEFINITION
};


namespace gpu::background {
    class BackgroundHandle {
        public:
    virtual void init(size_t num_neurons)=0;
    virtual void create_neurons(size_t num_neurons)=0;
    virtual std::vector<double> get_background_activity()=0;
    virtual void update_input_for_all_neurons_on_gpu(size_t step, size_t number_local_neurons)=0;
    virtual void set_extra_infos(const std::unique_ptr<gpu::neurons::NeuronsExtraInfos::NeuronsExtraInfosHandle>& extra_infos_handle)=0;
    };

    std::shared_ptr<BackgroundHandle> set_constant_background(double c) CUDA_PTR_DEFINITION
    std::shared_ptr<BackgroundHandle> set_normal_background(double mean, double stddev) CUDA_PTR_DEFINITION
    std::shared_ptr<BackgroundHandle> set_fast_normal_background(double mean, double stddev, size_t multiplier) CUDA_PTR_DEFINITION
    
};

namespace gpu::models {

    class NeuronModelHandle {
        public:
    virtual FiredStatus* get_fired() =0;
    virtual void disable_neurons(const size_t* neuron_ids, size_t num_disabled_neurons) =0;
    virtual void enable_neurons(const size_t* neuron_ids, size_t num_disabled_neurons) =0;


virtual void init_neuron_model(RelearnTypes::number_neurons_type number_neurons)=0;

virtual void init_neurons(const RelearnTypes::number_neurons_type start_id, const RelearnTypes::number_neurons_type end_id)  =0;
virtual void update_activity(const size_t step, const double* syn_input, const double* stimulation, size_t number_local_neurons) =0;

virtual void create_neurons(const RelearnTypes::number_neurons_type creation_count) =0;
virtual void set_extra_infos(const std::unique_ptr<gpu::neurons::NeuronsExtraInfos::NeuronsExtraInfosHandle>& extra_infos_handle)=0;
    };
};


namespace gpu::models::izhekevich {

std::shared_ptr<NeuronModelHandle> construct_gpu(std::shared_ptr<gpu::background::BackgroundHandle> background_handle,const unsigned int h, double V_spike, double a, double b, double c, double d, double k1, double k2, double k3) CUDA_PTR_DEFINITION

};

namespace gpu::models::poisson {

std::shared_ptr<NeuronModelHandle> construct_gpu( std::shared_ptr<gpu::background::BackgroundHandle> background_handle,const unsigned int h,const double x_0,
    const double tau_x,
    const unsigned int refractory_period) CUDA_PTR_DEFINITION

};

namespace gpu::models::aeif {

std::shared_ptr<NeuronModelHandle> construct_gpu( std::shared_ptr<gpu::background::BackgroundHandle> background_handle,const unsigned int _h, double _C, double _g_L, double _E_L, double _V_T, double _d_T, double _tau_w, double _a, double _b, double _V_spike) CUDA_PTR_DEFINITION

};

namespace gpu::models::fitz_hugh_nagumo {

std::shared_ptr<NeuronModelHandle> construct_gpu(std::shared_ptr<gpu::background::BackgroundHandle> background_handle,const unsigned int _h,double _a, double _b, double _phi, double _init_w, double _init_x) CUDA_PTR_DEFINITION
};



