#pragma once

#include "NeuronModelDataHandle.h"
#include "../../structure/CudaArray.cuh"
#include "../NeuronsExtraInfos.cuh"
#include "../../neurons/input/BackgroundActivity.cuh"
#include "../../utils/GpuTypes.h"

namespace gpu {
struct NeuronModelData {
    gpu::Vector::CudaArray<double> x;
    gpu::neurons::NeuronsExtraInfos* extra_infos;
    unsigned int h;
    double scale;
    size_t cur_step;
    gpu::background::BackgroundActivity* background_calculator;

    gpu::Vector::CudaArray<double> stimulus;
    gpu::Vector::CudaArray<double> syn_input;
    gpu::Vector::CudaArray<FiredStatus> fired;

    __device__ double get_synaptic_input(RelearnGPUTypes::step_type step, const RelearnGPUTypes::neuron_id_type neuron_id);
    __device__ double get_background_activity(RelearnGPUTypes::step_type step, const RelearnGPUTypes::neuron_id_type neuron_id);
    __device__ double get_stimulus(RelearnGPUTypes::step_type step, const RelearnGPUTypes::neuron_id_type neuron_id);

    __device__ void set_x(const RelearnGPUTypes::neuron_id_type neuron_id, double new_value);
    __device__ double get_x(const RelearnGPUTypes::neuron_id_type neuron_id);
    __device__ void set_fired(const RelearnGPUTypes::neuron_id_type neuron_id, FiredStatus new_value);
    __device__ void set_fired(gpu::Vector::CudaArray<FiredStatus> new_values);
};

class NeuronModelDataHandleImpl : public NeuronModelDataHandle {
public:
    NeuronModelDataHandleImpl(NeuronModelData* dev_ptr, std::vector<double>* x, gpu::neurons::NeuronsExtraInfos* extra_infos, unsigned int h, double scale, size_t cur_step, gpu::background::BackgroundActivity* background_calculator, std::vector<double>* stimulus, std::vector<double>* syn_input, std::vector<FiredStatus>* fired);
    void _init(std::vector<double>* x, gpu::neurons::NeuronsExtraInfos* extra_infos, unsigned int h, double scale, size_t cur_step, gpu::background::BackgroundActivity* background_calculator, std::vector<double>* stimulus, std::vector<double>* syn_input, std::vector<FiredStatus>* fired);

    ~NeuronModelDataHandleImpl();

    virtual void set_x(const RelearnGPUTypes::neuron_id_type neuron_id, double new_value) override;
    virtual void fill_x(RelearnGPUTypes::neuron_id_type start_id, RelearnGPUTypes::neuron_id_type end_id, double new_value) override;
    virtual void set_fired(const RelearnGPUTypes::neuron_id_type neuron_id, const FiredStatus new_value) override;
    virtual void set_fired(std::vector<FiredStatus>* new_values) override;
    virtual bool get_fired(const RelearnGPUTypes::neuron_id_type neuron_id) override;
    virtual void set_extra_infos(const std::unique_ptr<gpu::neurons::NeuronsExtraInfosHandle>& extra_infos_handle) override;
    virtual RelearnGPUTypes::number_neurons_type get_extra_infos_number_local_neurons() override;
    virtual void init(RelearnGPUTypes::number_neurons_type number_neurons) override;

    virtual std::vector<FiredStatus> get_fired() const noexcept override;
    virtual void* get_device_ptr() override;

private:
    NeuronModelData* device_ptr;
    gpu::Vector::CudaArrayDeviceHandle<double> x_handle;
    gpu::neurons::NeuronsExtraInfos* extra_infos_handle;
    unsigned int* h_handle;
    double* scale_handle;
    size_t* cur_step_handle;
    gpu::background::BackgroundActivity* background_calculator_handle;
    gpu::Vector::CudaArrayDeviceHandle<double> stimulus_handle;
    gpu::Vector::CudaArrayDeviceHandle<double> syn_input_handle;
    gpu::Vector::CudaArrayDeviceHandle<FiredStatus> fired_handle;
};

namespace models {
    struct AEIFModelData {
        gpu::Vector::CudaArray<double> w;
    };

    class AEIFModelDataHandleImpl : public ModelDataHandle {
    public:
        AEIFModelDataHandleImpl(AEIFModelData* device_ptr, std::vector<double>* w_data, double E_L);
        void _init(std::vector<double>* w_data);

        virtual ~AEIFModelDataHandleImpl() override {
            cudaFree(device_ptr);
        }

        virtual void init(const RelearnGPUTypes::number_neurons_type number_neurons) override;

        virtual void create_neurons(RelearnGPUTypes::number_neurons_type creation_count) override;

        virtual void init_neurons(NeuronModelDataHandle* gpu_handle, RelearnGPUTypes::number_neurons_type start_id, RelearnGPUTypes::number_neurons_type end_id) override;

        virtual double get_secondary_variable(const RelearnGPUTypes::neuron_id_type neuron_id) const override;

        AEIFModelData* device_ptr;

        gpu::Vector::CudaArrayDeviceHandle<double> w_handle;
        double E_L;
    };

    struct FitzHughNagumoModelData {
        gpu::Vector::CudaArray<double> w;
    };

    class FitzHughNagumoModelDataHandleImpl : public ModelDataHandle {
    public:
        FitzHughNagumoModelDataHandleImpl(FitzHughNagumoModelData* device_ptr, std::vector<double>* w_data, double init_w, double init_x);
        void _init(std::vector<double>* w_data);

        virtual ~FitzHughNagumoModelDataHandleImpl() override {
            cudaFree(device_ptr);
        }

        virtual void init(const RelearnGPUTypes::number_neurons_type number_neurons) override;

        virtual void create_neurons(RelearnGPUTypes::number_neurons_type creation_count) override;

        virtual void init_neurons(NeuronModelDataHandle* gpu_handle, RelearnGPUTypes::number_neurons_type start_id, RelearnGPUTypes::number_neurons_type end_id) override;

        virtual double get_secondary_variable(const RelearnGPUTypes::neuron_id_type neuron_id) const override;

        FitzHughNagumoModelData* device_ptr;

        gpu::Vector::CudaArrayDeviceHandle<double> w_handle;
        double init_w;
        double init_x;
    };

    struct IzhikevichModelData {
        gpu::Vector::CudaArray<double> u;
    };

    class IzhikevichModelDataHandleImpl : public ModelDataHandle {
    public:
        IzhikevichModelDataHandleImpl(IzhikevichModelData* device_ptr, std::vector<double>* u_data, double c);

        void _init(std::vector<double>* u_data);

        virtual ~IzhikevichModelDataHandleImpl() override {
            cudaFree(device_ptr);
        }

        virtual void init(const RelearnGPUTypes::number_neurons_type number_neurons) override;

        virtual void create_neurons(RelearnGPUTypes::number_neurons_type creation_count) override;

        virtual void init_neurons(NeuronModelDataHandle* gpu_handle, RelearnGPUTypes::number_neurons_type start_id, RelearnGPUTypes::number_neurons_type end_id) override;

        virtual double get_secondary_variable(const RelearnGPUTypes::neuron_id_type neuron_id) const override;

        IzhikevichModelData* device_ptr;

        gpu::Vector::CudaArrayDeviceHandle<double> u_handle;
        double c;
    };

    struct PoissonModelData {
        gpu::Vector::CudaArray<double> refractory_time;
    };

    class PoissonModelDataHandleImpl : public ModelDataHandle {
    public:
        PoissonModelDataHandleImpl(PoissonModelData* device_ptr, std::vector<double>* refractory_time_data);
        void _init(std::vector<double>* refractory_time_data);

        virtual ~PoissonModelDataHandleImpl() override {
            cudaFree(device_ptr);
        }

        virtual void init(const RelearnGPUTypes::number_neurons_type number_neurons) override;

        virtual void create_neurons(RelearnGPUTypes::number_neurons_type creation_count) override;

        virtual void init_neurons(NeuronModelDataHandle* gpu_handle, RelearnGPUTypes::number_neurons_type start_id, RelearnGPUTypes::number_neurons_type end_id) override;

        virtual double get_secondary_variable(const RelearnGPUTypes::neuron_id_type neuron_id) const override;

        PoissonModelData* device_ptr;
        gpu::Vector::CudaArrayDeviceHandle<double> refractory_time_handle;
    };
}
}
