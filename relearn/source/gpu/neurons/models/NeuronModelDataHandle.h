#pragma once

#include <span>
#include <memory>
#include <vector>

#include "../../utils/GpuTypes.h"
#include "../../../shared/enums/FiredStatus.h"
#include "../../utils/Interface.h"

namespace gpu {
namespace neurons {
    class NeuronsExtraInfos;
}
namespace background {
    class BackgroundActivity;
}

/**
 * This handle contains the gpu data for NeuronModelGPU. Its only implementation is
 * NeuronModelDataHandleImpl in NeuronModelData.cuh.
 */
class NeuronModelDataHandle {
public:
    virtual ~NeuronModelDataHandle() = default;

    virtual void set_x(const RelearnGPUTypes::neuron_id_type neuron_id, double new_value) = 0;
    virtual void fill_x(RelearnGPUTypes::neuron_id_type start_id, RelearnGPUTypes::neuron_id_type end_id, double new_value) = 0;
    virtual void set_fired(const RelearnGPUTypes::neuron_id_type neuron_id, const FiredStatus new_value) = 0;
    virtual void set_fired(std::vector<FiredStatus>* new_values) = 0;
    virtual bool get_fired(const RelearnGPUTypes::neuron_id_type neuron_id) = 0;
    virtual void set_extra_infos(const std::unique_ptr<gpu::neurons::NeuronsExtraInfosHandle>& extra_infos_handle) = 0;
    virtual RelearnGPUTypes::number_neurons_type get_extra_infos_number_local_neurons() = 0;
    virtual std::vector<FiredStatus> get_fired() const noexcept = 0;
    virtual void* get_device_ptr() = 0;
    virtual void init(RelearnGPUTypes::number_neurons_type number_neurons) = 0;
};

std::shared_ptr<NeuronModelDataHandle> create_neuron_model_data(std::vector<double>* x, gpu::neurons::NeuronsExtraInfos* extra_infos, unsigned int h, double scale, size_t cur_step, gpu::background::BackgroundHandle* background_calculator, std::vector<double>* stimulus, std::vector<double>* syn_input, std::vector<FiredStatus>* fired);

namespace models {
    /**
     * This handle contains the gpu data for the specific neuron models.
     * There is an implementation for every neuron model in NeuronModelData.cuh.
     */
    class ModelDataHandle {
    public:
        virtual ~ModelDataHandle() = default;

        virtual void init(const RelearnGPUTypes::number_neurons_type number_neurons) = 0;

        virtual void create_neurons(RelearnGPUTypes::number_neurons_type creation_count) = 0;

        virtual void init_neurons(NeuronModelDataHandle* gpu_handle, RelearnGPUTypes::number_neurons_type start_id, RelearnGPUTypes::number_neurons_type end_id) = 0;

        virtual double get_secondary_variable(const RelearnGPUTypes::neuron_id_type neuron_id) const = 0;
    };

    /**
     * @brief Creates an new AEIFModelGPU instance.
     */
    std::unique_ptr<ModelDataHandle> create_aeif_model_data(double E_L);
    /**
     * @brief Creates an new FitzHughNagumoModelGPU instance.
     */
    std::unique_ptr<ModelDataHandle> create_fitzhughnagumo_model_data(double init_w, double init_x);
    /**
     * @brief Creates an new IzhikevichModelGPU instance.
     */
    std::unique_ptr<ModelDataHandle> create_izhikevich_model_data(double c);
    /**
     * @brief Creates an new PoissonModelGPU instance.
     */
    std::unique_ptr<ModelDataHandle> create_poisson_model_data();
}
}
