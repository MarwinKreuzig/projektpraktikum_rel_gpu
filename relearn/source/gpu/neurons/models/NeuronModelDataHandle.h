#pragma once

#include <span>
#include <memory>
#include <vector>

#include "../../utils/GpuTypes.h"
#include "../../../shared/enums/FiredStatus.h"
#include "../../utils/Interface.h"

namespace gpu {
class NeuronModelDataHandle {
public:
    virtual void set_x(const RelearnGPUTypes::neuron_id_type neuron_id, double new_value) = 0;
    virtual void fill_x(RelearnGPUTypes::neuron_id_type start_id, RelearnGPUTypes::neuron_id_type end_id, double new_value) = 0;
    virtual void set_fired(const RelearnGPUTypes::neuron_id_type neuron_id, const FiredStatus new_value) = 0;
    virtual void set_fired(std::vector<FiredStatus>* new_values) = 0;
    virtual bool get_fired(const RelearnGPUTypes::neuron_id_type neuron_id) = 0;
    virtual void set_extra_infos(const std::unique_ptr<gpu::neurons::NeuronsExtraInfosHandle>& extra_infos_handle) = 0;
    virtual RelearnGPUTypes::number_neurons_type get_extra_infos_number_local_neurons() = 0;
    virtual std::vector<FiredStatus> get_fired() const noexcept = 0;
};

namespace models {
    class ModelDataHandle {
    public:
        virtual ~ModelDataHandle() = default;

        virtual void init(const RelearnGPUTypes::number_neurons_type number_neurons) = 0;

        virtual void create_neurons(RelearnGPUTypes::number_neurons_type creation_count) = 0;

        virtual void init_neurons(NeuronModelDataHandle* gpu_handle, RelearnGPUTypes::number_neurons_type start_id, RelearnGPUTypes::number_neurons_type end_id) = 0;
    };
}
}
