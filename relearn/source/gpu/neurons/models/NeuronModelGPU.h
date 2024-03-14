#pragma once

#include "neurons/models/NeuronModel.h"
// #include "gpu/structure/CudaVector.cuh"
#include "util/RelearnException.h"
#include "NeuronModelDataHandle.h"

namespace gpu {
class NeuronModelGPU : public NeuronModel {
public:
    /**
     * @brief Constructs a new instance of type NeuronModelGPU with 0 neurons and default values for all parameters
     */
    NeuronModelGPU() = default;

    /**
     * @brief Constructs a new instance of type NeuronModelGPU with 0 neurons.
     * @param h The step size for the numerical integration
     * @param synaptic_input_calculator The object that is responsible for calculating the synaptic input
     * @param background_activity_calculator The object that is responsible for calculating the background activity
     * @param stimulus_calculator The object that is responsible for calculating the stimulus
     */
    NeuronModelGPU(std::unique_ptr<models::ModelDataHandle> model_data_handle, const unsigned int h, std::unique_ptr<SynapticInputCalculator>&& synaptic_input_calculator,
        std::unique_ptr<BackgroundActivityCalculator>&& background_activity_calculator, std::unique_ptr<Stimulus>&& stimulus_calculator);

    virtual std::shared_ptr<NeuronModelDataHandle> device_handle() override {
        return gpu_handle;
    }

    virtual void enable_neurons(const std::span<const NeuronID> neuron_ids) override;

    virtual void disable_neurons(const std::span<const NeuronID> neuron_ids) override;

    virtual void set_fired(const NeuronID neuron_id, const FiredStatus new_value) override;

    virtual bool get_fired(const NeuronID neuron_id) const override;
    virtual std::span<const FiredStatus> get_fired() const noexcept override;

    virtual void set_extra_infos(std::shared_ptr<NeuronsExtraInfo> new_extra_info) override;

    virtual void init_neurons(RelearnGPUTypes::number_neurons_type start_id, RelearnGPUTypes::number_neurons_type end_id) override {
        model_data_handle->init_neurons(gpu_handle.get(), start_id, end_id);
    }

    virtual void init(const RelearnGPUTypes::number_neurons_type number_neurons) override {
        NeuronModel::init(number_neurons);
        model_data_handle->init(number_neurons);
    }

    virtual void create_neurons(RelearnGPUTypes::number_neurons_type creation_count) override {
        NeuronModel::create_neurons(creation_count);
        const auto new_size = gpu_handle->get_extra_infos_number_local_neurons();
        model_data_handle->create_neurons(new_size);
    }

    virtual void update_activity_benchmark() override {
        RelearnException::fail("No gpu support");
    }

    virtual double get_secondary_variable(const NeuronID neuron_id) const override {
        RelearnException::check(neuron_id.get_neuron_id() < get_number_neurons(), "NeuronModelGPU::get_secondary_variable: id is too large: {}", neuron_id);
        return model_data_handle->get_secondary_variable(neuron_id.get_neuron_id());
    }

protected:
    std::shared_ptr<NeuronModelDataHandle> gpu_handle;
    RelearnGPUTypes::number_neurons_type number_neurons;
    std::unique_ptr<models::ModelDataHandle> model_data_handle;

    unsigned int h;
    double scale;

    size_t cur_step;
};
}
