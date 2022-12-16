#pragma once

#include "main.h"
#include "neurons/models/BackgroundActivityCalculators.h"
#include "neurons/models/NeuronModels.h"
#include "neurons/models/SynapticInputCalculators.h"

#include <type_traits>
#include <vector>

template <typename NeuronModelType>
class AdapterNeuronModel {
    NeuronModelType& model;

public:
    AdapterNeuronModel(NeuronModelType& neuron_model)
        : model(neuron_model) {
    }

    const std::vector<double>& get_background() {
        return model.get_background_activity();
    }

    const std::vector<double>& get_synaptic_input() {
        return model.get_synaptic_input();
    }

    const std::vector<double>& get_x() {
        return model.get_x();
    }

    void set_fired_status(FiredStatus fs) {
        for (auto& fired_status : model.fired) {
            fired_status = fs;
        }
    }

    void update_activity(NeuronID id) {
        model.update_activity(id);
    }

    void update_activity_benchmark(NeuronID id) {
        model.update_activity_benchmark(id);
    }
};

class FactoryNeuronModel {
public:
    template <typename NeuronModelType>
    static std::unique_ptr<NeuronModelType> construct_model(const unsigned int h,
        std::unique_ptr<SynapticInputCalculator>&& synaptic_input_calculator, std::unique_ptr<BackgroundActivityCalculator>&& background_activity_calculator) {
        if constexpr (std::is_same_v<NeuronModelType, models::PoissonModel>) {
            return construct_poisson_model(h, std::move(synaptic_input_calculator), std::move(background_activity_calculator));
        } else if constexpr (std::is_same_v<NeuronModelType, models::IzhikevichModel>) {
            return construct_izhikevich_model(h, std::move(synaptic_input_calculator), std::move(background_activity_calculator));
        } else if constexpr (std::is_same_v<NeuronModelType, models::FitzHughNagumoModel>) {
            return construct_fitzhughnaguma_model(h, std::move(synaptic_input_calculator), std::move(background_activity_calculator));
        } else {
            return construct_aeif_model(h, std::move(synaptic_input_calculator), std::move(background_activity_calculator));
        }
    }

    static std::unique_ptr<models::PoissonModel> construct_poisson_model(const unsigned int h,
        std::unique_ptr<SynapticInputCalculator>&& synaptic_input_calculator, std::unique_ptr<BackgroundActivityCalculator>&& background_activity_calculator) {
        return std::make_unique<models::PoissonModel>(h, std::move(synaptic_input_calculator), std::move(background_activity_calculator),
            models::PoissonModel::default_x_0, models::PoissonModel::default_tau_x, models::PoissonModel::default_refrac_time);
    }

    static std::unique_ptr<models::IzhikevichModel> construct_izhikevich_model(const unsigned int h,
        std::unique_ptr<SynapticInputCalculator>&& synaptic_input_calculator, std::unique_ptr<BackgroundActivityCalculator>&& background_activity_calculator) {
        return std::make_unique<models::IzhikevichModel>(h, std::move(synaptic_input_calculator), std::move(background_activity_calculator),
            models::IzhikevichModel::default_a, models::IzhikevichModel::default_b, models::IzhikevichModel::default_c, models::IzhikevichModel::default_d,
            models::IzhikevichModel::default_V_spike, models::IzhikevichModel::default_k1, models::IzhikevichModel::default_k2, models::IzhikevichModel::default_k3);
    }

    static std::unique_ptr<models::FitzHughNagumoModel> construct_fitzhughnaguma_model(const unsigned int h,
        std::unique_ptr<SynapticInputCalculator>&& synaptic_input_calculator, std::unique_ptr<BackgroundActivityCalculator>&& background_activity_calculator) {
        return std::make_unique<models::FitzHughNagumoModel>(h, std::move(synaptic_input_calculator), std::move(background_activity_calculator),
            models::FitzHughNagumoModel::default_a, models::FitzHughNagumoModel::default_b, models::FitzHughNagumoModel::default_phi);
    }

    static std::unique_ptr<models::AEIFModel> construct_aeif_model(const unsigned int h,
        std::unique_ptr<SynapticInputCalculator>&& synaptic_input_calculator, std::unique_ptr<BackgroundActivityCalculator>&& background_activity_calculator) {
        return std::make_unique<models::AEIFModel>(h, std::move(synaptic_input_calculator), std::move(background_activity_calculator),
            models::AEIFModel::default_C, models::AEIFModel::default_g_L, models::AEIFModel::default_E_L, models::AEIFModel::default_V_T,
            models::AEIFModel::default_d_T, models::AEIFModel::default_tau_w, models::AEIFModel::default_a, models::AEIFModel::default_b, models::AEIFModel::default_V_spike);
    }

    static std::unique_ptr<SynapticInputCalculator> construct_linear_input(const double synapse_conductance = 1.0) {
        return std::make_unique<LinearSynapticInputCalculator>(synapse_conductance);
    }

    static std::unique_ptr<SynapticInputCalculator> construct_logarithmic_input(const double synapse_conductance = 1.0, const double scaling_factor = 1.0) {
        return std::make_unique<LogarithmicSynapticInputCalculator>(synapse_conductance, scaling_factor);
    }

    static std::unique_ptr<BackgroundActivityCalculator> construct_null_background() {
        return std::make_unique<NullBackgroundActivityCalculator>();
    }

    static std::unique_ptr<ConstantBackgroundActivityCalculator> construct_constant_background(const double background = 1.0) {
        return std::make_unique<ConstantBackgroundActivityCalculator>(background);
    }

    static std::unique_ptr<NormalBackgroundActivityCalculator> construct_normal_background(const double mean = 1.0, const double stddev = 1.0) {
        return std::make_unique<NormalBackgroundActivityCalculator>(mean, stddev);
    }
};
