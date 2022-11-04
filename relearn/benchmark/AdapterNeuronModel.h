#pragma once

#include "main.h"
#include "neurons/models/NeuronModels.h"

#include <vector>

class AdapterNeuronModel {
    NeuronModel& model;

public:
    AdapterNeuronModel(NeuronModel& neuron_model)
        : model(neuron_model) { }

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

    void calculate_serial_initialize(const std::vector<UpdateStatus>& disable_flags) {
        //model.update_electrical_activity_serial_initialize(disable_flags);
    }

    void calculate_background_activity(const std::vector<UpdateStatus>& disable_flags) {
        // model.update_electrical_activity_calculate_background(disable_flags);
    }

    void calculate_input(const NetworkGraph& network_graph, const CommunicationMap<NeuronID>& firing_neuron_ids_incoming, const std::vector<UpdateStatus>& disable_flags) {
        // model.update_electrical_activity_calculate_input(network_graph_plastic, disable_flags);
    }

    void update_activity(const std::vector<UpdateStatus>& disable_flags) {
        // model.update_electrical_activity_update_activity(disable_flags);
    }
};
