/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "LinearSynapticInputCalculator.h"

#include "neurons/NetworkGraph.h"
#include "util/Random.h"
#include "util/Timers.h"

void LinearSynapticInputCalculator::update_synaptic_input(const NetworkGraph& network_graph, const std::vector<FiredStatus> fired, const std::vector<UpdateStatus>& disable_flags) {
    Timers::start(TimerRegion::CALC_SYNAPTIC_INPUT);

    const auto& fired_status_comm = get_fired_status_communicator();

    const auto number_local_neurons = get_number_neurons();
    const auto k = get_k();
    auto& synaptic_input = get_inner_synaptic_input();

#pragma omp parallel for shared(network_graph, disable_flags, std::ranges::binary_search) default(none)
    for (auto neuron_id = 0; neuron_id < number_local_neurons; ++neuron_id) {
        if (disable_flags[neuron_id] == UpdateStatus::Disabled) {
            continue;
        }

        NeuronID id{ neuron_id };
        /**
         * Determine synaptic input from neurons connected to me
         */

        // Walk through the local in-edges of my neuron
        const NetworkGraph::LocalEdges& local_in_edges = network_graph.get_local_in_edges(id);

        auto total_input = 0.0;
        for (const auto& [src_neuron_id, edge_val] : local_in_edges) {
            const auto spike = fired[src_neuron_id.get_neuron_id()];
            if (spike == FiredStatus::Fired) {
                total_input += k * edge_val;
            }
        }

        // Walk through the distant in-edges of my neuron
        const NetworkGraph::DistantEdges& in_edges = network_graph.get_distant_in_edges(id);

        for (const auto& [key, edge_val] : in_edges) {
            const auto& rank = key.get_rank();
            const auto& initiator_neuron_id = key.get_neuron_id();

            const auto contains_id = fired_status_comm->contains(rank, initiator_neuron_id);
            if (contains_id) {
                total_input += k * edge_val;
            }
        }

        synaptic_input[neuron_id] = total_input;
    }

    Timers::stop_and_add(TimerRegion::CALC_SYNAPTIC_INPUT);
}

void LinearSynapticInputCalculator::update_background_activity(const std::vector<UpdateStatus>& disable_flags) {
    Timers::start(TimerRegion::CALC_SYNAPTIC_BACKGROUND);

    const auto background_activity_stddev = get_background_activity_stddev();
    const auto background_activity_mean = get_background_activity_mean();
    const auto base_background_activity = get_base_background_activity();

    const auto number_local_neurons = get_number_neurons();

    auto& background_activity = get_inner_background_activity();

    // There might be background activity
    if (background_activity_stddev > 0.0) {
        for (size_t neuron_id = 0; neuron_id < number_local_neurons; ++neuron_id) {
            if (disable_flags[neuron_id] == UpdateStatus::Disabled) {
                continue;
            }

            const double rnd = RandomHolder::get_random_normal_double(RandomHolderKey::NeuronModel, background_activity_mean, background_activity_stddev);
            const double input = base_background_activity + rnd;
            background_activity[neuron_id] = input;
        }
    } else {
        std::ranges::fill(background_activity, base_background_activity);
    }

    Timers::stop_and_add(TimerRegion::CALC_SYNAPTIC_BACKGROUND);
}
