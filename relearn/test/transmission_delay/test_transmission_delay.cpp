/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_transmission_delay.h"

#include "../harness/adapter/neurons/NeuronTypesAdapter.h"
#include "../harness/adapter/network_graph/NetworkGraphAdapter.h"
#include "../harness/adapter/neuron_id/NeuronIdAdapter.h"

#include "neurons/input/TransmissionDelayer.h"

static void sort_by_neuron_id(std::vector<std::pair<RankNeuronId, RelearnTypes::static_synapse_weight>>& vec) {
    std::sort(vec.begin(), vec.end(), [](auto& p1,auto& p2) {return p1.first.get_neuron_id().get_neuron_id() < p2.first.get_neuron_id().get_neuron_id();});
}


TEST_F(TransmissionDelayTest, testNoDelay) {
    const MPIRank my_rank = MPIRank{0};
    const auto num_neurons = NeuronIdAdapter::get_random_number_neurons(mt);

    const auto num_connections_per_vertex = RandomAdapter::get_random_integer(0,10, mt);
    const auto& network_graph = NetworkGraphAdapter::create_network_graph(num_neurons, my_rank, num_connections_per_vertex, mt);

    ConstantTransmissionDelayer delayer(0);

    const auto& fired_status = NeuronTypesAdapter::get_fired_status(num_neurons, mt);

    delayer.prepare_update(num_neurons);
    for(auto neuron_id=0;neuron_id<num_neurons;neuron_id++) {
        const RankNeuronId source(MPIRank{0},NeuronID{neuron_id});
        if(fired_status[neuron_id] == FiredStatus::Fired) {
            const auto& out_edges =  NetworkGraphAdapter::get_all_out_edges(*network_graph, my_rank, NeuronID {neuron_id});
            for(const auto& [rni, weight] : out_edges) {
                delayer.register_fired_input(rni.get_neuron_id(), source, weight, num_neurons);
            }
        }
    }

    for(auto neuron_id=0;neuron_id<num_neurons;neuron_id++) {
        std::vector<std::pair<RankNeuronId, RelearnTypes::static_synapse_weight>> expected_fired_inputs;
            const auto& in_edges = NetworkGraphAdapter::get_all_in_edges(*network_graph, my_rank, NeuronID {neuron_id});
            for(const auto& [rni, weight] : in_edges) {
                if(fired_status[rni.get_neuron_id().get_neuron_id()] == FiredStatus::Fired) {
                    expected_fired_inputs.emplace_back(rni, weight);
                }
            }

            const auto& actual_fired_inputs = delayer.get_delayed_inputs(NeuronID(neuron_id));
            ASSERT_EQ(actual_fired_inputs.size(), expected_fired_inputs.size());
            std::vector<std::pair<RankNeuronId, RelearnTypes::static_synapse_weight>> sorted_actual_fired_inputs = actual_fired_inputs;
        sort_by_neuron_id(sorted_actual_fired_inputs);
        sort_by_neuron_id(expected_fired_inputs);
            ASSERT_EQ(expected_fired_inputs, sorted_actual_fired_inputs);
    }
}



TEST_F(TransmissionDelayTest, testConstantDelay) {
    const MPIRank my_rank = MPIRank{0};
    const auto num_neurons = NeuronIdAdapter::get_random_number_neurons(mt);
    const auto delay = RandomAdapter::get_random_integer(10,50, mt);
    const auto num_steps = delay * 5;

    const auto num_connections_per_vertex = RandomAdapter::get_random_integer(0,10, mt);
    const auto& network_graph = NetworkGraphAdapter::create_network_graph(num_neurons, my_rank, num_connections_per_vertex, mt);

    ConstantTransmissionDelayer delayer(delay);

    std::vector<std::vector<FiredStatus>> fired_status_vector;

    for(auto i=0;i<num_steps;i++) {
        const auto &fired_status = NeuronTypesAdapter::get_fired_status(num_neurons, mt);
        fired_status_vector.push_back(fired_status);
    }

    for(auto i=0;i<num_steps;i++) {
        //Update delayer
        delayer.prepare_update(num_neurons);
        for(auto neuron_id=0;neuron_id<num_neurons;neuron_id++) {
            const RankNeuronId source(MPIRank{0},NeuronID{neuron_id});
            if(fired_status_vector[i][neuron_id] == FiredStatus::Fired) {
                const auto& out_edges = NetworkGraphAdapter::get_all_out_edges(*network_graph, my_rank, NeuronID {neuron_id});
                for(const auto& [rni, weight] : out_edges) {
                    delayer.register_fired_input(rni.get_neuron_id(), source, weight, num_neurons);
                }
            }
        }

        //Check delayer
        for(auto neuron_id=0;neuron_id<num_neurons;neuron_id++) {
            std::vector<std::pair<RankNeuronId, RelearnTypes::static_synapse_weight>> expected_fired_inputs;
            const auto& in_edges = NetworkGraphAdapter::get_all_in_edges(*network_graph, my_rank, NeuronID {neuron_id});
            if(i>=delay) {
                for (const auto &[rni, weight]: in_edges) {
                    if (fired_status_vector[i - delay][rni.get_neuron_id().get_neuron_id()] == FiredStatus::Fired) {
                        expected_fired_inputs.emplace_back(rni, weight);
                    }
                }
            }

            const auto& actual_fired_inputs = delayer.get_delayed_inputs(NeuronID(neuron_id));
            ASSERT_EQ(actual_fired_inputs.size(), expected_fired_inputs.size());
            std::vector<std::pair<RankNeuronId, RelearnTypes::static_synapse_weight>> sorted_actual_fired_inputs = actual_fired_inputs;
            sort_by_neuron_id(sorted_actual_fired_inputs);
            sort_by_neuron_id(expected_fired_inputs);
            ASSERT_EQ(expected_fired_inputs, sorted_actual_fired_inputs);
        }
    }

}