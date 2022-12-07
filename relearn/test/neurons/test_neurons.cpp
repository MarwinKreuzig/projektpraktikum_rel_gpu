#include "test_neurons.h"

#include "neurons/models/NeuronModels.h"
#include "neurons/models/SynapticElements.h"
#include "neurons/CalciumCalculator.h"
#include "neurons/Neurons.h"
#include "structure/Partition.h"

#include <vector>

TEST_F(NeuronsTest, testNeuronsConstructor) {
    auto partition = std::make_shared<Partition>(1, 0);

    auto model = std::make_unique<models::PoissonModel>();
    auto calcium = std::make_unique<CalciumCalculator>();
    auto dends_ex = std::make_unique<SynapticElements>(ElementType::Dendrite, 0.2);
    auto dends_in = std::make_unique<SynapticElements>(ElementType::Dendrite, 0.2);
    auto axs = std::make_unique<SynapticElements>(ElementType::Axon, 0.2);

    Neurons neurons{ partition, std::move(model), std::move(calcium), std::move(axs), std::move(dends_ex), std::move(dends_in) };
}

TEST_F(NeuronsTest, testSignalTypeCheck) {
    const auto num_test_synapses = get_random_integer(50, 500);
    const auto num_neurons = get_random_number_neurons() + 10;
    const auto num_synapses = get_random_integer(10, 100);
    const auto num_ranks = get_random_number_ranks();
    std::vector<SignalType> signal_types{};
    const auto network_graph = std::make_shared<NetworkGraph>(num_neurons, 0);

    for (const auto& neuron_id : NeuronID::range(num_neurons)) {
        const auto signal_type = get_random_signal_type();
        signal_types.emplace_back(signal_type);
    }

    for (int synapse_nr = 0; synapse_nr < num_synapses; synapse_nr++) {
        auto weight = get_random_double(0.1, 20.0);
        const auto src = get_random_neuron_id(num_neurons);
        const auto tgt = get_random_neuron_id(num_neurons, src);
        const auto tgt_rank = get_random_rank(num_ranks);
        if (signal_types[src.get_neuron_id()] == SignalType::Inhibitory) {
            weight = -weight;
        }
        if (tgt_rank == 0) {
            network_graph->add_synapse({ tgt, src, weight });
        } else {
            network_graph->add_synapse({ RankNeuronId(tgt_rank, tgt), src, weight });
        }
    }

    ASSERT_NO_THROW(Neurons::check_signal_types(network_graph, signal_types, 0));

    for (int test_synapse = 0; test_synapse < num_test_synapses; test_synapse++) {
        const auto src = get_random_neuron_id(num_neurons);
        const auto tgt_rank = get_random_rank(num_ranks);
        const auto tgt = get_random_neuron_id(num_neurons, src);
        const RankNeuronId tgt_rni(tgt_rank, tgt);
        auto weight = get_random_double(0.1, 20.0);
        if (signal_types[src.get_neuron_id()] == SignalType::Excitatory) {
            weight = -weight;
        }
        const auto& out_egdes = network_graph->get_all_out_edges(src);
        for (const auto& [out_tgt_rni, out_weight] : out_egdes) {
            if (tgt_rni == out_tgt_rni) {
                weight = weight - out_weight;
            }
        }
        if (tgt_rank == 0) {
            network_graph->add_synapse({ tgt, src, weight });
        } else {
            network_graph->add_synapse({ RankNeuronId(tgt_rank, tgt), src, weight });
        }
        ASSERT_THROW(Neurons::check_signal_types(network_graph, signal_types, 0), RelearnException);
        if (tgt_rank == 0) {
            network_graph->add_synapse({ tgt, src, -weight });
        } else {
            network_graph->add_synapse({ RankNeuronId(tgt_rank, tgt), src, -weight });
        }
        ASSERT_NO_THROW(Neurons::check_signal_types(network_graph, signal_types, 0));
    }
}