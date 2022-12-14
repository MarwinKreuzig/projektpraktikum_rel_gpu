#include "test_neurons.h"

#include "RandomAdapter.h"
#include "mpi/mpi_rank_adapter.h"
#include "neurons/neuron_types_adapter.h"
#include "tagged_id/tagged_id_adapter.h"

#include "neurons/models/BackgroundActivityCalculators.h"
#include "neurons/models/NeuronModels.h"
#include "neurons/models/SynapticElements.h"
#include "neurons/models/SynapticInputCalculator.h"
#include "neurons/models/SynapticInputCalculators.h"
#include "neurons/CalciumCalculator.h"
#include "neurons/Neurons.h"
#include "structure/Partition.h"

#include <vector>

TEST_F(NeuronsTest, testNeuronsConstructor) {
    auto partition = std::make_shared<Partition>(1, MPIRank::root_rank());

    auto model = std::make_unique<models::PoissonModel>();
    auto calcium = std::make_unique<CalciumCalculator>();
    auto dends_ex = std::make_unique<SynapticElements>(ElementType::Dendrite, 0.2);
    auto dends_in = std::make_unique<SynapticElements>(ElementType::Dendrite, 0.2);
    auto axs = std::make_unique<SynapticElements>(ElementType::Axon, 0.2);

    Neurons neurons{ partition, std::move(model), std::move(calcium), std::move(axs), std::move(dends_ex), std::move(dends_in) };
}

TEST_F(NeuronsTest, testSignalTypeCheck) {
    const auto num_test_synapses = RandomAdapter::get_random_integer(50, 500, mt);
    const auto num_neurons = TaggedIdAdapter::get_random_number_neurons(mt) + 10;
    const auto num_synapses = RandomAdapter::get_random_integer(10, 100, mt);
    const auto num_ranks = MPIRankAdapter::get_random_number_ranks(mt);
    std::vector<SignalType> signal_types{};
    const auto network_graph = std::make_shared<NetworkGraph>(num_neurons, MPIRank::root_rank());

    for (const auto& neuron_id : NeuronID::range(num_neurons)) {
        const auto signal_type = NeuronTypesAdapter::get_random_signal_type(mt);
        signal_types.emplace_back(signal_type);
    }

    for (int synapse_nr = 0; synapse_nr < num_synapses; synapse_nr++) {
        auto weight = RandomAdapter::get_random_double(0.1, 20.0, mt);
        const auto src = TaggedIdAdapter::get_random_neuron_id(num_neurons, mt);
        const auto tgt = TaggedIdAdapter::get_random_neuron_id(num_neurons, src, mt);
        const auto tgt_rank = MPIRankAdapter::get_random_mpi_rank(num_ranks, mt);
        if (signal_types[src.get_neuron_id()] == SignalType::Inhibitory) {
            weight = -weight;
        }
        if (tgt_rank == MPIRank::root_rank()) {
            network_graph->add_synapse({ tgt, src, weight });
        } else {
            network_graph->add_synapse({ RankNeuronId(tgt_rank, tgt), src, weight });
        }
    }

    ASSERT_NO_THROW(Neurons::check_signal_types(network_graph, signal_types, MPIRank::root_rank()));

    for (int test_synapse = 0; test_synapse < num_test_synapses; test_synapse++) {
        const auto src = TaggedIdAdapter::get_random_neuron_id(num_neurons, mt);
        const auto tgt_rank = MPIRankAdapter::get_random_mpi_rank(num_ranks, mt);
        const auto tgt = TaggedIdAdapter::get_random_neuron_id(num_neurons, src, mt);
        const RankNeuronId tgt_rni(tgt_rank, tgt);
        auto weight = RandomAdapter::get_random_double(0.1, 20.0, mt);
        if (signal_types[src.get_neuron_id()] == SignalType::Excitatory) {
            weight = -weight;
        }
        const auto& out_egdes = network_graph->get_all_out_edges(src);
        for (const auto& [out_tgt_rni, out_weight] : out_egdes) {
            if (tgt_rni == out_tgt_rni) {
                weight = weight - out_weight;
            }
        }
        if (tgt_rank == MPIRank::root_rank()) {
            network_graph->add_synapse({ tgt, src, weight });
        } else {
            network_graph->add_synapse({ RankNeuronId(tgt_rank, tgt), src, weight });
        }
        ASSERT_THROW(Neurons::check_signal_types(network_graph, signal_types, MPIRank::root_rank()), RelearnException);
        if (tgt_rank == MPIRank::root_rank()) {
            network_graph->add_synapse({ tgt, src, -weight });
        } else {
            network_graph->add_synapse({ RankNeuronId(tgt_rank, tgt), src, -weight });
        }
        ASSERT_NO_THROW(Neurons::check_signal_types(network_graph, signal_types, MPIRank::root_rank()));
    }
}

TEST_F(NeuronsTest, testStaticConnectionsChecker) {
    auto num_neurons = TaggedIdAdapter::get_random_number_neurons(mt) + 30;
    auto num_static_neurons = RandomAdapter::get_random_integer(15, static_cast<int>(num_neurons) - 10, mt);

    std::vector<NeuronID> static_neurons{};
    for (auto i = 0; i < num_static_neurons; i++) {
        NeuronID static_neuron;
        do {
            static_neuron = TaggedIdAdapter::get_random_neuron_id(num_neurons, mt);
        } while (std::find(static_neurons.begin(), static_neurons.end(), static_neuron) != static_neurons.end());
        static_neurons.emplace_back(static_neuron);
    }

    auto partition = std::make_shared<Partition>(1, MPIRank::root_rank());
    auto model = std::make_unique<models::PoissonModel>(models::PoissonModel::default_h,
        std::make_unique<LinearSynapticInputCalculator>(SynapticInputCalculator::default_conductance),
        std::make_unique<NullBackgroundActivityCalculator>(),
        models::PoissonModel::default_x_0,
        models::PoissonModel::default_tau_x,
        models::PoissonModel::default_refrac_time);
    auto calcium = std::make_unique<CalciumCalculator>();
    calcium->set_initial_calcium_calculator([](MPIRank /*mpi_rank*/, NeuronID::value_type /*neuron_id*/) { return 0.0; });
    calcium->set_target_calcium_calculator([](MPIRank /*mpi_rank*/, NeuronID::value_type /*neuron_id*/) { return 0.0; });
    auto dends_ex = std::make_unique<SynapticElements>(ElementType::Dendrite, 0.2);
    auto dends_in = std::make_unique<SynapticElements>(ElementType::Dendrite, 0.2);
    auto axs = std::make_unique<SynapticElements>(ElementType::Axon, 0.2);

    Neurons neurons{ partition, std::move(model), std::move(calcium), std::move(axs), std::move(dends_ex), std::move(dends_in) };

    auto network_graph_static = std::make_shared<NetworkGraph>(num_neurons, MPIRank::root_rank());
    auto network_graph_plastic = std::make_shared<NetworkGraph>(num_neurons, MPIRank::root_rank());

    auto num_synapses_static = RandomAdapter::get_random_integer(30, 100, mt);
    auto num_synapses_plastic = RandomAdapter::get_random_integer(30, 100, mt);

    for (auto i = 0; i < num_synapses_static; i++) {
        auto src = TaggedIdAdapter::get_random_neuron_id(num_neurons, mt);
        auto tgt = TaggedIdAdapter::get_random_neuron_id(num_neurons, NeuronID(src), mt);
        auto weight = RandomAdapter::get_random_double<double>(0.1, 10, mt);
        network_graph_static->add_synapse({ tgt, src, weight });
    }

    for (auto i = 0; i < num_synapses_plastic; i++) {
        NeuronID src, tgt;
        do {
            src = TaggedIdAdapter::get_random_neuron_id(num_neurons, mt);
            tgt = TaggedIdAdapter::get_random_neuron_id(num_neurons, mt);
        } while (std::find(static_neurons.begin(), static_neurons.end(), src) != static_neurons.end() || std::find(static_neurons.begin(), static_neurons.end(), tgt) != static_neurons.end()
            || src == tgt);
        auto weight = RandomAdapter::get_random_double<double>(0.1, 10, mt);
        network_graph_plastic->add_synapse({ NeuronID{ tgt }, NeuronID{ src }, weight });
    }

    neurons.init(num_neurons);
    neurons.set_network_graph(network_graph_static, network_graph_plastic);

    neurons.set_static_neurons(static_neurons);

    const auto num_tries = RandomAdapter::get_random_integer(10, 100, mt);
    for (auto i = 0; i < num_tries; i++) {
        const bool src_is_static = RandomAdapter::get_random_bool(mt);
        const bool tgt_is_static = !src_is_static || RandomAdapter::get_random_bool(mt);

        NeuronID src, tgt;
        if (src_is_static) {
            src = static_neurons[RandomAdapter::get_random_integer<int>(0, static_neurons.size() - 1, mt)];
        } else {
            do {
                src = TaggedIdAdapter::get_random_neuron_id(num_neurons, mt);
            } while (std::find(static_neurons.begin(), static_neurons.end(), src) != static_neurons.end());
        }
        if (tgt_is_static) {
            tgt = static_neurons[RandomAdapter::get_random_integer<int>(0, static_neurons.size() - 1, mt)];
        } else {
            do {
                tgt = TaggedIdAdapter::get_random_neuron_id(num_neurons, mt);
            } while (std::find(static_neurons.begin(), static_neurons.end(), tgt) != static_neurons.end());
        }
        if (tgt == src)
            continue;
        double weight = RandomAdapter::get_random_double<double>(0.1, 100.0, mt);
        if (RandomAdapter::get_random_bool(mt)) {
            weight = -weight;
        }

        network_graph_plastic->add_synapse({ tgt, src, weight });

        ASSERT_THROW(neurons.set_static_neurons(static_neurons), RelearnException);
        network_graph_plastic->add_synapse({ tgt, src, -weight });
        neurons.set_static_neurons(static_neurons);
    }
}