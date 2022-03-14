#include "main.h"

#include <random>

LocalSynapses generate_local_synapses(int number_neurons, int number_synapses) {
    std::vector<LocalSynapse> synapses{};
    synapses.reserve(number_neurons * number_synapses);

    std::mt19937 mt{};
    std::uniform_int_distribution<int> uid(0, number_neurons - 1);

    for (auto neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
        for (auto synapse_id = 0; synapse_id < number_synapses; synapse_id++) {
            auto random_id = uid(mt);

            const NeuronID source_id{ neuron_id };
            const NeuronID target_id{ random_id };

            const auto weight = 1;

            synapses.emplace_back(target_id, source_id, weight);
        }
    }

    return synapses;
}

DistantInSynapses generate_distant_in_synapses(int number_neurons, int number_synapses) {
    std::vector<DistantInSynapse> synapses{};
    synapses.reserve(number_neurons * number_synapses);

    std::mt19937 mt{};
    std::uniform_int_distribution<int> uid(0, number_neurons - 1);
    std::uniform_int_distribution<int> uid_rank(1, 32);

    for (auto neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
        for (auto synapse_id = 0; synapse_id < number_synapses; synapse_id++) {
            auto random_id = uid(mt);
            auto random_rank = uid_rank(mt);

            const NeuronID source_id{ random_id };
            const NeuronID target_id{ neuron_id };

            const RankNeuronId rni{ random_rank, source_id };

            const auto weight = 1;

            synapses.emplace_back(target_id, rni, weight);
        }
    }

    return synapses;
}

DistantOutSynapses generate_distant_out_synapses(int number_neurons, int number_synapses) {
    std::vector<DistantOutSynapse> synapses{};
    synapses.reserve(number_neurons * number_synapses);

    std::mt19937 mt{};
    std::uniform_int_distribution<int> uid(0, number_neurons - 1);
    std::uniform_int_distribution<int> uid_rank(1, 32);

    for (auto neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
        for (auto synapse_id = 0; synapse_id < number_synapses; synapse_id++) {
            auto random_id = uid(mt);
            auto random_rank = uid_rank(mt);

            const NeuronID source_id{ neuron_id };
            const NeuronID target_id{ random_id };

            const RankNeuronId rni{ random_rank, target_id };

            const auto weight = 1;

            synapses.emplace_back(rni, source_id, weight);
        }
    }

    return synapses;
}

BENCHMARK_MAIN();
