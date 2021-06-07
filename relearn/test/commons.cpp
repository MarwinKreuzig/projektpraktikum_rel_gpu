#include "commons.h"

#include "../source/io/LogFiles.h"
#include "../source/mpi/MPIWrapper.h"
#include "../source/util/RelearnException.h"

#include <mutex>

std::mt19937 mt;
std::once_flag some_flag;

void setup() {

    auto lambda = []() {
        char* argument = (char*)"./runTests";
        MPIWrapper::init(1, &argument);
        MPIWrapper::init_buffer_octree(1);
        LogFiles::init();
    };

    std::call_once(some_flag, lambda);

    RelearnException::hide_messages = true;
}

NetworkGraph generate_random_network_graph(size_t num_neurons, size_t num_synapses, double threshold_exc) {
    std::uniform_int_distribution<size_t> uid(0, num_neurons - 1);
    std::uniform_real_distribution<double> urd(0, 1.0);

    NetworkGraph ng(num_neurons);

    for (size_t synapse_id = 0; synapse_id < num_synapses; synapse_id++) {
        const auto neuron_id_1 = uid(mt);
        auto neuron_id_2 = uid(mt);

        if (neuron_id_2 == neuron_id_1) {
            neuron_id_2 = (neuron_id_1 + 1) % num_neurons;
        }

        const auto uniform_double = urd(mt);
        const auto weight = (uniform_double < threshold_exc) ? 1 : -1;

        RankNeuronId target_id{ 0, neuron_id_1 };
        RankNeuronId source_id{ 0, neuron_id_2 };

        ng.add_edge_weight(target_id, source_id, weight);
    }

    return ng;
}

std::vector<size_t> generate_random_ids(size_t id_low, size_t id_high, size_t num_disables) {
    std::vector<size_t> disable_ids(num_disables);

    std::uniform_int_distribution<size_t> uid(id_low, id_high);

    for (size_t i = 0; i < num_disables; i++) {
        disable_ids[i] = uid(mt);
    }

    return disable_ids;
}
