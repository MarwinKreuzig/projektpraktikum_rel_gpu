#pragma once

#include <vector>

#include "benchmark/benchmark.h"

#include "Types.h"
#include "neurons/NetworkGraph.h"


constexpr bool excessive_testing = false;

LocalSynapses generate_local_synapses(int number_neurons, int number_synapses);

DistantInSynapses generate_distant_in_synapses(int number_neurons, int number_synapses);

DistantOutSynapses generate_distant_out_synapses(int number_neurons, int number_synapses);

template <typename SynapseType>
std::vector<SynapseType> invert_synapses(const std::vector<SynapseType>& synapses) {
    std::vector<SynapseType> inverted_synapses{};
    inverted_synapses.reserve(synapses.size());

    for (const auto& [target, source, weight] : synapses) {
        inverted_synapses.emplace_back(target, source, -weight);
    }

    return inverted_synapses;
}

template <typename SynapseType>
void add_synapses(NetworkGraph& ng, const std::vector<SynapseType>& synapses) {
    for (const auto& synapse : synapses) {
        ng.add_synapse(synapse);
    }
}
