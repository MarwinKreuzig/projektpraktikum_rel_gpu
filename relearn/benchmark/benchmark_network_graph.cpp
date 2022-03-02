#include "benchmark/benchmark.h"

#include "neurons/NetworkGraph.h"

#include <iostream>
#include <random>
#include <utility>

std::vector<LocalSynapse> generate_local_synapses(int number_neurons, int number_synapses) {
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

std::vector<LocalSynapse> invert_synapses(const std::vector<LocalSynapse>& synapses) {
    std::vector<LocalSynapse> inverted_synapses{};
    inverted_synapses.reserve(synapses.size());

    for (const auto& [target, source, weight] : synapses) {
        inverted_synapses.emplace_back(target, source, -weight);
    }

    return inverted_synapses;
}

void add_synapses(NetworkGraph& ng, const std::vector<LocalSynapse>& synapses) {
    for (const auto& synapse : synapses) {
        ng.add_synapse(synapse);
    }
}

static void BM_NetworkGraph_InsertLocal(benchmark::State& state) {
    const auto number_neurons = state.range(0);
    const auto number_synapses = state.range(1);

    for (auto _ : state) {
        state.PauseTiming();
        
        NetworkGraph ng(number_neurons, 0);
        const auto& synapses = generate_local_synapses(number_neurons, number_synapses);

        state.ResumeTiming();

        add_synapses(ng, synapses);
    }
}

static void BM_NetworkGraph_RemoveLocal(benchmark::State& state) {
    const auto number_neurons = state.range(0);
    const auto number_synapses = state.range(1);

    for (auto _ : state) {
        state.PauseTiming();

        NetworkGraph ng(number_neurons, 0);
        
        const auto& synapses = generate_local_synapses(number_neurons, number_synapses);
        add_synapses(ng, synapses);

        const auto& inverted_synapses = invert_synapses(synapses);

        state.ResumeTiming();

        add_synapses(ng, inverted_synapses);
    }
}

BENCHMARK(BM_NetworkGraph_InsertLocal)->Unit(benchmark::kMillisecond)->Args({ 1000, 10 })->Args({ 1000, 20 })->Args({ 2000, 10 })->Args({ 2000, 20 });
BENCHMARK(BM_NetworkGraph_RemoveLocal)->Unit(benchmark::kMillisecond)->Args({ 1000, 10 })->Args({ 1000, 20 })->Args({ 2000, 10 })->Args({ 2000, 20 });

BENCHMARK_MAIN();
