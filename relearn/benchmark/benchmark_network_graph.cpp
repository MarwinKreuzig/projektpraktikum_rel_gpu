/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "main.h"

#include "factory/network_graph_factory.h"

static void BM_NetworkGraph_InsertLocal(benchmark::State& state) {
    const auto number_neurons = state.range(0);
    const auto number_synapses_per_neuron = state.range(1);

    for (auto _ : state) {
        state.PauseTiming();

        NetworkGraph ng(MPIRank::root_rank());
        ng.init(number_neurons);
        const auto& synapses = NetworkGraphFactory::generate_local_synapses(number_neurons, number_synapses_per_neuron);

        state.ResumeTiming();

        NetworkGraphFactory::add_synapses(ng, synapses);
    }
}

static void BM_NetworkGraph_RemoveLocal(benchmark::State& state) {
    const auto number_neurons = state.range(0);
    const auto number_synapses_per_neuron = state.range(1);

    for (auto _ : state) {
        state.PauseTiming();

        NetworkGraph ng(MPIRank::root_rank());
        ng.init(number_neurons);

        const auto& synapses = NetworkGraphFactory::generate_local_synapses(number_neurons, number_synapses_per_neuron);
        NetworkGraphFactory::add_synapses(ng, synapses);

        const auto& inverted_synapses = NetworkGraphFactory::invert_synapses(synapses);

        state.ResumeTiming();

        NetworkGraphFactory::add_synapses(ng, inverted_synapses);
    }
}

static void BM_NetworkGraph_InsertDistantIn(benchmark::State& state) {
    const auto number_neurons = state.range(0);
    const auto number_synapses_per_neuron = state.range(1);

    for (auto _ : state) {
        state.PauseTiming();

        NetworkGraph ng(MPIRank::root_rank());
        ng.init(number_neurons);

        const auto& synapses = NetworkGraphFactory::generate_distant_in_synapses(number_neurons, number_synapses_per_neuron);

        state.ResumeTiming();

        NetworkGraphFactory::add_synapses(ng, synapses);
    }
}

static void BM_NetworkGraph_RemoveDistantIn(benchmark::State& state) {
    const auto number_neurons = state.range(0);
    const auto number_synapses_per_neuron = state.range(1);

    for (auto _ : state) {
        state.PauseTiming();

        NetworkGraph ng(MPIRank::root_rank());
        ng.init(number_neurons);

        const auto& synapses = NetworkGraphFactory::generate_distant_in_synapses(number_neurons, number_synapses_per_neuron);
        NetworkGraphFactory::add_synapses(ng, synapses);

        const auto& inverted_synapses = NetworkGraphFactory::invert_synapses(synapses);

        state.ResumeTiming();

        NetworkGraphFactory::add_synapses(ng, inverted_synapses);
    }
}

static void BM_NetworkGraph_InsertDistantOut(benchmark::State& state) {
    const auto number_neurons = state.range(0);
    const auto number_synapses_per_neuron = state.range(1);

    for (auto _ : state) {
        state.PauseTiming();

        NetworkGraph ng(MPIRank::root_rank());
        ng.init(number_neurons);

        const auto& synapses = NetworkGraphFactory::generate_distant_out_synapses(number_neurons, number_synapses_per_neuron);

        state.ResumeTiming();

        NetworkGraphFactory::add_synapses(ng, synapses);
    }
}

static void BM_NetworkGraph_RemoveDistantOut(benchmark::State& state) {
    const auto number_neurons = state.range(0);
    const auto number_synapses_per_neuron = state.range(1);

    for (auto _ : state) {
        state.PauseTiming();

        NetworkGraph ng(MPIRank::root_rank());
        ng.init(number_neurons);

        const auto& synapses = NetworkGraphFactory::generate_distant_out_synapses(number_neurons, number_synapses_per_neuron);
        NetworkGraphFactory::add_synapses(ng, synapses);

        const auto& inverted_synapses = NetworkGraphFactory::invert_synapses(synapses);

        state.ResumeTiming();

        NetworkGraphFactory::add_synapses(ng, inverted_synapses);
    }
}

void CustomArgsNetwork(benchmark::internal::Benchmark* b) {
    if constexpr (excessive_testing) {
        const auto neuron_sizes = { 1000,
            2000,
            5000,
            10000 };

        const auto synapse_sizes = {
            10, 20, 50, 100
        };

        for (const auto neuron_size : neuron_sizes) {
            for (const auto synapse_size : synapse_sizes) {
                b->Args({ neuron_size, synapse_size });
            }
        }
    } else {
        b->Args({ 5000, 20 });
    }
}

BENCHMARK(BM_NetworkGraph_InsertLocal)->Unit(benchmark::kMillisecond)->Apply(CustomArgsNetwork)->Iterations(static_few_iterations);
BENCHMARK(BM_NetworkGraph_RemoveLocal)->Unit(benchmark::kMillisecond)->Apply(CustomArgsNetwork)->Iterations(static_few_iterations);

BENCHMARK(BM_NetworkGraph_InsertDistantIn)->Unit(benchmark::kMillisecond)->Apply(CustomArgsNetwork)->Iterations(static_few_iterations);
BENCHMARK(BM_NetworkGraph_RemoveDistantIn)->Unit(benchmark::kMillisecond)->Apply(CustomArgsNetwork)->Iterations(static_few_iterations);

BENCHMARK(BM_NetworkGraph_InsertDistantOut)->Unit(benchmark::kMillisecond)->Apply(CustomArgsNetwork)->Iterations(static_few_iterations);
BENCHMARK(BM_NetworkGraph_RemoveDistantOut)->Unit(benchmark::kMillisecond)->Apply(CustomArgsNetwork)->Iterations(static_few_iterations);
