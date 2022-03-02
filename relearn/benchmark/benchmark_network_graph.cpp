#include "main.h"

constexpr auto number_iterations = 1000;

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

static void BM_NetworkGraph_InsertDistantIn(benchmark::State& state) {
    const auto number_neurons = state.range(0);
    const auto number_synapses = state.range(1);

    for (auto _ : state) {
        state.PauseTiming();

        NetworkGraph ng(number_neurons, 0);
        const auto& synapses = generate_distant_in_synapses(number_neurons, number_synapses);

        state.ResumeTiming();

        add_synapses(ng, synapses);
    }
}

static void BM_NetworkGraph_RemoveDistantIn(benchmark::State& state) {
    const auto number_neurons = state.range(0);
    const auto number_synapses = state.range(1);

    for (auto _ : state) {
        state.PauseTiming();

        NetworkGraph ng(number_neurons, 0);

        const auto& synapses = generate_distant_in_synapses(number_neurons, number_synapses);
        add_synapses(ng, synapses);

        const auto& inverted_synapses = invert_synapses(synapses);

        state.ResumeTiming();

        add_synapses(ng, inverted_synapses);
    }
}

static void BM_NetworkGraph_InsertDistantOut(benchmark::State& state) {
    const auto number_neurons = state.range(0);
    const auto number_synapses = state.range(1);

    for (auto _ : state) {
        state.PauseTiming();

        NetworkGraph ng(number_neurons, 0);
        const auto& synapses = generate_distant_out_synapses(number_neurons, number_synapses);

        state.ResumeTiming();

        add_synapses(ng, synapses);
    }
}

static void BM_NetworkGraph_RemoveDistantOut(benchmark::State& state) {
    const auto number_neurons = state.range(0);
    const auto number_synapses = state.range(1);

    for (auto _ : state) {
        state.PauseTiming();

        NetworkGraph ng(number_neurons, 0);

        const auto& synapses = generate_distant_out_synapses(number_neurons, number_synapses);
        add_synapses(ng, synapses);

        const auto& inverted_synapses = invert_synapses(synapses);

        state.ResumeTiming();

        add_synapses(ng, inverted_synapses);
    }
}

void CustomArgs(benchmark::internal::Benchmark* b) {
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

BENCHMARK(BM_NetworkGraph_InsertLocal)->Unit(benchmark::kMillisecond)->Apply(CustomArgs)->Iterations(number_iterations);
BENCHMARK(BM_NetworkGraph_RemoveLocal)->Unit(benchmark::kMillisecond)->Apply(CustomArgs)->Iterations(number_iterations);

BENCHMARK(BM_NetworkGraph_InsertDistantIn)->Unit(benchmark::kMillisecond)->Apply(CustomArgs)->Iterations(number_iterations);
BENCHMARK(BM_NetworkGraph_RemoveDistantIn)->Unit(benchmark::kMillisecond)->Apply(CustomArgs)->Iterations(number_iterations);

BENCHMARK(BM_NetworkGraph_InsertDistantOut)->Unit(benchmark::kMillisecond)->Apply(CustomArgs)->Iterations(number_iterations);
BENCHMARK(BM_NetworkGraph_RemoveDistantOut)->Unit(benchmark::kMillisecond)->Apply(CustomArgs)->Iterations(number_iterations);
