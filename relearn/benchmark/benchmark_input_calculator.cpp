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

#include "factory/extra_info_factory.h"
#include "factory/input_factory.h"
#include "factory/network_graph_factory.h"

#include <numeric>

static void BM_Linear_Input_No_Fired(benchmark::State& state) {
    const auto number_neurons = state.range(0);
    const auto number_synapses_per_neuron = state.range(1);

    auto network_graph_plastic = NetworkGraphFactory::construct_network_graph(number_neurons);
    auto synapses_plastic = NetworkGraphFactory::generate_local_synapses(number_neurons, number_synapses_per_neuron);
    NetworkGraphFactory::add_synapses(*network_graph_plastic, synapses_plastic);

    auto network_graph_static = NetworkGraphFactory::construct_network_graph(number_neurons);
    auto synapses_static = NetworkGraphFactory::generate_local_synapses(number_neurons, number_synapses_per_neuron);
    NetworkGraphFactory::add_synapses(*network_graph_static, synapses_static);

    std::vector<FiredStatus> fired_status(number_neurons, FiredStatus::Inactive);

    auto input_calculator = InputFactory::construct_linear_input();
    input_calculator->init(number_neurons);

    auto extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    extra_info->init(number_neurons);

    input_calculator->set_extra_infos(extra_info);

    NeuronsExtraInfoFactory::enable_all(extra_info);

    for (auto _ : state) {
        state.ResumeTiming();
        input_calculator->update_input(1000, *network_graph_static, *network_graph_plastic, fired_status);
        state.PauseTiming();

        auto sum = 0.0;
        const auto& values = input_calculator->get_synaptic_input();

        for (const auto val : values) {
            sum += val;
        }

        benchmark::DoNotOptimize(sum);
    }
}

static void BM_Linear_Input_All_Fired(benchmark::State& state) {
    const auto number_neurons = state.range(0);
    const auto number_synapses_per_neuron = state.range(1);

    auto network_graph_plastic = NetworkGraphFactory::construct_network_graph(number_neurons);
    auto synapses_plastic = NetworkGraphFactory::generate_local_synapses(number_neurons, number_synapses_per_neuron);
    NetworkGraphFactory::add_synapses(*network_graph_plastic, synapses_plastic);

    auto network_graph_static = NetworkGraphFactory::construct_network_graph(number_neurons);
    auto synapses_static = NetworkGraphFactory::generate_local_synapses(number_neurons, number_synapses_per_neuron);
    NetworkGraphFactory::add_synapses(*network_graph_static, synapses_static);

    std::vector<FiredStatus> fired_status(number_neurons, FiredStatus::Fired);

    auto input_calculator = InputFactory::construct_linear_input();
    input_calculator->init(number_neurons);

    auto extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    extra_info->init(number_neurons);

    input_calculator->set_extra_infos(extra_info);

    NeuronsExtraInfoFactory::enable_all(extra_info);

    for (auto _ : state) {
        state.ResumeTiming();
        input_calculator->update_input(1000, *network_graph_static, *network_graph_plastic, fired_status);
        state.PauseTiming();

        auto sum = 0.0;
        const auto& values = input_calculator->get_synaptic_input();

        for (const auto val : values) {
            sum += val;
        }

        benchmark::DoNotOptimize(sum);
    }
}

static void BM_Logarithmic_Input_No_Fired(benchmark::State& state) {
    const auto number_neurons = state.range(0);
    const auto number_synapses_per_neuron = state.range(1);

    auto network_graph_plastic = NetworkGraphFactory::construct_network_graph(number_neurons);
    auto synapses_plastic = NetworkGraphFactory::generate_local_synapses(number_neurons, number_synapses_per_neuron);
    NetworkGraphFactory::add_synapses(*network_graph_plastic, synapses_plastic);

    auto network_graph_static = NetworkGraphFactory::construct_network_graph(number_neurons);
    auto synapses_static = NetworkGraphFactory::generate_local_synapses(number_neurons, number_synapses_per_neuron);
    NetworkGraphFactory::add_synapses(*network_graph_static, synapses_static);

    std::vector<FiredStatus> fired_status(number_neurons, FiredStatus::Inactive);

    auto input_calculator = InputFactory::construct_logarithmic_input();
    input_calculator->init(number_neurons);

    auto extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    extra_info->init(number_neurons);

    input_calculator->set_extra_infos(extra_info);

    NeuronsExtraInfoFactory::enable_all(extra_info);

    for (auto _ : state) {
        state.ResumeTiming();
        input_calculator->update_input(1000, *network_graph_static, *network_graph_plastic, fired_status);
        state.PauseTiming();

        auto sum = 0.0;
        const auto& values = input_calculator->get_synaptic_input();

        for (const auto val : values) {
            sum += val;
        }

        benchmark::DoNotOptimize(sum);
    }
}

static void BM_Logarithmic_Input_All_Fired(benchmark::State& state) {
    const auto number_neurons = state.range(0);
    const auto number_synapses_per_neuron = state.range(1);

    auto network_graph_plastic = NetworkGraphFactory::construct_network_graph(number_neurons);
    auto synapses_plastic = NetworkGraphFactory::generate_local_synapses(number_neurons, number_synapses_per_neuron);
    NetworkGraphFactory::add_synapses(*network_graph_plastic, synapses_plastic);

    auto network_graph_static = NetworkGraphFactory::construct_network_graph(number_neurons);
    auto synapses_static = NetworkGraphFactory::generate_local_synapses(number_neurons, number_synapses_per_neuron);
    NetworkGraphFactory::add_synapses(*network_graph_static, synapses_static);

    std::vector<FiredStatus> fired_status(number_neurons, FiredStatus::Fired);

    auto input_calculator = InputFactory::construct_logarithmic_input();
    input_calculator->init(number_neurons);

    auto extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    extra_info->init(number_neurons);

    input_calculator->set_extra_infos(extra_info);

    NeuronsExtraInfoFactory::enable_all(extra_info);

    for (auto _ : state) {
        state.ResumeTiming();
        input_calculator->update_input(1000, *network_graph_static, *network_graph_plastic, fired_status);
        state.PauseTiming();

        auto sum = 0.0;
        const auto& values = input_calculator->get_synaptic_input();

        for (const auto val : values) {
            sum += val;
        }

        benchmark::DoNotOptimize(sum);
    }
}

static void BM_TanH_Input_No_Fired(benchmark::State& state) {
    const auto number_neurons = state.range(0);
    const auto number_synapses_per_neuron = state.range(1);

    auto network_graph_plastic = NetworkGraphFactory::construct_network_graph(number_neurons);
    auto synapses_plastic = NetworkGraphFactory::generate_local_synapses(number_neurons, number_synapses_per_neuron);
    NetworkGraphFactory::add_synapses(*network_graph_plastic, synapses_plastic);

    auto network_graph_static = NetworkGraphFactory::construct_network_graph(number_neurons);
    auto synapses_static = NetworkGraphFactory::generate_local_synapses(number_neurons, number_synapses_per_neuron);
    NetworkGraphFactory::add_synapses(*network_graph_static, synapses_static);

    std::vector<FiredStatus> fired_status(number_neurons, FiredStatus::Inactive);

    auto input_calculator = InputFactory::construct_tanh_input();
    input_calculator->init(number_neurons);

    auto extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    extra_info->init(number_neurons);

    input_calculator->set_extra_infos(extra_info);

    NeuronsExtraInfoFactory::enable_all(extra_info);

    for (auto _ : state) {
        state.ResumeTiming();
        input_calculator->update_input(1000, *network_graph_static, *network_graph_plastic, fired_status);
        state.PauseTiming();

        auto sum = 0.0;
        const auto& values = input_calculator->get_synaptic_input();

        for (const auto val : values) {
            sum += val;
        }

        benchmark::DoNotOptimize(sum);
    }
}

static void BM_TanH_Input_All_Fired(benchmark::State& state) {
    const auto number_neurons = state.range(0);
    const auto number_synapses_per_neuron = state.range(1);

    auto network_graph_plastic = NetworkGraphFactory::construct_network_graph(number_neurons);
    auto synapses_plastic = NetworkGraphFactory::generate_local_synapses(number_neurons, number_synapses_per_neuron);
    NetworkGraphFactory::add_synapses(*network_graph_plastic, synapses_plastic);

    auto network_graph_static = NetworkGraphFactory::construct_network_graph(number_neurons);
    auto synapses_static = NetworkGraphFactory::generate_local_synapses(number_neurons, number_synapses_per_neuron);
    NetworkGraphFactory::add_synapses(*network_graph_static, synapses_static);

    std::vector<FiredStatus> fired_status(number_neurons, FiredStatus::Fired);

    auto input_calculator = InputFactory::construct_tanh_input();
    input_calculator->init(number_neurons);

    auto extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    extra_info->init(number_neurons);

    input_calculator->set_extra_infos(extra_info);

    NeuronsExtraInfoFactory::enable_all(extra_info);

    for (auto _ : state) {
        state.ResumeTiming();
        input_calculator->update_input(1000, *network_graph_static, *network_graph_plastic, fired_status);
        state.PauseTiming();

        auto sum = 0.0;
        const auto& values = input_calculator->get_synaptic_input();

        for (const auto val : values) {
            sum += val;
        }

        benchmark::DoNotOptimize(sum);
    }
}

void CustomArgsInput(benchmark::internal::Benchmark* b) {
    b->Args({ 50000, static_number_synapses });
}

BENCHMARK(BM_Linear_Input_No_Fired)->Unit(benchmark::kMillisecond)->Apply(CustomArgsInput)->Iterations(static_number_iterations);
BENCHMARK(BM_Linear_Input_All_Fired)->Unit(benchmark::kMillisecond)->Apply(CustomArgsInput)->Iterations(static_number_iterations);

BENCHMARK(BM_Logarithmic_Input_No_Fired)->Unit(benchmark::kMillisecond)->Apply(CustomArgsInput)->Iterations(static_number_iterations);
BENCHMARK(BM_Logarithmic_Input_All_Fired)->Unit(benchmark::kMillisecond)->Apply(CustomArgsInput)->Iterations(static_number_iterations);

BENCHMARK(BM_TanH_Input_No_Fired)->Unit(benchmark::kMillisecond)->Apply(CustomArgsInput)->Iterations(static_number_iterations);
BENCHMARK(BM_TanH_Input_All_Fired)->Unit(benchmark::kMillisecond)->Apply(CustomArgsInput)->Iterations(static_number_iterations);
