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

#include "AdapterNeuronModel.h"

#include "factory/background_factory.h"
#include "factory/extra_info_factory.h"
#include "factory/input_factory.h"
#include "factory/neuron_model_factory.h"

#include <numeric>

template <typename NeuronModelType>
static void BM_NeuronModel_UpdateActivityBenchmark(benchmark::State& state) {
    const auto number_neurons = state.range(0);

    auto synaptic_input = InputFactory::construct_linear_input();
    auto background = BackgroundFactory::construct_null_background();

    auto extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    extra_info->init(number_neurons);

    auto model = NeuronModelFactory::construct_model<NeuronModelType>(10, std::move(synaptic_input), std::move(background), std::make_unique<Stimulus>());
    model->init(number_neurons);
    model->set_extra_infos(extra_info);

    NeuronsExtraInfoFactory::enable_all(extra_info);

    AdapterNeuronModel<NeuronModelType> adapter{ *model };

    for (auto _ : state) {
        state.ResumeTiming();

        for (const auto& neuron_id : NeuronID::range(number_neurons)) {
            adapter.update_activity_benchmark(neuron_id);
        }

        state.PauseTiming();

        auto sum = 0.0;
        const auto& x = adapter.get_x();

        for (const auto val : x) {
            sum += val;
        }

        benchmark::DoNotOptimize(sum);
    }
}

template <typename NeuronModelType>
static void BM_NeuronModel_UpdateFullActivity(benchmark::State& state) {
    const auto number_neurons = state.range(0);

    auto synaptic_input = InputFactory::construct_linear_input();
    auto background = BackgroundFactory::construct_null_background();

    auto extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    extra_info->init(number_neurons);

    auto model = NeuronModelFactory::construct_model<NeuronModelType>(10, std::move(synaptic_input), std::move(background), std::make_unique<Stimulus>());
    model->init(number_neurons);
    model->set_extra_infos(extra_info);

    NeuronsExtraInfoFactory::enable_all(extra_info);

    AdapterNeuronModel<NeuronModelType> adapter{ *model };

    for (auto _ : state) {
        state.ResumeTiming();
        adapter.update_activity();
        state.PauseTiming();

        auto sum = 0.0;
        const auto& x = adapter.get_x();

        for (const auto val : x) {
            sum += val;
        }

        benchmark::DoNotOptimize(sum);
    }
}

template <typename NeuronModelType>
static void BM_NeuronModel_UpdateFullActivityBenchmark(benchmark::State& state) {
    const auto number_neurons = state.range(0);

    auto synaptic_input = InputFactory::construct_linear_input();
    auto background = BackgroundFactory::construct_null_background();

    auto extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    extra_info->init(number_neurons);

    auto model = NeuronModelFactory::construct_model<NeuronModelType>(10, std::move(synaptic_input), std::move(background), std::make_unique<Stimulus>());
    model->init(number_neurons);
    model->set_extra_infos(extra_info);

    NeuronsExtraInfoFactory::enable_all(extra_info);

    AdapterNeuronModel<NeuronModelType> adapter{ *model };

    for (auto _ : state) {
        state.ResumeTiming();
        adapter.update_activity_benchmark();
        state.PauseTiming();

        auto sum = 0.0;
        const auto& x = adapter.get_x();

        for (const auto val : x) {
            sum += val;
        }

        benchmark::DoNotOptimize(sum);
    }
}

// BENCHMARK(BM_NeuronModel_UpdateActivityBenchmark<models::PoissonModel>)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons)->Iterations(static_number_iterations);
// BENCHMARK(BM_NeuronModel_UpdateActivityBenchmark<models::IzhikevichModel>)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons)->Iterations(static_number_iterations);
// BENCHMARK(BM_NeuronModel_UpdateActivityBenchmark<models::FitzHughNagumoModel>)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons)->Iterations(static_number_iterations);
// BENCHMARK(BM_NeuronModel_UpdateActivityBenchmark<models::AEIFModel>)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons)->Iterations(static_number_iterations);

// BENCHMARK(BM_NeuronModel_UpdateFullActivity<models::PoissonModel>)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons)->Iterations(static_number_iterations);
// BENCHMARK(BM_NeuronModel_UpdateFullActivity<models::IzhikevichModel>)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons)->Iterations(static_number_iterations);
// BENCHMARK(BM_NeuronModel_UpdateFullActivity<models::FitzHughNagumoModel>)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons)->Iterations(static_number_iterations);
// BENCHMARK(BM_NeuronModel_UpdateFullActivity<models::AEIFModel>)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons)->Iterations(static_number_iterations);

// BENCHMARK(BM_NeuronModel_UpdateFullActivityBenchmark<models::PoissonModel>)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons)->Iterations(static_number_iterations);
// BENCHMARK(BM_NeuronModel_UpdateFullActivityBenchmark<models::IzhikevichModel>)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons)->Iterations(static_number_iterations);
// BENCHMARK(BM_NeuronModel_UpdateFullActivityBenchmark<models::FitzHughNagumoModel>)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons)->Iterations(static_number_iterations);
// BENCHMARK(BM_NeuronModel_UpdateFullActivityBenchmark<models::AEIFModel>)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons)->Iterations(static_number_iterations);
