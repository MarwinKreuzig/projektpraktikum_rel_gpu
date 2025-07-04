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

#include "factory/calcium_factory.h"
#include "factory/extra_info_factory.h"

#include <numeric>

static void BM_CalciumCalculator_No_Decay_No_Fired(benchmark::State& state) {
    const auto number_neurons = state.range(0);
    const auto decay = state.range(1);
    const auto beta = state.range(2);

    std::vector<FiredStatus> fired_status(number_neurons, FiredStatus::Inactive);

    auto calcium_calculator = CalciumFactory::construct_calcium_calculator_no_decay();
    calcium_calculator->init(number_neurons);
    calcium_calculator->set_tau_C(static_cast<double>(decay));
    calcium_calculator->set_beta(1.0 / static_cast<double>(beta));

    auto extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    extra_info->init(number_neurons);

    calcium_calculator->set_extra_infos(extra_info);

    NeuronsExtraInfoFactory::enable_all(extra_info);

    for (auto _ : state) {
        calcium_calculator->update_calcium(100, fired_status);
        state.PauseTiming();

        auto sum = 0.0;
        const auto& values = calcium_calculator->get_calcium();

        for (const auto val : values) {
            sum += val;
        }

        benchmark::DoNotOptimize(sum);
        state.ResumeTiming();
    }

    int i = 0;
    i++;
}

static void BM_CalciumCalculator_No_Decay_All_Fired(benchmark::State& state) {
    const auto number_neurons = state.range(0);
    const auto decay = state.range(1);
    const auto beta = state.range(2);

    std::vector<FiredStatus> fired_status(number_neurons, FiredStatus::Fired);

    auto calcium_calculator = CalciumFactory::construct_calcium_calculator_no_decay();
    calcium_calculator->init(number_neurons);
    calcium_calculator->set_tau_C(static_cast<double>(decay));
    calcium_calculator->set_beta(1.0 / static_cast<double>(beta));

    auto extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    extra_info->init(number_neurons);

    calcium_calculator->set_extra_infos(extra_info);

    NeuronsExtraInfoFactory::enable_all(extra_info);

    for (auto _ : state) {
        calcium_calculator->update_calcium(100, fired_status);
        state.PauseTiming();

        auto sum = 0.0;
        const auto& values = calcium_calculator->get_calcium();

        for (const auto val : values) {
            sum += val;
        }

        benchmark::DoNotOptimize(sum);
        state.ResumeTiming();
    }
}

static void BM_CalciumCalculator_Relative_Decay_No_Fired(benchmark::State& state) {
    const auto number_neurons = state.range(0);

    std::vector<FiredStatus> fired_status(number_neurons, FiredStatus::Inactive);

    auto calcium_calculator = CalciumFactory::construct_calcium_calculator_relative_decay();
    calcium_calculator->init(number_neurons);

    auto extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    extra_info->init(number_neurons);

    calcium_calculator->set_extra_infos(extra_info);

    NeuronsExtraInfoFactory::enable_all(extra_info);

    for (auto _ : state) {
        calcium_calculator->update_calcium(100, fired_status);
        state.PauseTiming();

        auto sum = 0.0;
        const auto& values = calcium_calculator->get_calcium();

        for (const auto val : values) {
            sum += val;
        }

        benchmark::DoNotOptimize(sum);
        state.ResumeTiming();
    }
}

static void BM_CalciumCalculator_Relative_Decay_All_Fired(benchmark::State& state) {
    const auto number_neurons = state.range(0);

    std::vector<FiredStatus> fired_status(number_neurons, FiredStatus::Fired);

    auto calcium_calculator = CalciumFactory::construct_calcium_calculator_relative_decay();
    calcium_calculator->init(number_neurons);

    auto extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    extra_info->init(number_neurons);

    calcium_calculator->set_extra_infos(extra_info);

    NeuronsExtraInfoFactory::enable_all(extra_info);

    for (auto _ : state) {
        calcium_calculator->update_calcium(100, fired_status);
        state.PauseTiming();

        auto sum = 0.0;
        const auto& values = calcium_calculator->get_calcium();

        for (const auto val : values) {
            sum += val;
        }

        benchmark::DoNotOptimize(sum);
        state.ResumeTiming();
    }
}

static void BM_CalciumCalculator_Absolute_Decay_No_Fired(benchmark::State& state) {
    const auto number_neurons = state.range(0);

    std::vector<FiredStatus> fired_status(number_neurons, FiredStatus::Inactive);

    auto calcium_calculator = CalciumFactory::construct_calcium_calculator_absolute_decay();
    calcium_calculator->init(number_neurons);

    auto extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    extra_info->init(number_neurons);

    calcium_calculator->set_extra_infos(extra_info);

    NeuronsExtraInfoFactory::enable_all(extra_info);

    for (auto _ : state) {
        calcium_calculator->update_calcium(100, fired_status);
        state.PauseTiming();

        auto sum = 0.0;
        const auto& values = calcium_calculator->get_calcium();

        for (const auto val : values) {
            sum += val;
        }

        benchmark::DoNotOptimize(sum);
        state.ResumeTiming();
    }
}

static void BM_CalciumCalculator_Absolute_Decay_All_Fired(benchmark::State& state) {
    const auto number_neurons = state.range(0);

    std::vector<FiredStatus> fired_status(number_neurons, FiredStatus::Fired);

    auto calcium_calculator = CalciumFactory::construct_calcium_calculator_absolute_decay();
    calcium_calculator->init(number_neurons);

    auto extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    extra_info->init(number_neurons);

    calcium_calculator->set_extra_infos(extra_info);

    NeuronsExtraInfoFactory::enable_all(extra_info);

    for (auto _ : state) {
        calcium_calculator->update_calcium(100, fired_status);
        state.PauseTiming();

        auto sum = 0.0;
        const auto& values = calcium_calculator->get_calcium();

        for (const auto val : values) {
            sum += val;
        }

        benchmark::DoNotOptimize(sum);
        state.ResumeTiming();
    }
}

BENCHMARK(BM_CalciumCalculator_No_Decay_No_Fired)->Unit(benchmark::kMillisecond)->Args({ static_number_neurons, 1000, 100 })->Iterations(static_few_iterations);
BENCHMARK(BM_CalciumCalculator_No_Decay_No_Fired)->Unit(benchmark::kMillisecond)->Args({ static_number_neurons, 5000, 100 })->Iterations(static_few_iterations);
BENCHMARK(BM_CalciumCalculator_No_Decay_No_Fired)->Unit(benchmark::kMillisecond)->Args({ static_number_neurons, 10000, 100 })->Iterations(static_few_iterations);
BENCHMARK(BM_CalciumCalculator_No_Decay_No_Fired)->Unit(benchmark::kMillisecond)->Args({ static_number_neurons, 15000, 100 })->Iterations(static_few_iterations);
BENCHMARK(BM_CalciumCalculator_No_Decay_No_Fired)->Unit(benchmark::kMillisecond)->Args({ static_number_neurons, 20000, 100 })->Iterations(static_few_iterations);
BENCHMARK(BM_CalciumCalculator_No_Decay_No_Fired)->Unit(benchmark::kMillisecond)->Args({ static_number_neurons, 50000, 100 })->Iterations(static_few_iterations);

BENCHMARK(BM_CalciumCalculator_No_Decay_No_Fired)->Unit(benchmark::kMillisecond)->Args({ static_number_neurons, 1000, 1000 })->Iterations(static_few_iterations);
BENCHMARK(BM_CalciumCalculator_No_Decay_No_Fired)->Unit(benchmark::kMillisecond)->Args({ static_number_neurons, 5000, 1000 })->Iterations(static_few_iterations);
BENCHMARK(BM_CalciumCalculator_No_Decay_No_Fired)->Unit(benchmark::kMillisecond)->Args({ static_number_neurons, 10000, 1000 })->Iterations(static_few_iterations);
BENCHMARK(BM_CalciumCalculator_No_Decay_No_Fired)->Unit(benchmark::kMillisecond)->Args({ static_number_neurons, 15000, 1000 })->Iterations(static_few_iterations);
BENCHMARK(BM_CalciumCalculator_No_Decay_No_Fired)->Unit(benchmark::kMillisecond)->Args({ static_number_neurons, 20000, 1000 })->Iterations(static_few_iterations);
BENCHMARK(BM_CalciumCalculator_No_Decay_No_Fired)->Unit(benchmark::kMillisecond)->Args({ static_number_neurons, 50000, 1000 })->Iterations(static_few_iterations);

BENCHMARK(BM_CalciumCalculator_No_Decay_No_Fired)->Unit(benchmark::kMillisecond)->Args({ static_number_neurons, 1000, 10000 })->Iterations(static_few_iterations);
BENCHMARK(BM_CalciumCalculator_No_Decay_No_Fired)->Unit(benchmark::kMillisecond)->Args({ static_number_neurons, 5000, 10000 })->Iterations(static_few_iterations);
BENCHMARK(BM_CalciumCalculator_No_Decay_No_Fired)->Unit(benchmark::kMillisecond)->Args({ static_number_neurons, 10000, 10000 })->Iterations(static_few_iterations);
BENCHMARK(BM_CalciumCalculator_No_Decay_No_Fired)->Unit(benchmark::kMillisecond)->Args({ static_number_neurons, 15000, 10000 })->Iterations(static_few_iterations);
BENCHMARK(BM_CalciumCalculator_No_Decay_No_Fired)->Unit(benchmark::kMillisecond)->Args({ static_number_neurons, 20000, 10000 })->Iterations(static_few_iterations);
BENCHMARK(BM_CalciumCalculator_No_Decay_No_Fired)->Unit(benchmark::kMillisecond)->Args({ static_number_neurons, 50000, 10000 })->Iterations(static_few_iterations);

// BENCHMARK(BM_CalciumCalculator_No_Decay_All_Fired)->Unit(benchmark::kMillisecond)->Args({ static_number_neurons, 1000 })->Iterations(static_few_iterations);

// BENCHMARK(BM_CalciumCalculator_Relative_Decay_No_Fired)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons)->Iterations(static_few_iterations);
// BENCHMARK(BM_CalciumCalculator_Relative_Decay_All_Fired)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons)->Iterations(static_few_iterations);

// BENCHMARK(BM_CalciumCalculator_Absolute_Decay_No_Fired)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons)->Iterations(static_few_iterations);
// BENCHMARK(BM_CalciumCalculator_Absolute_Decay_All_Fired)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons)->Iterations(static_few_iterations);
