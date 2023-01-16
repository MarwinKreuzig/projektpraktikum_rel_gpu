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

#include "factory/background_factory.h"
#include "factory/extra_info_factory.h"

#include "neurons/enums/UpdateStatus.h"

static void BM_null_background_activity_100(benchmark::State& state) {
    const auto number_neurons = state.range(0);

    auto background = BackgroundFactory::construct_null_background();
    background->init(number_neurons);

    auto extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    extra_info->init(number_neurons);

    background->set_extra_infos(std::move(extra_info));

    for (auto _ : state) {
        state.ResumeTiming();
        background->update_input(1000);
        state.PauseTiming();

        auto sum = 0.0;
        const auto& activity = background->get_background_activity();

        for (const auto val : activity) {
            sum += val;
        }

        benchmark::DoNotOptimize(sum);
    }
}

static void BM_constant_background_activity_100(benchmark::State& state) {
    const auto number_neurons = state.range(0);

    auto background = BackgroundFactory::construct_constant_background(2.0);
    background->init(number_neurons);

    auto extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    extra_info->init(number_neurons);

    background->set_extra_infos(std::move(extra_info));

    for (auto _ : state) {
        state.ResumeTiming();
        background->update_input(1000);
        state.PauseTiming();

        auto sum = 0.0;
        const auto& activity = background->get_background_activity();

        for (const auto val : activity) {
            sum += val;
        }

        benchmark::DoNotOptimize(sum);
    }
}

static void BM_normal_background_activity_100(benchmark::State& state) {
    const auto number_neurons = state.range(0);

    auto background = BackgroundFactory::construct_normal_background(2.0, 1.5);
    background->init(number_neurons);

    auto extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    extra_info->init(number_neurons);

    background->set_extra_infos(std::move(extra_info));

    for (auto _ : state) {
        state.ResumeTiming();
        background->update_input(1000);
        state.PauseTiming();

        auto sum = 0.0;
        const auto& activity = background->get_background_activity();

        for (const auto val : activity) {
            sum += val;
        }

        benchmark::DoNotOptimize(sum);
    }
}

static void BM_fast_normal_background_activity_100(benchmark::State& state) {
    const auto number_neurons = state.range(0);

    auto background = BackgroundFactory::construct_fast_normal_background(2.0, 1.5, 5);
    background->init(number_neurons);

    auto extra_info = NeuronsExtraInfoFactory::construct_extra_info();
    extra_info->init(number_neurons);

    background->set_extra_infos(std::move(extra_info));

    for (auto _ : state) {
        state.ResumeTiming();
        background->update_input(1000);
        state.PauseTiming();

        auto sum = 0.0;
        const auto& activity = background->get_background_activity();

        for (const auto val : activity) {
            sum += val;
        }

        benchmark::DoNotOptimize(sum);
    }
}

// BENCHMARK(BM_null_background_activity_100)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons);
// BENCHMARK(BM_constant_background_activity_100)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons);
// BENCHMARK(BM_normal_background_activity_100)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons);
// BENCHMARK(BM_fast_normal_background_activity_100)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons);
