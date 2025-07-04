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

#include "algorithm/Kernel/Gamma.h"
#include "algorithm/Kernel/Gaussian.h"
#include "algorithm/Kernel/Linear.h"
#include "algorithm/Kernel/Weibull.h"
#include "util/Vec3.h"

#include <utility>

#include <range/v3/numeric/accumulate.hpp>

static void BM_Gaussian_Kernel(benchmark::State& state) {
    const auto number_pairs = state.range(0);

    std::vector<std::pair<Vec3d, Vec3d>> pairs{};
    pairs.reserve(number_pairs);

    for (auto i = 0; i < number_pairs; i++) {
        Vec3d source{ i * 1.0, i * 13.5, i * 18.4 };
        Vec3d target{ i + 82.3, i * 472.4, i * (-1.3) };

        pairs.emplace_back(source, target);
    }

    std::vector<double> attractivenesses(number_pairs, 0.0);

    for (auto _ : state) {
        for (auto i = 0; i < number_pairs; i++) {
            const auto& [source, target] = pairs[i];
            const auto attr = GaussianDistributionKernel::calculate_attractiveness_to_connect(source, target, 3);

            attractivenesses[i] = attr;
        }

        state.PauseTiming();

        const auto sum = ranges::accumulate(attractivenesses, 0.0);
        benchmark::DoNotOptimize(sum);

        state.ResumeTiming();
    }
}

static void BM_Gamma_Kernel(benchmark::State& state) {
    const auto number_pairs = state.range(0);

    std::vector<std::pair<Vec3d, Vec3d>> pairs{};
    pairs.reserve(number_pairs);

    for (auto i = 0; i < number_pairs; i++) {
        Vec3d source{ i * 1.0, i * 13.5, i * 18.4 };
        Vec3d target{ i + 82.3, i * 472.4, i * (-1.3) };

        pairs.emplace_back(source, target);
    }

    std::vector<double> attractivenesses(number_pairs, 0.0);

    for (auto _ : state) {

        for (auto i = 0; i < number_pairs; i++) {
            const auto& [source, target] = pairs[i];
            const auto attr = GammaDistributionKernel::calculate_attractiveness_to_connect(source, target, 3);

            attractivenesses[i] = attr;
        }

        state.PauseTiming();

        const auto sum = ranges::accumulate(attractivenesses, 0.0);
        benchmark::DoNotOptimize(sum);

        state.ResumeTiming();
    }
}

static void BM_Linear_Kernel(benchmark::State& state) {
    const auto number_pairs = state.range(0);

    std::vector<std::pair<Vec3d, Vec3d>> pairs{};
    pairs.reserve(number_pairs);

    for (auto i = 0; i < number_pairs; i++) {
        Vec3d source{ i * 1.0, i * 13.5, i * 18.4 };
        Vec3d target{ i + 82.3, i * 472.4, i * (-1.3) };

        pairs.emplace_back(source, target);
    }

    std::vector<double> attractivenesses(number_pairs, 0.0);

    for (auto _ : state) {

        for (auto i = 0; i < number_pairs; i++) {
            const auto& [source, target] = pairs[i];
            const auto attr = LinearDistributionKernel::calculate_attractiveness_to_connect(source, target, 3);

            attractivenesses[i] = attr;
        }

        state.PauseTiming();

        const auto sum = ranges::accumulate(attractivenesses, 0.0);
        benchmark::DoNotOptimize(sum);

        state.ResumeTiming();
    }
}

static void BM_Weibull_Kernel(benchmark::State& state) {
    const auto number_pairs = state.range(0);

    std::vector<std::pair<Vec3d, Vec3d>> pairs{};
    pairs.reserve(number_pairs);

    for (auto i = 0; i < number_pairs; i++) {
        Vec3d source{ i * 1.0, i * 13.5, i * 18.4 };
        Vec3d target{ i + 82.3, i * 472.4, i * (-1.3) };

        pairs.emplace_back(source, target);
    }

    std::vector<double> attractivenesses(number_pairs, 0.0);

    for (auto _ : state) {

        for (auto i = 0; i < number_pairs; i++) {
            const auto& [source, target] = pairs[i];
            const auto attr = WeibullDistributionKernel::calculate_attractiveness_to_connect(source, target, 3);

            attractivenesses[i] = attr;
        }

        state.PauseTiming();

        const auto sum = ranges::accumulate(attractivenesses, 0.0);
        benchmark::DoNotOptimize(sum);

        state.ResumeTiming();
    }
}

BENCHMARK(BM_Gaussian_Kernel)->Unit(benchmark::kMillisecond)->Arg(1000)->Iterations(static_many_iterations);
BENCHMARK(BM_Gaussian_Kernel)->Unit(benchmark::kMillisecond)->Arg(4000)->Iterations(static_many_iterations);
BENCHMARK(BM_Gaussian_Kernel)->Unit(benchmark::kMillisecond)->Arg(16000)->Iterations(static_many_iterations);
BENCHMARK(BM_Gaussian_Kernel)->Unit(benchmark::kMillisecond)->Arg(64000)->Iterations(static_many_iterations);
BENCHMARK(BM_Gaussian_Kernel)->Unit(benchmark::kMillisecond)->Arg(256000)->Iterations(static_many_iterations);

BENCHMARK(BM_Gamma_Kernel)->Unit(benchmark::kMillisecond)->Arg(1000)->Iterations(static_many_iterations);
BENCHMARK(BM_Gamma_Kernel)->Unit(benchmark::kMillisecond)->Arg(4000)->Iterations(static_many_iterations);
BENCHMARK(BM_Gamma_Kernel)->Unit(benchmark::kMillisecond)->Arg(16000)->Iterations(static_many_iterations);
BENCHMARK(BM_Gamma_Kernel)->Unit(benchmark::kMillisecond)->Arg(64000)->Iterations(static_many_iterations);
BENCHMARK(BM_Gamma_Kernel)->Unit(benchmark::kMillisecond)->Arg(256000)->Iterations(static_many_iterations);

BENCHMARK(BM_Linear_Kernel)->Unit(benchmark::kMillisecond)->Arg(1000)->Iterations(static_many_iterations);
BENCHMARK(BM_Linear_Kernel)->Unit(benchmark::kMillisecond)->Arg(4000)->Iterations(static_many_iterations);
BENCHMARK(BM_Linear_Kernel)->Unit(benchmark::kMillisecond)->Arg(16000)->Iterations(static_many_iterations);
BENCHMARK(BM_Linear_Kernel)->Unit(benchmark::kMillisecond)->Arg(64000)->Iterations(static_many_iterations);
BENCHMARK(BM_Linear_Kernel)->Unit(benchmark::kMillisecond)->Arg(256000)->Iterations(static_many_iterations);

BENCHMARK(BM_Weibull_Kernel)->Unit(benchmark::kMillisecond)->Arg(1000)->Iterations(static_many_iterations);
BENCHMARK(BM_Weibull_Kernel)->Unit(benchmark::kMillisecond)->Arg(4000)->Iterations(static_many_iterations);
BENCHMARK(BM_Weibull_Kernel)->Unit(benchmark::kMillisecond)->Arg(16000)->Iterations(static_many_iterations);
BENCHMARK(BM_Weibull_Kernel)->Unit(benchmark::kMillisecond)->Arg(64000)->Iterations(static_many_iterations);
BENCHMARK(BM_Weibull_Kernel)->Unit(benchmark::kMillisecond)->Arg(256000)->Iterations(static_many_iterations);
