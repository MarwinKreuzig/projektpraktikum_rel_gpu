#include "main.h"

#include "algorithm/Kernel/Gaussian.h"
#include "util/Vec3.h"

#include <utility>

static void BM_Gaussian_Kernel(benchmark::State& state) {
    const auto number_pairs = state.range(0);

    for (auto _ : state) {
        state.PauseTiming();

        std::vector<std::pair<Vec3d, Vec3d>> pairs{};
        pairs.reserve(number_pairs);

        for (auto i = 0; i < number_pairs; i++) {
            Vec3d source{ i * 1.0, i * 13.5, i * 18.4 };
            Vec3d target{ i + 82.3, i * 472.4, i * (-1.3) };

            pairs.emplace_back(source, target);
        }

        std::vector<double> attractivenesses(number_pairs, 0.0);

        state.ResumeTiming();

        for (auto i = 0; i < number_pairs; i++) {
            const auto& [source, target] = pairs[i];
            const auto attr = GaussianDistributionKernel::calculate_attractiveness_to_connect(source, target, 3);

            attractivenesses[i] = attr;
        }

        state.PauseTiming();

        const auto sum = std::reduce(attractivenesses.begin(), attractivenesses.end(), 0.0);
        benchmark::DoNotOptimize(sum);

        state.ResumeTiming();
    }
}

BENCHMARK(BM_Gaussian_Kernel)->Unit(benchmark::kMillisecond)->Arg(1000)->Iterations(1000);
BENCHMARK(BM_Gaussian_Kernel)->Unit(benchmark::kMillisecond)->Arg(4000)->Iterations(1000);
BENCHMARK(BM_Gaussian_Kernel)->Unit(benchmark::kMillisecond)->Arg(16000)->Iterations(1000);
BENCHMARK(BM_Gaussian_Kernel)->Unit(benchmark::kMillisecond)->Arg(64000)->Iterations(1000);
BENCHMARK(BM_Gaussian_Kernel)->Unit(benchmark::kMillisecond)->Arg(256000)->Iterations(1000);
