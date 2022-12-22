#include "main.h"

#include "factory/background_factory.h"

#include "neurons/enums/UpdateStatus.h"

static void BM_null_background_activity_0(benchmark::State& state) {
    const auto number_neurons = state.range(0);

    for (auto _ : state) {
        state.PauseTiming();

        auto background = BackgroundFactory::construct_null_background();
        background->init(number_neurons);

        std::vector<UpdateStatus> disable_flags(number_neurons, UpdateStatus::Disabled);

        state.ResumeTiming();

        background->update_input(1000, disable_flags);

        state.PauseTiming();

        auto sum = 0.0;
        const auto& activity = background->get_background_activity();

        for (const auto val : activity) {
            sum += val;
        }

        benchmark::DoNotOptimize(sum);

        state.ResumeTiming();
    }
}

static void BM_null_background_activity_100(benchmark::State& state) {
    const auto number_neurons = state.range(0);

    for (auto _ : state) {
        state.PauseTiming();

        auto background = BackgroundFactory::construct_null_background();
        background->init(number_neurons);

        std::vector<UpdateStatus> disable_flags(number_neurons, UpdateStatus::Enabled);

        state.ResumeTiming();

        background->update_input(1000, disable_flags);

        state.PauseTiming();

        auto sum = 0.0;
        const auto& activity = background->get_background_activity();

        for (const auto val : activity) {
            sum += val;
        }

        benchmark::DoNotOptimize(sum);

        state.ResumeTiming();
    }
}

static void BM_constant_background_activity_0(benchmark::State& state) {
    const auto number_neurons = state.range(0);

    for (auto _ : state) {
        state.PauseTiming();

        auto background = BackgroundFactory::construct_constant_background(2.0);
        background->init(number_neurons);

        std::vector<UpdateStatus> disable_flags(number_neurons, UpdateStatus::Disabled);

        state.ResumeTiming();

        background->update_input(1000, disable_flags);

        state.PauseTiming();

        auto sum = 0.0;
        const auto& activity = background->get_background_activity();

        for (const auto val : activity) {
            sum += val;
        }

        benchmark::DoNotOptimize(sum);

        state.ResumeTiming();
    }
}

static void BM_constant_background_activity_100(benchmark::State& state) {
    const auto number_neurons = state.range(0);

    for (auto _ : state) {
        state.PauseTiming();

        auto background = BackgroundFactory::construct_constant_background(2.0);
        background->init(number_neurons);

        std::vector<UpdateStatus> disable_flags(number_neurons, UpdateStatus::Enabled);

        state.ResumeTiming();

        background->update_input(1000, disable_flags);

        state.PauseTiming();

        auto sum = 0.0;
        const auto& activity = background->get_background_activity();

        for (const auto val : activity) {
            sum += val;
        }

        benchmark::DoNotOptimize(sum);

        state.ResumeTiming();
    }
}

static void BM_normal_background_activity_0(benchmark::State& state) {
    const auto number_neurons = state.range(0);

    for (auto _ : state) {
        state.PauseTiming();

        auto background = BackgroundFactory::construct_normal_background(2.0, 1.5);
        background->init(number_neurons);

        std::vector<UpdateStatus> disable_flags(number_neurons, UpdateStatus::Disabled);

        state.ResumeTiming();

        background->update_input(1000, disable_flags);

        state.PauseTiming();

        auto sum = 0.0;
        const auto& activity = background->get_background_activity();

        for (const auto val : activity) {
            sum += val;
        }

        benchmark::DoNotOptimize(sum);

        state.ResumeTiming();
    }
}

static void BM_normal_background_activity_100(benchmark::State& state) {
    const auto number_neurons = state.range(0);

    for (auto _ : state) {
        state.PauseTiming();

        auto background = BackgroundFactory::construct_normal_background(2.0, 1.5);
        background->init(number_neurons);

        std::vector<UpdateStatus> disable_flags(number_neurons, UpdateStatus::Enabled);

        state.ResumeTiming();

        background->update_input(1000, disable_flags);

        state.PauseTiming();

        auto sum = 0.0;
        const auto& activity = background->get_background_activity();

        for (const auto val : activity) {
            sum += val;
        }

        benchmark::DoNotOptimize(sum);

        state.ResumeTiming();
    }
}

BENCHMARK(BM_null_background_activity_0)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons);
BENCHMARK(BM_null_background_activity_100)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons);
BENCHMARK(BM_constant_background_activity_0)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons);
BENCHMARK(BM_constant_background_activity_100)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons);
BENCHMARK(BM_normal_background_activity_0)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons);
BENCHMARK(BM_normal_background_activity_100)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons);
