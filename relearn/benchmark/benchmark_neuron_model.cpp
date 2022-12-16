#include "main.h"

#include "AdapterNeuronModel.h"

#include "factory/background_factory.h"
#include "factory/input_factory.h"
#include "factory/neuron_model_factory.h"

#include <numeric>

constexpr auto static_number_neurons = 10000000;
constexpr auto static_number_synapses = 20;

template <typename NeuronModelType>
static void BM_NeuronModel_UpdateActivity(benchmark::State& state) {
    const auto number_neurons = state.range(0);

    for (auto _ : state) {
        state.PauseTiming();

        auto synaptic_input = InputFactory::construct_linear_input();
        auto background = BackgroundFactory::construct_null_background();

        auto model = NeuronModelFactory::construct_model<NeuronModelType>(10, std::move(synaptic_input), std::move(background), std::make_unique<Stimulus>());
        model->init(number_neurons);

        AdapterNeuronModel<NeuronModelType> adapter{ *model };

        state.ResumeTiming();

        for (const auto& neuron_id : NeuronID::range(number_neurons)) {
            adapter.update_activity(neuron_id);
        }

        state.PauseTiming();

        auto sum = 0.0;
        const auto& x = adapter.get_x();

        for (const auto val : x) {
            sum += val;
        }

        benchmark::DoNotOptimize(sum);

        state.ResumeTiming();
    }
}

template <typename NeuronModelType>
static void BM_NeuronModel_UpdateActivityBenchmark(benchmark::State& state) {
    const auto number_neurons = state.range(0);

    for (auto _ : state) {
        state.PauseTiming();

        auto synaptic_input = InputFactory::construct_linear_input();
        auto background = BackgroundFactory::construct_null_background();

        auto model = NeuronModelFactory::construct_model<NeuronModelType>(10, std::move(synaptic_input), std::move(background), std::make_unique<Stimulus>());
        model->init(number_neurons);

        AdapterNeuronModel<NeuronModelType> adapter{ *model };

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

        state.ResumeTiming();
    }
}

BENCHMARK(BM_NeuronModel_UpdateActivity<models::PoissonModel>)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons);
BENCHMARK(BM_NeuronModel_UpdateActivity<models::IzhikevichModel>)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons);
BENCHMARK(BM_NeuronModel_UpdateActivity<models::FitzHughNagumoModel>)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons);
BENCHMARK(BM_NeuronModel_UpdateActivity<models::AEIFModel>)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons);

BENCHMARK(BM_NeuronModel_UpdateActivityBenchmark<models::PoissonModel>)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons);
BENCHMARK(BM_NeuronModel_UpdateActivityBenchmark<models::IzhikevichModel>)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons);
BENCHMARK(BM_NeuronModel_UpdateActivityBenchmark<models::FitzHughNagumoModel>)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons);
BENCHMARK(BM_NeuronModel_UpdateActivityBenchmark<models::AEIFModel>)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons);
