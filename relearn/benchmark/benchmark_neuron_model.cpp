#include "main.h"

#include "AdapterNeuronModel.h"

#include "factory/background_factory.h"
#include "factory/input_factory.h"
#include "factory/neuron_model_factory.h"

#include <numeric>

template <typename NeuronModelType>
static void BM_NeuronModel_UpdateActivityBenchmark(benchmark::State& state) {
    const auto number_neurons = state.range(0);

    for (auto _ : state) {
        state.PauseTiming();

        auto synaptic_input = InputFactory::construct_linear_input();
        auto background = BackgroundFactory::construct_null_background();

        const auto stimulus = std::make_shared<Stimulus>();
        auto model = NeuronModelFactory::construct_model<NeuronModelType>(10, std::move(synaptic_input), std::move(background), stimulus);
        model->init(number_neurons);
        model->set_stimulus_calculator(stimulus);

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

template <typename NeuronModelType>
static void BM_NeuronModel_UpdateFullActivity(benchmark::State& state) {
    const auto number_neurons = state.range(0);

    for (auto _ : state) {
        state.PauseTiming();

        auto synaptic_input = InputFactory::construct_linear_input();
        auto background = BackgroundFactory::construct_null_background();

        const auto stimulus = std::make_shared<Stimulus>();
        auto model = NeuronModelFactory::construct_model<NeuronModelType>(10, std::move(synaptic_input), std::move(background), stimulus);
        model->init(number_neurons);
        model->set_stimulus_calculator(stimulus);

        AdapterNeuronModel<NeuronModelType> adapter{ *model };

        std::vector<UpdateStatus> update_flags(number_neurons, UpdateStatus::Enabled);

        state.ResumeTiming();
        adapter.update_activity(update_flags);
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
static void BM_NeuronModel_UpdateFullActivityBenchmark(benchmark::State& state) {
    const auto number_neurons = state.range(0);

    for (auto _ : state) {
        state.PauseTiming();

        auto synaptic_input = InputFactory::construct_linear_input();
        auto background = BackgroundFactory::construct_null_background();

        const auto stimulus = std::make_shared<Stimulus>();
        auto model = NeuronModelFactory::construct_model<NeuronModelType>(10, std::move(synaptic_input), std::move(background), stimulus);
        model->init(number_neurons);
        model->set_stimulus_calculator(stimulus);

        AdapterNeuronModel<NeuronModelType> adapter{ *model };

        std::vector<UpdateStatus> update_flags(number_neurons, UpdateStatus::Enabled);

        state.ResumeTiming();
        adapter.update_activity_benchmark(update_flags);
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

BENCHMARK(BM_NeuronModel_UpdateActivityBenchmark<models::PoissonModel>)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons)->Iterations(50);
BENCHMARK(BM_NeuronModel_UpdateActivityBenchmark<models::IzhikevichModel>)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons)->Iterations(50);
BENCHMARK(BM_NeuronModel_UpdateActivityBenchmark<models::FitzHughNagumoModel>)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons)->Iterations(50);
BENCHMARK(BM_NeuronModel_UpdateActivityBenchmark<models::AEIFModel>)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons)->Iterations(50);

BENCHMARK(BM_NeuronModel_UpdateFullActivity<models::PoissonModel>)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons)->Iterations(50);
BENCHMARK(BM_NeuronModel_UpdateFullActivity<models::IzhikevichModel>)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons)->Iterations(50);
BENCHMARK(BM_NeuronModel_UpdateFullActivity<models::FitzHughNagumoModel>)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons)->Iterations(50);
BENCHMARK(BM_NeuronModel_UpdateFullActivity<models::AEIFModel>)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons)->Iterations(50);
                               
BENCHMARK(BM_NeuronModel_UpdateFullActivityBenchmark<models::PoissonModel>)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons)->Iterations(50);
BENCHMARK(BM_NeuronModel_UpdateFullActivityBenchmark<models::IzhikevichModel>)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons)->Iterations(50);
BENCHMARK(BM_NeuronModel_UpdateFullActivityBenchmark<models::FitzHughNagumoModel>)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons)->Iterations(50);
BENCHMARK(BM_NeuronModel_UpdateFullActivityBenchmark<models::AEIFModel>)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons)->Iterations(50);
