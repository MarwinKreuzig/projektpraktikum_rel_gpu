#include "main.h"

#include "AdapterNeuronModel.h"

#include <numeric>

constexpr auto static_number_neurons = 1000000;

template <typename NeuronModelType>
static void BM_NeuronModel_SerialInitialize(benchmark::State& state) {

    const auto number_neurons = state.range(0);

    std::vector<UpdateStatus> disable_flags(number_neurons, UpdateStatus::ENABLED);

    for (auto _ : state) {
        state.PauseTiming();

        NeuronModelType model{};
        model.init(number_neurons);

        AdapterNeuronModel adapter{ model };

        state.ResumeTiming();

        adapter.calculate_serial_initialize(disable_flags);
    }
}

template <typename NeuronModelType>
static void BM_NeuronModel_BackgroundActivity0(benchmark::State& state) {

    const auto number_neurons = state.range(0);

    std::vector<UpdateStatus> disable_flags(number_neurons, UpdateStatus::ENABLED);

    for (auto _ : state) {
        state.PauseTiming();

        NeuronModelType model{};
        model.init(number_neurons);

        AdapterNeuronModel adapter{ model };

        state.ResumeTiming();

        adapter.calculate_background_activity(disable_flags);

        state.PauseTiming();

        const auto& background = adapter.get_background();
        const auto sum = std::reduce(background.begin(), background.end(), 0.0);

        benchmark::DoNotOptimize(sum);

        state.ResumeTiming();
    }
}

template <typename NeuronModelType>
static void BM_NeuronModel_BackgroundActivityConstant(benchmark::State& state) {

    const auto number_neurons = state.range(0);

    std::vector<UpdateStatus> disable_flags(number_neurons, UpdateStatus::ENABLED);

    for (auto _ : state) {
        state.PauseTiming();

        NeuronModelType model{};
        model.init(number_neurons);

        std::vector<ModelParameter> parameter = model.get_parameter();
        Parameter<double> parameter_4 = std::get<Parameter<double>>(parameter[4]);
        parameter_4.set_value(1.0);

        AdapterNeuronModel adapter{ model };

        state.ResumeTiming();

        adapter.calculate_background_activity(disable_flags);

        state.PauseTiming();

        const auto& background = adapter.get_background();
        const auto sum = std::reduce(background.begin(), background.end(), 0.0);

        benchmark::DoNotOptimize(sum);

        state.ResumeTiming();
    }
}

template <typename NeuronModelType>
static void BM_NeuronModel_BackgroundActivityVariable(benchmark::State& state) {

    const auto number_neurons = state.range(0);

    std::vector<UpdateStatus> disable_flags(number_neurons, UpdateStatus::ENABLED);

    for (auto _ : state) {
        state.PauseTiming();

        NeuronModelType model{};
        model.init(number_neurons);

        std::vector<ModelParameter> parameter = model.get_parameter();
        Parameter<double> parameter_4 = std::get<Parameter<double>>(parameter[4]);
        parameter_4.set_value(1.0);

        Parameter<double> parameter_5 = std::get<Parameter<double>>(parameter[5]);
        parameter_5.set_value(13.0);

        Parameter<double> parameter_6 = std::get<Parameter<double>>(parameter[6]);
        parameter_6.set_value(0.4);

        AdapterNeuronModel adapter{ model };

        state.ResumeTiming();

        adapter.calculate_background_activity(disable_flags);

        state.PauseTiming();

        const auto& background = adapter.get_background();
        const auto sum = std::reduce(background.begin(), background.end(), 0.0);

        benchmark::DoNotOptimize(sum);

        state.ResumeTiming();
    }
}

BENCHMARK(BM_NeuronModel_SerialInitialize<models::PoissonModel>)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons);
BENCHMARK(BM_NeuronModel_SerialInitialize<models::IzhikevichModel>)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons);
BENCHMARK(BM_NeuronModel_SerialInitialize<models::FitzHughNagumoModel>)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons);
BENCHMARK(BM_NeuronModel_SerialInitialize<models::AEIFModel>)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons);

BENCHMARK(BM_NeuronModel_BackgroundActivity0<models::PoissonModel>)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons);
BENCHMARK(BM_NeuronModel_BackgroundActivity0<models::IzhikevichModel>)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons);
BENCHMARK(BM_NeuronModel_BackgroundActivity0<models::FitzHughNagumoModel>)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons);
BENCHMARK(BM_NeuronModel_BackgroundActivity0<models::AEIFModel>)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons);

BENCHMARK(BM_NeuronModel_BackgroundActivityConstant<models::PoissonModel>)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons);
BENCHMARK(BM_NeuronModel_BackgroundActivityConstant<models::IzhikevichModel>)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons);
BENCHMARK(BM_NeuronModel_BackgroundActivityConstant<models::FitzHughNagumoModel>)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons);
BENCHMARK(BM_NeuronModel_BackgroundActivityConstant<models::AEIFModel>)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons);

BENCHMARK(BM_NeuronModel_BackgroundActivityVariable<models::PoissonModel>)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons);
BENCHMARK(BM_NeuronModel_BackgroundActivityVariable<models::IzhikevichModel>)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons);
BENCHMARK(BM_NeuronModel_BackgroundActivityVariable<models::FitzHughNagumoModel>)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons);
BENCHMARK(BM_NeuronModel_BackgroundActivityVariable<models::AEIFModel>)->Unit(benchmark::kMillisecond)->Arg(static_number_neurons);
