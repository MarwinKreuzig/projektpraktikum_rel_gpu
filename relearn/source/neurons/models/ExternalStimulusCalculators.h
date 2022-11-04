#pragma once

#include "ExternalStimulusCalculator.h"
#include "Types.h"

class NullExternalStimulusCalculator : public ExternalStimulusCalculator {
public:
    /**
     * @brief Constructs a new object of type NullBackgroundActivityCalculator
     */
    NullExternalStimulusCalculator() = default;

    virtual ~NullExternalStimulusCalculator() = default;

    /**
     * @brief This activity calculator does not provide any input
     * @param step The current update step
     * @param disable_flags Unused
     */
    void update_input([[maybe_unused]] const size_t step, [[maybe_unused]] const std::vector<UpdateStatus>& disable_flags) override {
    }

    /**
     * @brief Creates a clone of this instance (without neurons), copies all parameters
     * @return A copy of this instance
     */
    [[nodiscard]] std::unique_ptr<ExternalStimulusCalculator> clone() const override {
        return std::make_unique<NullExternalStimulusCalculator>();
    }
};

class FunctionExternalStimulusCalculator : public ExternalStimulusCalculator {
public:
    /**
     * @brief Constructs a new object of type NullBackgroundActivityCalculator
     */
    FunctionExternalStimulusCalculator(std::unique_ptr<ExternalStimulusFunction>&& external_stimulus_function ) : external_stimulus_function(std::move(external_stimulus_function)) {

    }

    FunctionExternalStimulusCalculator(FunctionExternalStimulusCalculator& f) = delete;

    virtual ~FunctionExternalStimulusCalculator() = default;

    /**
     * @brief This activity calculator does not provide any input
     * @param step The current update step
     * @param disable_flags Unused
     */
    void update_input([[maybe_unused]] const size_t step, [[maybe_unused]] const std::vector<UpdateStatus>& disable_flags) override {
        const auto number_neurons = get_number_neurons();
        RelearnException::check(disable_flags.size() == number_neurons,
                                "ConstantBackgroundActivityCalculator::update_input: Size of disable flags doesn't match number of local neurons: {} vs {}", disable_flags.size(), number_neurons);

        Timers::start(TimerRegion::CALC_EXTERNAL_STIMULUS);
        reset_external_stimulus();

        auto external_stimulus_vector = (*external_stimulus_function)(step);

        for (auto [neuron_id, voltage] : external_stimulus_vector) {
            const auto input = disable_flags[neuron_id] == UpdateStatus::Disabled ? 0.0 : voltage ;
            set_external_stimulus(neuron_id, input);
        }
        Timers::stop_and_add(TimerRegion::CALC_EXTERNAL_STIMULUS);
    }

    /**
     * @brief Creates a clone of this instance (without neurons), copies all parameters
     * @return A copy of this instance
     */
    [[nodiscard]] std::unique_ptr<ExternalStimulusCalculator> clone() const override {
        return std::make_unique<FunctionExternalStimulusCalculator>(std::make_unique<ExternalStimulusFunction>(*external_stimulus_function));
    }

private:
    std::unique_ptr<ExternalStimulusFunction> external_stimulus_function{};
};