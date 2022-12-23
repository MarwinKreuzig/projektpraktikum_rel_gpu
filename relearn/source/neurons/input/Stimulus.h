#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "Types.h"
#include "io/InteractiveNeuronIO.h"
#include "neurons/LocalAreaTranslator.h"
#include "neurons/enums/UpdateStatus.h"
#include "util/MPIRank.h"
#include "util/TaggedID.h"
#include "util/Timers.h"

#include <filesystem>
#include <memory>
#include <vector>

class Stimulus {
public:
    Stimulus() = default;

    Stimulus(RelearnTypes::stimuli_function_type stimulus_function)
        : stimulus_function(std::move(stimulus_function)) {
    }

    Stimulus(const std::filesystem::path& stimulus_file, const MPIRank mpi_rank, std::shared_ptr<LocalAreaTranslator> local_area_translator) {
        stimulus_function = InteractiveNeuronIO::load_stimulus_interrupts(stimulus_file, mpi_rank, std::move(local_area_translator));
    }

    void init(const RelearnTypes::number_neurons_type number_neurons) {
        stimulus.resize(number_neurons, 0.0);
    }

    void create_neurons(const RelearnTypes::number_neurons_type creation_count) {
        const auto current_size = stimulus.size();
        const auto new_size = current_size + creation_count;

        stimulus.resize(new_size, 0.0);
    }

    void update_stimulus(const RelearnTypes::step_type step, const std::span<const UpdateStatus> disable_flags) {
        if (!stimulus_function.operator bool()) {
            return;
        }

        Timers::start(TimerRegion::CALC_STIMULUS);
        std::fill(stimulus.begin(), stimulus.end(), 0.0);
        const auto& stimuli = stimulus_function(step);
        for (const auto& [neuron_ids, intensity] : stimuli) {
            for (const auto& neuron_id : neuron_ids) {
                stimulus[neuron_id.get_neuron_id()] = intensity;
            }
        }
        Timers::stop_and_add(TimerRegion::CALC_STIMULUS);
    }

    [[nodiscard]] double get_stimulus(const NeuronID neuron_id) const {
        return stimulus[neuron_id.get_neuron_id()];
    }

    [[nodiscard]] std::unique_ptr<Stimulus> clone() const {
        return std::make_unique<Stimulus>(stimulus_function);
    }

private:
    std::vector<double> stimulus{};
    RelearnTypes::stimuli_function_type stimulus_function{};
};
