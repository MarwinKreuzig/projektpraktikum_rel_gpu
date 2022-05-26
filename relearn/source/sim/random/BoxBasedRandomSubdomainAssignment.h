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

#include "Config.h"
#include "sim/NeuronToSubdomainAssignment.h"
#include "util/RelearnException.h"
#include "util/TaggedID.h"
#include "util/Vec3.h"

#include <functional>
#include <memory>
#include <vector>

class Partition;

class BoxBasedRandomSubdomainAssignment : public NeuronToSubdomainAssignment {
public:
    BoxBasedRandomSubdomainAssignment(std::shared_ptr<Partition> partition, const double fraction_excitatory_neurons, const double um_per_neuron)
        : NeuronToSubdomainAssignment(std::move(partition))
        , um_per_neuron_(um_per_neuron) {

        RelearnException::check(fraction_excitatory_neurons >= 0.0 && fraction_excitatory_neurons <= 1.0,
            "BoxBasedRandomSubdomainAssignment::BoxBasedRandomSubdomainAssignment: The requested fraction of excitatory neurons is not in [0.0, 1.0]: {}", fraction_excitatory_neurons);
        RelearnException::check(um_per_neuron > 0.0, "BoxBasedRandomSubdomainAssignment::BoxBasedRandomSubdomainAssignment: The requested um per neuron is <= 0.0: {}", um_per_neuron);

        set_requested_ratio_excitatory_neurons(fraction_excitatory_neurons);
    }

    BoxBasedRandomSubdomainAssignment(const BoxBasedRandomSubdomainAssignment& other) = delete;
    BoxBasedRandomSubdomainAssignment(BoxBasedRandomSubdomainAssignment&& other) = delete;

    BoxBasedRandomSubdomainAssignment& operator=(const BoxBasedRandomSubdomainAssignment& other) = delete;
    BoxBasedRandomSubdomainAssignment& operator=(BoxBasedRandomSubdomainAssignment&& other) = delete;

    ~BoxBasedRandomSubdomainAssignment() override = default;

    /**
     * @brief Returns a function object that is used to fix calculated subdomain boundaries.
     *      It rounds the boundaries up to the next multiple of um_per_neuron
     * @return A function object that corrects subdomain boundaries
     */
    [[nodiscard]] std::function<box_size_type(box_size_type)> get_subdomain_boundary_fix() const override {
        auto lambda = [multiple = um_per_neuron_](box_size_type arg) -> box_size_type {
            arg.round_to_larger_multiple(multiple);
            return arg;
        };

        return lambda;
    }

    /**
     * @brief Returns the micrometer per neuron box
     * @return The micrometer per neuron box
     */
    [[nodiscard]] double get_um_per_neuron() const noexcept {
        return um_per_neuron_;
    }

    std::pair<std::vector<LoadedNeuron>, NeuronID::value_type> place_neurons_in_box(const box_size_type& min, const box_size_type& max, NeuronID::value_type number_neurons, NeuronID::value_type first_id);

private:
    const double um_per_neuron_{}; // Micrometer per neuron in one dimension
};