/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#pragma once

#include "NeuronToSubdomainAssignment.h"

#include <tuple>
#include <vector>

// This class fills every subdomain with neurons at
// random positions. The size of the simulation box and the number of neurons per
// subdomain depend on the requested neuron density, i.e., micrometer per neuron
// in each of the three dimensions.
class SubdomainFromNeuronDensity : public NeuronToSubdomainAssignment {
public:
    SubdomainFromNeuronDensity(size_t num_neurons, double desired_frac_neurons_exc, double um_per_neuron);

    SubdomainFromNeuronDensity(const SubdomainFromNeuronDensity& other) = delete;
    SubdomainFromNeuronDensity(SubdomainFromNeuronDensity&& other) = delete;

    SubdomainFromNeuronDensity& operator=(const SubdomainFromNeuronDensity& other) = delete;
    SubdomainFromNeuronDensity& operator=(SubdomainFromNeuronDensity&& other) = delete;

    ~SubdomainFromNeuronDensity() override = default;

    [[nodiscard]] std::tuple<Position, Position> get_subdomain_boundaries(const Vec3s& subdomain_3idx, size_t num_subdomains_per_axis) const noexcept override;

    [[nodiscard]] std::tuple<Position, Position> get_subdomain_boundaries(const Vec3s& subdomain_3idx, const Vec3s& num_subdomains_per_axis) const noexcept override;

    void fill_subdomain(size_t subdomain_idx, size_t num_subdomains, const Position& min, const Position& max) override;

    [[nodiscard]] std::vector<size_t> neuron_global_ids(size_t subdomain_idx, size_t num_subdomains,
        size_t local_id_start, size_t local_id_end) const override;

    constexpr static double default_um_per_neuron = 26;

private:
    const double um_per_neuron_{ 0.0 }; // Micrometer per neuron in one dimension

    void place_neurons_in_area(
        const NeuronToSubdomainAssignment::Position& offset,
        const NeuronToSubdomainAssignment::Position& length_of_box,
        size_t num_neurons, size_t subdomain_idx);
};
