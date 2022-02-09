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

#include "../NeuronIdTranslator.h"
#include "../NeuronToSubdomainAssignment.h"
#include "../SynapseLoader.h"

#include <memory>
#include <tuple>
#include <vector>

class Partition;

/**
 * This class fills every subdomain with neurons at random positions. The size of the simulation box and the number of neurons per
 * subdomain depend on the requested neuron density, i.e., micrometer per neuron in each of the three dimensions.
 * It does not necessarily place the requested number of neurons, but it always places a number of neurons in
 * [number_neurons - number_subdomains + 1, number_neurons + number_subdomains - 1].
 * It inherits from NeuronToSubdomainAssignment.
 */
class SubdomainFromNeuronDensity : public NeuronToSubdomainAssignment {
public:
    /**
     * @brief Constructs a new object with the specified parameters
     * @param number_neurons The number of neurons
     * @param fraction_excitatory_neurons The fraction of excitatory neurons, must be in [0.0, 1.0]
     * @param um_per_neuron The box length in which a single neuron is placed, must be > 0.0
     * @exception Throws a RelearnException if fraction_excitatory_neurons if not from [0.0, 1.0] or um_per_neuron <= 0.0
     */
    SubdomainFromNeuronDensity(size_t number_neurons, double fraction_excitatory_neurons, double um_per_neuron, std::shared_ptr<Partition> partition);

    SubdomainFromNeuronDensity(const SubdomainFromNeuronDensity& other) = delete;
    SubdomainFromNeuronDensity(SubdomainFromNeuronDensity&& other) = delete;

    SubdomainFromNeuronDensity& operator=(const SubdomainFromNeuronDensity& other) = delete;
    SubdomainFromNeuronDensity& operator=(SubdomainFromNeuronDensity&& other) = delete;

    ~SubdomainFromNeuronDensity() override = default;

    /**
     * @brief This method is not implemented for this class
     */
    [[nodiscard]] std::vector<NeuronID> get_neuron_global_ids_in_subdomain(size_t subdomain_index_1d, size_t total_number_subdomains) const override;

    /**
     * @brief Returns a function object that is used to fix calculated subdomain boundaries.
     *      It rounds the boundaries up to the next multiple of um_per_neuron
     * @return A function object that corrects subdomain boundaries
     */
    std::function<Vec3d(Vec3d)> get_subdomain_boundary_fix() const override {
        auto lambda = [multiple = um_per_neuron_](Vec3d arg) -> Vec3d {
            arg.round_to_larger_multiple(multiple);
            return arg;
        };

        return lambda;
    }

    constexpr static double default_um_per_neuron = 26.0;

protected:
    void post_initialization() override;

    /**
     * @brief Fills the subdomain with the given index and the boundaries. Reads the whole file to determine the which neuron fall into the specified box
     * @param local_subdomain_index The local index of the subdomain which's neurons are to be filled
     * @param total_number_subdomains The total number of local_subdomains
     * @exception Throws a RelearnException if the subdomain is already loaded or if some erros while processing the file
     */
    void fill_subdomain(size_t local_subdomain_index, size_t total_number_subdomains) override;

    void calculate_total_number_neurons() const override;

private:
    const double um_per_neuron_{ default_um_per_neuron }; // Micrometer per neuron in one dimension

    void place_neurons_in_area(
        const NeuronToSubdomainAssignment::box_size_type& offset,
        const NeuronToSubdomainAssignment::box_size_type& length_of_box,
        size_t number_neurons, size_t subdomain_index_1d);
};
