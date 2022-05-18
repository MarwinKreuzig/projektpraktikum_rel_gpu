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

#include "sim/NeuronIdTranslator.h"
#include "sim/NeuronToSubdomainAssignment.h"
#include "sim/SynapseLoader.h"

#include <memory>
#include <tuple>
#include <vector>

class Partition;

/**
 * This class fills every subdomain with neurons at random positions. It assigns a fixed number
 * of neurons to each MPI rank.
 * It inherits from NeuronToSubdomainAssignment.
 */
class SubdomainFromNeuronPerRank : public NeuronToSubdomainAssignment {
public:
    /**
     * @brief Constructs a new object with the specified parameters
     * @param number_neurons_per_rank The number of neurons per MPI rank
     * @param fraction_excitatory_neurons The fraction of excitatory neurons, must be in [0.0, 1.0]
     * @param um_per_neuron The box length in which a single neuron is placed, must be > 0.0
     * @exception Throws a RelearnException if fraction_excitatory_neurons if not from [0.0, 1.0] or um_per_neuron <= 0.0
     */
    SubdomainFromNeuronPerRank(size_t number_neurons_per_rank, double fraction_excitatory_neurons, double um_per_neuron, std::shared_ptr<Partition> partition);

    SubdomainFromNeuronPerRank(const SubdomainFromNeuronPerRank& other) = delete;
    SubdomainFromNeuronPerRank(SubdomainFromNeuronPerRank&& other) = delete;

    SubdomainFromNeuronPerRank& operator=(const SubdomainFromNeuronPerRank& other) = delete;
    SubdomainFromNeuronPerRank& operator=(SubdomainFromNeuronPerRank&& other) = delete;

    ~SubdomainFromNeuronPerRank() override = default;

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
    size_t number_neurons_per_rank{ Constants::uninitialized };

    const double um_per_neuron_{}; // Micrometer per neuron in one dimension

    void place_neurons_in_area(
        const NeuronToSubdomainAssignment::box_size_type& offset,
        const NeuronToSubdomainAssignment::box_size_type& length_of_box,
        size_t number_neurons, size_t subdomain_idx);
};
