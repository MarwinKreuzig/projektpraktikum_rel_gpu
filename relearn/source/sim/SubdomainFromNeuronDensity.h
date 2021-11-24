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

#include "../structure/SynapseLoader.h"
#include "../structure/NeuronIdTranslator.h"
#include "NeuronToSubdomainAssignment.h"

#include <memory>
#include <tuple>
#include <vector>

class Partition;

/**
 * This class fills every subdomain with neurons at random positions. The size of the simulation box and the number of neurons per
 * subdomain depend on the requested neuron density, i.e., micrometer per neuron in each of the three dimensions. 
 * It inherits from NeuronToSubdomainAssignment.
 */
class SubdomainFromNeuronDensity : public NeuronToSubdomainAssignment {
public:
    /**
     * @brief Constructs a new object with the specified parameters
     * @param num_neurons The number of neurons
     * @param desired_frac_neurons_exc The fraction of excitatory neurons, must be in [0.0, 1.0]
     * @param um_per_neuron The box length in which a single neuron is placed, must be > 0.0
     * @exception Throws a RelearnException if desired_frac_neurons_exc if not from [0.0, 1.0] or um_per_neuron <= 0.0
     */
    SubdomainFromNeuronDensity(const size_t num_neurons, const double desired_frac_neurons_exc, const double um_per_neuron, std::shared_ptr<Partition> partition);

    SubdomainFromNeuronDensity(const SubdomainFromNeuronDensity& other) = delete;
    SubdomainFromNeuronDensity(SubdomainFromNeuronDensity&& other) = delete;

    SubdomainFromNeuronDensity& operator=(const SubdomainFromNeuronDensity& other) = delete;
    SubdomainFromNeuronDensity& operator=(SubdomainFromNeuronDensity&& other) = delete;

    ~SubdomainFromNeuronDensity() override = default;

    std::shared_ptr<SynapseLoader> get_synapse_loader() const noexcept override {
        return synapse_loader;
    }

    std::shared_ptr<NeuronIdTranslator> get_neuron_id_translator() const noexcept override {
        return neuron_id_translator;
    }

    /** 
     * @brief Returns the subdomain boundaries for a given subdomain
     * @param subdomain_3idx The 3d index of the subdomain which's boundaries are requested
     * @param num_subdomains_per_axis The number of subdomains per axis (the same for all dimensions), != 0
     * @return A tuple with (1) the minimum and (2) the maximum positions in the subdomain
     */
    [[nodiscard]] std::tuple<box_size_type, box_size_type> get_subdomain_boundaries(const Vec3s& subdomain_3idx, const size_t num_subdomains_per_axis) const noexcept override;

    /** 
     * @brief Returns the subdomain boundaries for a given subdomain
     * @param subdomain_3idx The 3d index of the subdomain which's boundaries are requested
     * @param num_subdomains_per_axis The number of subdomains per axis (can have varying number per dimension)
     * @return A tuple with (1) the minimum and (2) the maximum positions in the subdomain
     */
    [[nodiscard]] std::tuple<box_size_type, box_size_type> get_subdomain_boundaries(const Vec3s& subdomain_3idx, const Vec3s& num_subdomains_per_axis) const noexcept override;

    /**
     * @brief Fills the subdomain with the given index and the boundaries. Reads the whole file to determine the which neuron fall into the specified box
     * @param subdomain_idx The 1d index of the subdomain which's neurons are to be filled
     * @param num_subdomains The total number of subdomains
     * @param min The subdomain's minimum position
     * @param max The subdomain's maximum position
     * @exception Throws a RelearnException if the subdomain is already loaded or if some erros while processing the file 
     */
    void fill_subdomain(const size_t subdomain_idx, const size_t num_subdomains, const box_size_type& min, const box_size_type& max) override;

    /**
     * @brief This method is not implemented for this class
     */
    [[nodiscard]] std::vector<size_t> neuron_global_ids(const size_t subdomain_idx, const size_t num_subdomains) const override;

    constexpr static double default_um_per_neuron = 26.0;

private:
    std::shared_ptr<RandomSynapseLoader> synapse_loader{};
    std::shared_ptr<RandomNeuronIdTranslator> neuron_id_translator{};

    const double um_per_neuron_{ default_um_per_neuron }; // Micrometer per neuron in one dimension

    void place_neurons_in_area(
        const NeuronToSubdomainAssignment::box_size_type& offset,
        const NeuronToSubdomainAssignment::box_size_type& length_of_box,
        const size_t num_neurons, const size_t subdomain_idx);
};
