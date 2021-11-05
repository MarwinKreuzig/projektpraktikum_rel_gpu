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

#include "../Config.h"
#include "../sim/NeuronToSubdomainAssignment.h"

#include <fstream>
#include <optional>
#include <string>
#include <vector>

class Partition;

/**
 * This class inherits form NeuronToSubdomainAssignment. It reads the neurons with their positions from a file and, 
 * based on this, determines the size of the simulation box and the number of neurons in every individual subdomain.
 */
class SubdomainFromFile : public NeuronToSubdomainAssignment {
public:
    /**
     * @brief Constructs a new object and reads the specified file to determine the simulation box' size.
     *      Does not load any neurons.
     * @param file_path The path to the file to load
     * @exception Throws a RelearnException if there occurred some erros while processing the file 
     */
    explicit SubdomainFromFile(const std::string& file_path);

    SubdomainFromFile(const SubdomainFromFile& other) = delete;
    SubdomainFromFile(SubdomainFromFile&& other) = delete;

    SubdomainFromFile& operator=(const SubdomainFromFile& other) = delete;
    SubdomainFromFile& operator=(SubdomainFromFile&& other) = delete;

    ~SubdomainFromFile() override = default;

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
     * @brief Reads all neuron ids from a file and returns those.
     *      The file must be ascendingly sorted
     * @param file_path The path to the file to load
     * @return Empty if the file did not meet the sorting requirement, the ascending ids otherwise
     */
    [[nodiscard]] static std::optional<std::vector<size_t>> read_neuron_ids_from_file(const std::string& file_path);

    /**
     * @brief Returns the global ids for a given subdomain and local start and end ids
     * @param subdomain_idx The 1d index of the subdomain which's neurons are to be filled
     * @param num_subdomains The total number of subdomains
     * @param local_id_start The first local id
     * @param local_id_end The last local id
     * @exception Throws a RelearnException if the subdomain is not loaded
     * @return The global ids for the specified subdomain
     */
    [[nodiscard]] std::vector<size_t> neuron_global_ids(const size_t subdomain_idx, const size_t num_subdomains,
        const size_t local_id_start, const size_t local_id_end) const override;

    /**
     * @brief Returns the number of neurons in the associated file
     * @return The number of neurons in the associated file
     */
    [[nodiscard]] size_t get_total_num_neurons_in_file() const noexcept {
        return total_num_neurons_in_file;
    }

private:
    void read_dimensions_from_file();

    [[nodiscard]] std::vector<NeuronToSubdomainAssignment::Node> read_nodes_from_file(const box_size_type& min, const box_size_type& max);

    std::ifstream file{};
    size_t total_num_neurons_in_file{ Constants::uninitialized };
};
