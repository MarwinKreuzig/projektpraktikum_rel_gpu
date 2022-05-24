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
#include "sim/SynapseLoader.h"

#include <filesystem>
#include <fstream>
#include <memory>
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
    SubdomainFromFile(const std::filesystem::path& file_path, std::optional<std::filesystem::path> file_path_positions, std::shared_ptr<Partition> partition);

    SubdomainFromFile(const SubdomainFromFile& other) = delete;
    SubdomainFromFile(SubdomainFromFile&& other) = delete;

    SubdomainFromFile& operator=(const SubdomainFromFile& other) = delete;
    SubdomainFromFile& operator=(SubdomainFromFile&& other) = delete;

    ~SubdomainFromFile() override = default;

    /**
     * @brief Reads all neuron ids from a file and returns those.
     *      The file must be ascendingly sorted
     * @param file_path The path to the file to load
     * @return Empty if the file did not meet the sorting requirement, the ascending ids otherwise
     */
    [[nodiscard]] static std::optional<std::vector<NeuronID>> read_neuron_ids_from_file(const std::filesystem::path& file_path);

    /**
     * @brief Returns the global ids for a given subdomain
     * @param subdomain_index_1d The 1d index of the subdomain which is inquired
     * @param total_number_subdomains The total number of subdomains
     * @exception Throws a RelearnException if the subdomain is not loaded
     * @return The global ids for the specified subdomain
     */
    [[nodiscard]] std::vector<NeuronID> get_neuron_global_ids_in_subdomain(size_t subdomain_index_1d, size_t total_number_subdomains) const override;

protected:
    /**
     * @brief Fills the subdomain with the given index and the boundaries. Reads the whole file to determine the which neuron fall into the specified box
     * @param local_subdomain_index The local index of the subdomain which's neurons are to be filled
     * @param total_number_subdomains The total number of subdomains
     * @exception Throws a RelearnException if the subdomain is already loaded or if some erros while processing the file
     */
    void fill_subdomain(size_t local_subdomain_index, size_t total_number_subdomains) override;

    void post_initialization() override;

    void calculate_total_number_neurons() const override {
        set_total_number_placed_neurons(total_num_neurons_in_file);
    }

private:
    void read_dimensions_from_file();

    [[nodiscard]] std::vector<NeuronToSubdomainAssignment::Node> read_nodes_from_file(const box_size_type& min, const box_size_type& max);

    std::filesystem::path path{};
    size_t total_num_neurons_in_file{ Constants::uninitialized };
};
