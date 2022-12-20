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

#include <filesystem>
#include <memory>
#include <optional>

class Partition;

/**
 * This class inherits form NeuronToSubdomainAssignment. It reads the neurons with their positions from a file and,
 * based on this, determines the size of the simulation box and the number of neurons in every individual subdomain.
 */
class SubdomainFromFile : public NeuronToSubdomainAssignment {
public:
    /**
     * @brief Constructs a new object and reads the specified file to determine the simulation box' size.
     *      Loads all neurons in the file to prevent reading it twice.
     * @param path_to_neurons The path to the file with the neurons to load
     * @param path_to_synapses The file path to the synapses, can be empty if none should be loaded
     * @param partition The partition
     * @exception Throws a RelearnException if some errors occurred while processing the file
     */
    SubdomainFromFile(const std::filesystem::path& path_to_neurons, std::optional<std::filesystem::path> path_to_synapses, std::shared_ptr<Partition> partition);

    SubdomainFromFile(const SubdomainFromFile& other) = delete;
    SubdomainFromFile(SubdomainFromFile&& other) = delete;

    SubdomainFromFile& operator=(const SubdomainFromFile& other) = delete;
    SubdomainFromFile& operator=(SubdomainFromFile&& other) = delete;

    ~SubdomainFromFile() override = default;

    /**
     * @brief Prints relevant metrics to the essentials
     * @param essentials The essentials
     */
    void print_essentials(const std::unique_ptr<Essentials>& essentials) override;

protected:
    void fill_all_subdomains() override {
        // This method is empty as the loading actually happens in the constructor
    }

private:
    void read_neurons_from_file(const std::filesystem::path& path_to_neurons);
};
