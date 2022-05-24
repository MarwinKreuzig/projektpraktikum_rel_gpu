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
#include "Types.h"
#include "sim/NeuronIdTranslator.h"

#include <filesystem>
#include <map>
#include <memory>
#include <vector>

class Partition;

/**
 * This class offers the translation based on a path to the neuron files.
 * It inherits from NeuronIdTranslator.
 */
class FileNeuronIdTranslator : public NeuronIdTranslator {
protected:
    std::filesystem::path path_to_neurons{};
    std::shared_ptr<Partition> partition{};

    std::vector<std::vector<NeuronID>> global_neuron_ids{};

public:
    using position_type = RelearnTypes::position_type;

    /**
     * @brief Constructs a translator based on the partition and the provided file to the neurons
     * @param partition The Partition object 
     * @param path_to_neurons The path to the file in which the neurons are stored
     */
    FileNeuronIdTranslator(std::shared_ptr<Partition> partition, std::filesystem::path path_to_neurons)
        : partition(std::move(partition))
        , path_to_neurons(std::move(path_to_neurons)) { }

    /**
     * @brief Sets the global ids for the subdomainglobal_neuron_ids
     * @param subdomain_idx The subdomain index
     * @param global_ids The global ids (indexed by the local ids)
     */
    void set_global_ids(std::vector<std::vector<NeuronID>> global_ids) {
        global_neuron_ids = std::move(global_ids);
    }

    /**
     * @brief Checks if a global_id belongs to the current MPI rank
     * @param global_id The global neuron id 
     * @return True iff the global neuron id belongs to the current MPI rank
     */
    [[nodiscard]] bool is_neuron_local(NeuronID global_id) const override;

    /**
     * @brief Translated the global neuron id to the local neuron id
     * @param global_id The global neuron id
     * @return The local neuron id
     */
    [[nodiscard]] NeuronID get_local_id(NeuronID global_id) const override;

    /**
     * @brief Translated the local neuron id to the global neuron id
     * @param global_id The local neuron id
     * @return The global neuron id
     */
    [[nodiscard]] NeuronID get_global_id(NeuronID local_id) const override;
   
    /**
     * @brief Creates the number of neurons locally, i.e., every MPI rank must specify its own number of creations.
     *      Might throw a RelearnException if this is not supported.
     * @param number_local_creations The number of new neurons on this rank. 
     */
    void create_neurons(size_t number_local_creations) override;

private:
    std::map<NeuronID, position_type> load_neuron_positions(const std::vector<NeuronID>& global_ids);
};
