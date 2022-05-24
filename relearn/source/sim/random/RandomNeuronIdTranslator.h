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
#include "neurons/helper/RankNeuronId.h"
#include "sim/NeuronIdTranslator.h"
#include "util/TaggedID.h"

#include <functional>
#include <map>
#include <memory>
#include <vector>

class Partition;

/**
 * This class offers the translation based on the local ids of each rank,
 * i.e., in enumerates all neurons by their subdomain id and then their local id.
 * It inherits from NeuronIdTranslator.
 */
class RandomNeuronIdTranslator : public NeuronIdTranslator {
    std::vector<size_t> mpi_rank_to_local_start_id{};
    std::shared_ptr<Partition> partition{};

    size_t number_local_neurons{ Constants::uninitialized };

public:
    /**
     * @brief Constructs a translator based on the partition and the provided file to the neurons
     * @param partition The Partition object 
     */
    explicit RandomNeuronIdTranslator(std::shared_ptr<Partition> partition);

    /**
     * @brief Constructs a translator based on the partition and the provided file to the neurons
     * @param partition The Partition object 
     * @param gather_function A function that maps the local number of neurons to a vector containing the local number of neurons per MPI rank
     */
    RandomNeuronIdTranslator(std::shared_ptr<Partition> partition, const std::function<std::vector<size_t>(size_t)>& gather_function);

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
};
