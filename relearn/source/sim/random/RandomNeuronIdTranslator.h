/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "../NeuronIdTranslator.h"
#include "../../Config.h"
#include "../../neurons/helper/RankNeuronId.h"

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
    RandomNeuronIdTranslator(std::shared_ptr<Partition> partition, std::function<std::vector<size_t>(size_t)> gather_function);

    /**
     * @brief Checks if a global_id belongs to the current MPI rank
     * @param global_id The global neuron id 
     * @return True iff the global neuron id belongs to the current MPI rank
     */
    [[nodiscard]] bool is_neuron_local(size_t global_id) const override;

    /**
     * @brief Translated the global neuron id to the local neuron id
     * @param global_id The global neuron id
     * @return The local neuron id
     */
    [[nodiscard]] size_t get_local_id(size_t global_id) const override;

    /**
     * @brief Translated the local neuron id to the global neuron id
     * @param global_id The local neuron id
     * @return The global neuron id
     */
    [[nodiscard]] size_t get_global_id(size_t local_id) const override;

    /**
     * @brief Translated a bunch of global neuron ids to RankNeuronIds
     * @param global_ids The global neuron ids
     * @return A translation map from global neuron id to RankNeuronId
     */
    [[nodiscard]] std::map<neuron_id, RankNeuronId> translate_global_ids(const std::vector<neuron_id>& global_ids) override;

    /**
     * @brief Translated a RankNeuronId to a global neuron id 
     * @param rni The rank neuron id
     * @return The global neuron id of that neuron
     */
    [[nodiscard]] neuron_id translate_rank_neuron_id(const RankNeuronId& rni) override;

    /**
     * @brief Translated a bunch of RankNeuronIds to global neuron ids
     * @param ids The RankNeuronIds
     * @return A translation map from RankNeuronId to global neuron id
     */
    [[nodiscard]] std::map<RankNeuronId, neuron_id> translate_rank_neuron_ids(const std::vector<RankNeuronId>& ids) override;
};
