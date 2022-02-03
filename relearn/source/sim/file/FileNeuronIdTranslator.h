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
#include "../../Types.h"
#include "../../neurons/helper/RankNeuronId.h"
#include "../../util/Vec3.h"

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

    std::vector<std::vector<size_t>> global_neuron_ids{};

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
     * @brief Sets the global ids for the subdomain
     * @param subdomain_idx The subdomain index
     * @param global_ids The global ids (indexed by the local ids)
     */
    void set_global_ids(std::vector<std::vector<size_t>> global_ids) {
        global_neuron_ids = std::move(global_ids);
    }

    /**
     * @brief Checks if a global_id belongs to the current MPI rank
     * @param global_id The global neuron id 
     * @return True iff the global neuron id belongs to the current MPI rank
     */
    [[nodiscard]] virtual bool is_neuron_local(size_t global_id) const;

    /**
     * @brief Translated the global neuron id to the local neuron id
     * @param global_id The global neuron id
     * @return The local neuron id
     */
    [[nodiscard]] virtual size_t get_local_id(size_t global_id) const;

    /**
     * @brief Translated the local neuron id to the global neuron id
     * @param global_id The local neuron id
     * @return The global neuron id
     */
    [[nodiscard]] virtual size_t get_global_id(size_t local_id) const;

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

private:
    std::map<neuron_id, position_type> load_neuron_positions(const std::vector<neuron_id>& global_ids);
};
