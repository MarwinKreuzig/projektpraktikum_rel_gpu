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
#include "../neurons/helper/RankNeuronId.h"
#include "../util/Vec3.h"

#include <filesystem>
#include <map>
#include <memory>
#include <vector>

class Partition;

/**
 * This class offers a way to translate neuron ids:
 * Local id to global id and vice versa, also to and from RankNeuronId
 */
class NeuronIdTranslator {
public:
    using neuron_id = size_t;
    using position_type = RelearnTypes::position_type;

protected:
    std::shared_ptr<Partition> partition{};

    std::vector<std::vector<size_t>> global_neuron_ids{};

public:
    /**
     * @brief Creates a new translator with the associated partition
     * @param partition The Partition object
    */
    explicit NeuronIdTranslator(std::shared_ptr<Partition> partition);

    /**
     * @brief Sets the global ids for the subdomain
     * @param subdomain_idx The subdomain index
     * @param global_ids The global ids (indexed by the local ids)
     */
    void set_global_ids(size_t subdomain_idx, std::vector<size_t> global_ids) {
        global_neuron_ids[subdomain_idx] = std::move(global_ids);
    }

    /**
     * @brief Checks if a global_id belongs to the current MPI rank
     * @param global_id The global neuron id 
     * @return True iff the global neuron id belongs to the current MPI rank
     */
    bool is_neuron_local(size_t global_id) const;

    /**
     * @brief Translated the global neuron id to the local neuron id
     * @param global_id The global neuron id
     * @return The local neuron id
     */
    [[nodiscard]] size_t get_local_id(size_t global_id) const;

    /**
     * @brief Translated the local neuron id to the global neuron id
     * @param global_id The local neuron id
     * @return The global neuron id
     */
    [[nodiscard]] size_t get_global_id(size_t local_id) const;

    /**
     * @brief Translated a bunch of global neuron ids to RankNeuronIds
     * @param global_ids The global neuron ids
     * @return A translation map from global neuron id to RankNeuronId
     */
    virtual std::map<neuron_id, RankNeuronId> translate_global_ids(const std::vector<neuron_id>& global_ids) = 0;

    /**
     * @brief Translated a RankNeuronId to a global neuron id 
     * @param rni The rank neuron id
     * @return The global neuron id of that neuron
     */
    virtual neuron_id translate_rank_neuron_id(const RankNeuronId& rni) = 0;

    /**
     * @brief Translated a bunch of RankNeuronIds to global neuron ids
     * @param ids The RankNeuronIds
     * @return A translation map from RankNeuronId to global neuron id
     */
    virtual std::map<RankNeuronId, neuron_id> translate_rank_neuron_ids(const std::vector<RankNeuronId>& ids) = 0;

    virtual ~NeuronIdTranslator() = default;
};

/**
 * This class offers the translation based on a path to the neuron files.
 * It inherits from NeuronIdTranslator.
 */
class FileNeuronIdTranslator : public NeuronIdTranslator {
protected:
    std::filesystem::path path_to_neurons{};

public:
    /**
     * @brief Constructs a translator based on the partition and the provided file to the neurons
     * @param partition The Partition object 
     * @param path_to_neurons The path to the file in which the neurons are stored
     */
    FileNeuronIdTranslator(std::shared_ptr<Partition> partition, std::filesystem::path path_to_neurons)
        : NeuronIdTranslator(std::move(partition))
        , path_to_neurons(std::move(path_to_neurons)) { }

    /**
     * @brief Translated a bunch of global neuron ids to RankNeuronIds
     * @param global_ids The global neuron ids
     * @return A translation map from global neuron id to RankNeuronId
     */
    std::map<neuron_id, RankNeuronId> translate_global_ids(const std::vector<neuron_id>& global_ids) override;

    /**
     * @brief Translated a RankNeuronId to a global neuron id 
     * @param rni The rank neuron id
     * @return The global neuron id of that neuron
     */
    neuron_id translate_rank_neuron_id(const RankNeuronId& rni) override;

    /**
     * @brief Translated a bunch of RankNeuronIds to global neuron ids
     * @param ids The RankNeuronIds
     * @return A translation map from RankNeuronId to global neuron id
     */
    std::map<RankNeuronId, neuron_id> translate_rank_neuron_ids(const std::vector<RankNeuronId>& ids) override;

private:
    std::map<neuron_id, position_type> load_neuron_positions(const std::vector<neuron_id>& global_ids);
};

/**
 * This class offers the translation based on the local ids of each rank,
 * i.e., in enumerates all neurons by their subdomain id and then their local id.
 * It inherits from NeuronIdTranslator.
 */
class RandomNeuronIdTranslator : public NeuronIdTranslator {
    std::vector<size_t> mpi_rank_to_local_start_id{};

    size_t number_local_neurons{ Constants::uninitialized };

public:
    /**
     * @brief Constructs a translator based on the partition and the provided file to the neurons
     * @param partition The Partition object 
     */
    explicit RandomNeuronIdTranslator(std::shared_ptr<Partition> partition);

    /**
     * @brief Translated a bunch of global neuron ids to RankNeuronIds
     * @param global_ids The global neuron ids
     * @return A translation map from global neuron id to RankNeuronId
     */
    std::map<neuron_id, RankNeuronId> translate_global_ids(const std::vector<neuron_id>& global_ids) override;

    /**
     * @brief Translated a RankNeuronId to a global neuron id 
     * @param rni The rank neuron id
     * @return The global neuron id of that neuron
     */
    neuron_id translate_rank_neuron_id(const RankNeuronId& rni) override;

    /**
     * @brief Translated a bunch of RankNeuronIds to global neuron ids
     * @param ids The RankNeuronIds
     * @return A translation map from RankNeuronId to global neuron id
     */
    std::map<RankNeuronId, neuron_id> translate_rank_neuron_ids(const std::vector<RankNeuronId>& ids) override;
};
