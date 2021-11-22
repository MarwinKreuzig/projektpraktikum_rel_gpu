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

class NeuronIdTranslator {
protected:
    using neuron_id = size_t;
    using position_type = RelearnTypes::position_type;

    std::shared_ptr<Partition> partition{};

    size_t number_local_neurons{ Constants::uninitialized };

public:
    NeuronIdTranslator(std::shared_ptr<Partition> partition, size_t number_local_neurons);

    virtual std::map<neuron_id, RankNeuronId> translate_global_ids(const std::vector<neuron_id>& global_ids) = 0;

    virtual neuron_id translate_rank_neuron_id(const RankNeuronId& rni) = 0;

    virtual std::map<RankNeuronId, neuron_id> translate_rank_neuron_ids(const std::vector<RankNeuronId>& ids) = 0;

    virtual ~NeuronIdTranslator() = default;
};

class FileNeuronIdTranslator : public NeuronIdTranslator {
protected:
    std::filesystem::path path_to_neurons{};

public:
    FileNeuronIdTranslator(std::shared_ptr<Partition> partition, size_t number_local_neurons, std::filesystem::path path_to_neurons)
        : NeuronIdTranslator(std::move(partition), number_local_neurons)
        , path_to_neurons(std::move(path_to_neurons)) { }

    std::map<neuron_id, RankNeuronId> translate_global_ids(const std::vector<neuron_id>& global_ids) override;

    neuron_id translate_rank_neuron_id(const RankNeuronId& rni) override;

    std::map<RankNeuronId, neuron_id> translate_rank_neuron_ids(const std::vector<RankNeuronId>& ids) override;

private:
    std::map<neuron_id, position_type> load_neuron_positions(const std::vector<neuron_id>& global_ids);
};

class RandomNeuronIdTranslator : public NeuronIdTranslator {
    std::vector<size_t> mpi_rank_to_local_start_id{};

public:
    RandomNeuronIdTranslator(std::shared_ptr<Partition> partition, size_t number_local_neurons);

    std::map<neuron_id, RankNeuronId> translate_global_ids(const std::vector<neuron_id>& global_ids) override;

    neuron_id translate_rank_neuron_id(const RankNeuronId& rni) override;

    std::map<RankNeuronId, neuron_id> translate_rank_neuron_ids(const std::vector<RankNeuronId>& ids) override;
};
