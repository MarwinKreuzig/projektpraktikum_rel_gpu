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
public:
    using neuron_id = size_t;
    using position_type = RelearnTypes::position_type;

protected:
    std::shared_ptr<Partition> partition{};

    std::vector<std::vector<size_t>> global_neuron_ids{};

public:
    explicit NeuronIdTranslator(std::shared_ptr<Partition> partition);

    void set_global_ids(size_t subdomain_idx, std::vector<size_t> global_ids) {
        global_neuron_ids[subdomain_idx] = std::move(global_ids);
    }

    bool is_neuron_local(size_t neuron_id) const;

    [[nodiscard]] size_t get_local_id(size_t global_id) const;

    [[nodiscard]] size_t get_global_id(size_t local_id) const;

    virtual std::map<neuron_id, RankNeuronId> translate_global_ids(const std::vector<neuron_id>& global_ids) = 0;

    virtual neuron_id translate_rank_neuron_id(const RankNeuronId& rni) = 0;

    virtual std::map<RankNeuronId, neuron_id> translate_rank_neuron_ids(const std::vector<RankNeuronId>& ids) = 0;

    virtual ~NeuronIdTranslator() = default;
};

class FileNeuronIdTranslator : public NeuronIdTranslator {
protected:
    std::filesystem::path path_to_neurons{};

public:
    FileNeuronIdTranslator(std::shared_ptr<Partition> partition, std::filesystem::path path_to_neurons)
        : NeuronIdTranslator(std::move(partition))
        , path_to_neurons(std::move(path_to_neurons)) { }

    std::map<neuron_id, RankNeuronId> translate_global_ids(const std::vector<neuron_id>& global_ids) override;

    neuron_id translate_rank_neuron_id(const RankNeuronId& rni) override;

    std::map<RankNeuronId, neuron_id> translate_rank_neuron_ids(const std::vector<RankNeuronId>& ids) override;

private:
    std::map<neuron_id, position_type> load_neuron_positions(const std::vector<neuron_id>& global_ids);
};

class RandomNeuronIdTranslator : public NeuronIdTranslator {
    std::vector<size_t> mpi_rank_to_local_start_id{};

    size_t number_local_neurons{ Constants::uninitialized };

public:
    RandomNeuronIdTranslator(std::shared_ptr<Partition> partition, size_t number_local_neurons);

    std::map<neuron_id, RankNeuronId> translate_global_ids(const std::vector<neuron_id>& global_ids) override;

    neuron_id translate_rank_neuron_id(const RankNeuronId& rni) override;

    std::map<RankNeuronId, neuron_id> translate_rank_neuron_ids(const std::vector<RankNeuronId>& ids) override;
};
