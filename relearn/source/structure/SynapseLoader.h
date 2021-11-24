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

#include <filesystem>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

class Partition;

class SynapseLoader {
protected:
    std::shared_ptr<Partition> partition{};

public:
    using neuron_id = size_t;

    using source_neuron_id = neuron_id;
    using target_neuron_id = neuron_id;

    using synapse_weight = int;

    using synapse_type = std::tuple<source_neuron_id, target_neuron_id, synapse_weight>;

    using synapses_type = std::vector<synapse_type>;

    using local_synapses_type = synapses_type;
    using in_synapses_type = synapses_type;
    using out_synapses_type = synapses_type;

    using synapses_tuple_type = std::tuple<local_synapses_type, in_synapses_type, out_synapses_type>;

    explicit SynapseLoader(std::shared_ptr<Partition> partition)
        : partition(std::move(partition)) { }

    virtual std::pair<synapses_tuple_type, std::vector<neuron_id>> load_synapses(const std::vector<neuron_id>& affected_neuron_ids) = 0;

    virtual ~SynapseLoader() = default;
};

class FileSynapseLoader : public SynapseLoader {
    std::optional<std::filesystem::path> optional_path_to_file{};

public:
    FileSynapseLoader(std::shared_ptr<Partition> partition, const std::optional<std::filesystem::path>& path_to_synapses)
        : SynapseLoader(std::move(partition))
        , optional_path_to_file(path_to_synapses) { }

    std::pair<synapses_tuple_type, std::vector<neuron_id>> load_synapses(const std::vector<neuron_id>& affected_neuron_ids) override;
};

class RandomSynapseLoader : public SynapseLoader {
    explicit RandomSynapseLoader(std::shared_ptr<Partition> partition)
        : SynapseLoader(std::move(partition)) { }

    std::pair<synapses_tuple_type, std::vector<neuron_id>> load_synapses(const std::vector<neuron_id>& affected_neuron_ids) {
        return std::pair<synapses_tuple_type, std::vector<neuron_id>>();
    }
};
