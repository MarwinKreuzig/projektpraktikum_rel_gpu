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

#include "../neurons/helper/RankNeuronId.h"

#include <filesystem>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

class NeuronIdTranslator;
class Partition;

class SynapseLoader {

public:
    using LocalSynapse = std::tuple<size_t, size_t, int>;
    using InSynapse = std::tuple<RankNeuronId, size_t, int>;
    using OutSynapse = std::tuple<size_t, RankNeuronId, int>;

    using LocalSynapses = std::vector<LocalSynapse>;
    using InSynapses = std::vector<InSynapse>;
    using OutSynapses = std::vector<OutSynapse>;

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

protected:
    std::shared_ptr<Partition> partition{};
    std::shared_ptr<NeuronIdTranslator> nit{};

    virtual std::pair<synapses_tuple_type, std::vector<neuron_id>> internal_load_synapses() = 0;

public:
    SynapseLoader(std::shared_ptr<Partition> partition, std::shared_ptr<NeuronIdTranslator> neuron_id_translator)
        : partition(std::move(partition))
        , nit(std::move(neuron_id_translator)) { }

    std::tuple<LocalSynapses, InSynapses, OutSynapses> load_synapses();

    virtual ~SynapseLoader()
        = default;
};

class FileSynapseLoader : public SynapseLoader {
    std::optional<std::filesystem::path> optional_path_to_file{};

protected:
    std::pair<synapses_tuple_type, std::vector<neuron_id>> internal_load_synapses() override;

public:
    FileSynapseLoader(std::shared_ptr<Partition> partition, std::shared_ptr<NeuronIdTranslator> neuron_id_translator, const std::optional<std::filesystem::path>& path_to_synapses)
        : SynapseLoader(std::move(partition), std::move(neuron_id_translator))
        , optional_path_to_file(path_to_synapses) { }
};

class RandomSynapseLoader : public SynapseLoader {

protected:
    std::pair<synapses_tuple_type, std::vector<neuron_id>> internal_load_synapses() override {
        return std::pair<synapses_tuple_type, std::vector<neuron_id>>();
    }

public:
    RandomSynapseLoader(std::shared_ptr<Partition> partition, std::shared_ptr<NeuronIdTranslator> neuron_id_translator)
        : SynapseLoader(std::move(partition), std::move(neuron_id_translator)) { }
};
