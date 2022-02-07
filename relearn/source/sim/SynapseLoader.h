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

#include "../Types.h"

#include <memory>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

class NeuronIdTranslator;
class Partition;

/**
 * SynapseLoader is a type that abstracts away the mechanics of how synapses are loaded.
 * It provides load_synapses and relies on NeuronIdTranslator.
 */
class SynapseLoader {
public:
    using LocalSynapse = std::tuple<NeuronID, NeuronID, int>;
    using InSynapse = std::tuple<RankNeuronId, NeuronID, int>;
    using OutSynapse = std::tuple<NeuronID, RankNeuronId, int>;

protected:
    using source_neuron_id = NeuronID;
    using target_neuron_id = NeuronID;

    using synapse_weight = int;

    using synapse_type = std::tuple<source_neuron_id, target_neuron_id, synapse_weight>;

    using synapses_type = std::vector<synapse_type>;

    using local_synapses_type = synapses_type;
    using in_synapses_type = synapses_type;
    using out_synapses_type = synapses_type;

    using synapses_tuple_type = std::tuple<local_synapses_type, in_synapses_type, out_synapses_type>;

    std::shared_ptr<Partition> partition{};
    std::shared_ptr<NeuronIdTranslator> nit{};

    virtual std::pair<synapses_tuple_type, std::vector<NeuronID>> internal_load_synapses() = 0;

public:
    /**
     * @brief Constructs a SynapseLoader with the given Partition and NeuronIdTranslator
     * @param partition The partition to use
     * @param neuron_id_translator The neuron id translator that is used to determine if neuron ids are local
     */
    SynapseLoader(std::shared_ptr<Partition> partition, std::shared_ptr<NeuronIdTranslator> neuron_id_translator)
        : partition(std::move(partition))
        , nit(std::move(neuron_id_translator)) { }

    SynapseLoader(const SynapseLoader&) = default;

    SynapseLoader& operator=(const SynapseLoader&) = default;

    SynapseLoader(SynapseLoader&&) = default;

    SynapseLoader& operator=(SynapseLoader&&) = default;

    virtual ~SynapseLoader() = default;

    /**
     * @brief Loads all synapses that affect the local neurons, which are
     *      (1) local synapses (local neuron to local neuron)
     *      (2) in synapses (non-local neuron to local neuron)
     *      (3) out synpases (local neuron to non-local neuron)
     * @return A tuple of (local, in, out) synapes
     */
    std::tuple<LocalSynapses, DistantInSynapses, DistantOutSynapses> load_synapses();

    virtual ~SynapseLoader()
        = default;
};
