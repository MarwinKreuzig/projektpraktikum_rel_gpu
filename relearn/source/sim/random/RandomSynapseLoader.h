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

#include "sim/SynapseLoader.h"

#include <memory>
#include <utility>
#include <vector>

class NeuronIdTranslator;
class Partition;

class RandomSynapseLoader : public SynapseLoader {
protected:
    std::pair<synapses_tuple_type, std::vector<NeuronID>> internal_load_synapses() override {
        return std::pair<synapses_tuple_type, std::vector<NeuronID>>();
    }

public:
    /**
     * @brief Constructs a RandomSynapseLoader with the given Partition and NeuronIdTranslator.
     *      Does not provide any synapses
     * @param partition The partition to use
     * @param neuron_id_translator The neuron id translator that is used to determine if neuron ids are local
     */
    RandomSynapseLoader(std::shared_ptr<Partition> partition, std::shared_ptr<NeuronIdTranslator> neuron_id_translator)
        : SynapseLoader(std::move(partition), std::move(neuron_id_translator)) { }
};