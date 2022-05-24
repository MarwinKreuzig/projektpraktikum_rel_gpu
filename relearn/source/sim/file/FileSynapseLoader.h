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
#include "util/RelearnException.h"

#include <filesystem>
#include <memory>
#include <optional>
#include <utility>

class NeuronIdTranslator;
class Partition;

class FileSynapseLoader : public SynapseLoader {
    std::optional<std::filesystem::path> optional_path_to_file{};

protected:
    synapses_tuple_type internal_load_synapses() override;

public:
    /**
     * @brief Constructs a FileSynapseLoader with the given Partition and NeuronIdTranslator.
     *      Can load synapses from a file
     * @param partition The partition to use
     * @param neuron_id_translator The neuron id translator that is used to determine if neuron ids are local
     * @param path_to_synapses The path to the synapses, can be empty
     */
    FileSynapseLoader(std::shared_ptr<Partition> partition, std::optional<std::filesystem::path> path_to_synapses);
};

