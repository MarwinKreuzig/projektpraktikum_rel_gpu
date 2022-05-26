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

#include "Types.h"

#include "sim/LoadedNeuron.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <tuple>
#include <vector>

/**
 * This class provides a static interface to load/store neurons and synapses from/to files,
 * as well as other linked information
 */
class NeuronIO {
public:
    using position_type = RelearnTypes::position_type;

    /** 
     * @brief Reads all neurons from the file and returns those.
     *      The file must be ascendingly sorted wrt. to the neuron ids (starting at 1).
     * @param file_path The path to the file to load
     * @return Returns a tuple with (1) all loaded neurons and (2) additional information
     */
    [[nodiscard]] static std::tuple<std::vector<LoadedNeuron>, LoadedNeuronsInfo> load_neurons_from_file(const std::filesystem::path& file_path);

    /**
     * @brief Reads all neuron ids from a file and returns those.
     *      The file must be ascendingly sorted wrt. to the neuron ids (starting at 1).
     * @param file_path The path to the file to load
     * @return Empty if the file did not meet the sorting requirement, the ascending ids otherwise
     */
    [[nodiscard]] static std::optional<std::vector<NeuronID>> read_neuron_ids_from_file(const std::filesystem::path& file_path);

    /**
     * @brief Reads all local synapses from a file and returns those.
     *      Checks that no id is larger or equal to number_local_neurons.
     * @param file_path The path to the file to load
     * @param number_local_neurons The number of local neurons
     * @return All empty synapses
     */
    [[nodiscard]] static LocalSynapses read_local_synapses_from_file(const std::filesystem::path& file_path, std::uint64_t number_local_neurons);
};