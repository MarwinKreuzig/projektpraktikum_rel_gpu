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
     *      The file must be ascendingly sorted wrt. to the neuron ids (starting at 1). All positions must be non-negative
     * @param file_path The path to the file to load
     * @exception Throws a RelearnException if a position has a negative component or the ids are not sorted properly
     * @return Returns a tuple with (1) all loaded neurons and (2) additional information
     */
    [[nodiscard]] static std::tuple<std::vector<LoadedNeuron>, LoadedNeuronsInfo> read_neurons(const std::filesystem::path& file_path);

    /**
     * @brief Reads all neurons from the file and returns those in their components.
     *      The file must be ascendingly sorted wrt. to the neuron ids (starting at 1).
     * @param file_path The path to the file to load
     * @exception Throws a RelearnException if a position has a negative component or the ids are not sorted properly
     * @return Returns a tuple with
     *      (1) The IDs (which index (2)-(4))
     *      (2) The positions
     *      (3) The area names
     *      (4) The signal types
     *      (5) additional information
     */
    [[nodiscard]] static std::tuple<std::vector<NeuronID>, std::vector<position_type>, std::vector<std::string>, std::vector<SignalType>, LoadedNeuronsInfo>
    read_neurons_componentwise(const std::filesystem::path& file_path);

    /**
     * @brief Writes all neurons to the file. The IDs must start at 0 and be ascending.
     * @param neurons The neurons
     * @param file_path The path to the file
     * @exception Throws a RelearnException if opening the file failed
     */
    static void write_neurons(const std::vector<LoadedNeuron>& neurons, const std::filesystem::path& file_path);

    /**
     * @brief Writes all neurons to the file. The IDs must start at 0 and be ascending. All vectors must have the same length.
     * @param ids The IDs
     * @param positions The positions
     * @param area_names The area names
     * @param signal_types The signal types
     * @param file_path The path to the file
     * @exception Throws a RelearnException if the vectors don't all have the same length, or opening the file failed
     */
    static void write_neurons_componentwise(const std::vector<NeuronID>& ids, const std::vector<position_type>& positions, 
        const std::vector<std::string>& area_names, const std::vector<SignalType>& signal_types, const std::filesystem::path& file_path);

    /**
     * @brief Reads all neuron ids from a file and returns those.
     *      The file must be ascendingly sorted wrt. to the neuron ids (starting at 1).
     * @param file_path The path to the file to load
     * @return Empty if the file did not meet the sorting requirement, the ascending ids otherwise
     */
    [[nodiscard]] static std::optional<std::vector<NeuronID>> read_neuron_ids(const std::filesystem::path& file_path);

    /**
     * @brief Reads all local synapses from a file and returns those.
     *      Checks that no id is larger or equal to number_local_neurons.
     * @param file_path The path to the file to load
     * @param number_local_neurons The number of local neurons
     * @exception Throws a RelearnException if the weight of one synapse is 0 or a loaded id is not from [1, number_local_neurons].
     * @return All empty synapses
     */
    [[nodiscard]] static LocalSynapses read_local_synapses(const std::filesystem::path& file_path, std::uint64_t number_local_neurons);
};