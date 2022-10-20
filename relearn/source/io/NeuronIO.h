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
#include <string>
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
     * @brief Reads all comments from the beginning of the file and returns those.
     *      Comments start with '#'. It stops at the file end or the fist non-comment line
     * @param file_path The path to the file to load
     * @exception Throws a RelearnException if opening the file failed
     * @return Returns all comments at the beginning of the file
     */
    [[nodiscard]] static std::vector<std::string> read_comments(const std::filesystem::path& file_path);

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
     * @brief Writes all neurons to the file
     * @param neurons The neurons
     * @param file_path The path to the file
     * @exception Throws a RelearnException if opening the file failed
     */
    static void write_neurons(const std::vector<LoadedNeuron>& neurons, const std::filesystem::path& file_path);

    /**
     * @brief Writes all neurons to the file. The IDs must start at 0 and be ascending. All vectors must have the same length.
     *      Does not check for correct IDs or non-negative positions.
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
     * @exception Throws a RelearnException if opening the file failed, the weight of one synapse is 0, or a loaded id is not from [1, number_local_neurons].
     * @return All local synapses
     */
    [[nodiscard]] static LocalSynapses read_local_synapses(const std::filesystem::path& file_path, NeuronID::value_type number_local_neurons);

    /**
     * @brief Write all local synapses to the specified file
     * @param local_synapses The local synapses
     * @param file_path The path to the file
     * @exception Throws a RelearnException if opening the file failed
     */
    static void write_local_synapses(const LocalSynapses& local_synapses, const std::filesystem::path& file_path);

    /**
     * @brief Reads all distant in-synapses from a file and returns those.
     *      Checks that no target id is larger or equal to number_local_neurons and that no source rank is larger or equal to number_mpi_ranks.
     * @param file_path The path to the file to load
     * @param number_local_neurons The number of local neurons
     * @param my_rank The current MPI rank
     * @param number_mpi_ranks The number of MPI ranks
     * @exception Throws a RelearnException if
     *      (1) opening the file failed
     *      (2) the weight of one synpapse is 0
     *      (3) a target rank is not my_rank
     *      (4) a source rank is not from [0, number_mpi_ranks)
     *      (5) a source rank is equal to my_rank
     *      (6) or a target id is not from [0, number_local_neurons)
     * @return All distant in-synapses
     */
    static DistantInSynapses read_distant_in_synapses(const std::filesystem::path& file_path, NeuronID::value_type number_local_neurons, int my_rank, int number_mpi_ranks);

    /**
     * @brief Writes all distant in-synapses to the specified file
     * @param distant_in_synapses The distant in-synapses
     * @param my_rank The current MPI rank
     * @param file_path The path to the file
     * @exception Throws a RelearnException if opening the file failed or if a source rank is equal to my_rank
     */
    static void write_distant_in_synapses(const DistantInSynapses& distant_in_synapses, int my_rank, const std::filesystem::path& file_path);

    /**
     * @brief Reads all distant out-synapses from a file and returns those.
     *      Checks that no source id is larger or equal to number_local_neurons and that no target rank is larger or equal to number_mpi_ranks.
     * @param file_path The path to the file to load
     * @param number_local_neurons The number of local neurons
     * @param my_rank The current MPI rank
     * @param number_mpi_ranks The number of MPI ranks
     * @exception Throws a RelearnException if
     *      (1) opening the file failed
     *      (2) the weight of one synpapse is 0
     *      (3) a source rank is not my_rank
     *      (4) a target rank is not from [0, number_mpi_ranks)
     *      (5) a target rank is equal to my_rank
     *      (6) or a source id is not from [0, number_local_neurons)
     * @return All distant out-synapses
     */
    static DistantOutSynapses read_distant_out_synapses(const std::filesystem::path& file_path, NeuronID::value_type number_local_neurons, int my_rank, int number_mpi_ranks);

    /**
     * @brief Writes all distant out-synapses to the specified file
     * @param distant_out_synapses The distant out-synapses
     * @param my_rank The current MPI rank
     * @param file_path The path to the file
     * @exception Throws a RelearnException if opening the file failed or if a target rank is equal to my_rank
     */
    static void write_distant_out_synapses(const DistantOutSynapses& distant_out_synapses, int my_rank, const std::filesystem::path& file_path);

    /**
     * @brief Reads all in-synapses from a file and returns those.
     *      Checks that no target id is larger or equal to number_local_neurons and that no source rank is larger or equal to number_mpi_ranks.
     * @param file_path The path to the file to load
     * @param number_local_neurons The number of local neurons
     * @param my_rank The current MPI rank
     * @param number_mpi_ranks The number of MPI ranks
     * @exception Throws a RelearnException if
     *      (1) opening the file failed
     *      (2) the weight of one synpapse is 0
     *      (3) a target rank is not my_rank
     *      (4) a source rank is not from [0, number_mpi_ranks)
     *      (5) or a target id is not from [0, number_local_neurons)
     * @return All in-synapses as a tuple: (1) The local ones and (2) the distant ones
     */
    static std::tuple<LocalSynapses, DistantInSynapses> read_in_synapses(const std::filesystem::path& file_path, NeuronID::value_type number_local_neurons, int my_rank, int number_mpi_ranks);

    /**
     * @brief Writes all in-synapses to the specified file
     * @param local_in_synapses The local in-synapses
     * @param distant_in_synapses The distant in-synapses
     * @param my_rank The current MPI rank
     * @param file_path The path to the file
     * @exception Throws a RelearnException if opening the file failed or if the source rank of a distant in-synapse is equal to my_rank
     */
    static void write_in_synapses(const LocalSynapses& local_in_synapses, const DistantInSynapses& distant_in_synapses, int my_rank, const std::filesystem::path& file_path);

    /**
     * @brief Reads all out-synapses from a file and returns those.
     *      Checks that no source id is larger or equal to number_local_neurons and that no target rank is larger or equal to number_mpi_ranks.
     * @param file_path The path to the file to load
     * @param number_local_neurons The number of local neurons
     * @param my_rank The current MPI rank
     * @param number_mpi_ranks The number of MPI ranks
     * @exception Throws a RelearnException if
     *      (1) opening the file failed
     *      (2) the weight of one synpapse is 0
     *      (3) a source rank is not my_rank
     *      (4) a target rank is not from [0, number_mpi_ranks)
     *      (5) or a source id is not from [0, number_local_neurons)
     * @return All out-synapses as a tuple: (1) The local ones and (2) the distant ones
     */
    static std::tuple<LocalSynapses, DistantOutSynapses> read_out_synapses(const std::filesystem::path& file_path, NeuronID::value_type number_local_neurons, int my_rank, int number_mpi_ranks);

    /**
     * @brief Writes all out-synapses to the specified file
     * @param local_out_synapses The local out-synapses
     * @param distant_out_synapses The distant out-synapses
     * @param my_rank The current MPI rank
     * @param file_path The path to the file
     * @exception Throws a RelearnException if opening the file failed or if the target rank of a distant out-synapse is equal to my_rank
     */
    static void write_out_synapses(const LocalSynapses& local_out_synapses, const DistantOutSynapses& distant_out_synapses, int my_rank, const std::filesystem::path& file_path);
};