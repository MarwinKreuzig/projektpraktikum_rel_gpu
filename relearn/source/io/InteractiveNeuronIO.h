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

#include "util/TaggedID.h"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <utility>
#include <vector>

/**
 * This class provides a static interface to load interrupts from files,
 * i.e., when during the simulation the neurons should be altered.
 */
class InteractiveNeuronIO {
public:
    /**
     * @brief Reads the file specified by the path and extracts all enable-interrupts.
     *      An enable-interrupt should enable a neuron during the simulation.
     *      The format of the file should be for each line:
     *      # <some comment>
     *      or
     *      {e, d, c} local_neuron_id*
     *      Only lines starting with e are processed
     * @param path_to_file The path to the interrupts file
     * @exception Throws a RelearnException if opening the file fails
     * @return A collection of pairs: (<simulation step>, <all neurons that should be enabled in the simulation step>)
     */
    [[nodiscard]] static std::vector<std::pair<size_t, std::vector<NeuronID>>> load_enable_interrupts(const std::filesystem::path& path_to_file);

    /**
     * @brief Reads the file specified by the path and extracts all disable-interrupts.
     *      A disable-interrupt should disable a neuron during the simulation.
     *      The format of the file should be for each line:
     *      # <some comment>
     *      or
     *      {e, d, c} local_neuron_id*
     *      Only lines starting with d are processed
     * @param path_to_file The path to the interrupts file
     * @exception Throws a RelearnException if opening the file fails
     * @return A collection of pairs: (<simulation step>, <all neurons that should be disabled in the simulation step>)
     */
    [[nodiscard]] static std::vector<std::pair<size_t, std::vector<NeuronID>>> load_disable_interrupts(const std::filesystem::path& path_to_file);

    /**
     * @brief Reads the file specified by the path and extracts all creation-interrupts.
     *      A creation-interrupt should create a certain number of neurons in a simulation step.
     *      The format of the file should be for each line:
     *      # <some comment>
     *      or
     *      {e, d, c} creation_count
     *      Only lines starting with c are processed
     * @param path_to_file The path to the interrupts file
     * @exception Throws a RelearnException if opening the file fails
     * @return A collection of pairs: (<simulation step>, <number of neurons to be created>)
     */
    [[nodiscard]] static std::vector<std::pair<size_t, size_t>> load_creation_interrupts(const std::filesystem::path& path_to_file);

    /**
     * @brief Reads the file specified by the path and extracts als stimulus-interrupts. 
     *      A stimulus-interrupt should provide additional background activity to a neuron in a stimulation step.
     *      The format of the file should be for each line:
     *      # <some comment>
     *      or
     *      <interval_description> <stimulus intensity> <neuron_id>*
     * @param path_to_file The path to the stimulus interrupts file
     * @exception Throws a RelearnException if opening the file fails
     * @return A function that specified for a given simulation step and a given neuron id, how much background it receives
     */
    [[nodiscard]] static std::function<double(std::uint64_t, NeuronID::value_type)> load_stimulus_interrupts(const std::filesystem::path& path_to_file);
};
