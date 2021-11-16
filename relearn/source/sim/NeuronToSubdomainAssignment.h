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
#include "../neurons/SignalType.h"
#include "../util/RelearnException.h"
#include "../util/Vec3.h"
#include "../util/TaggedID.h"

#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <vector>

class NeuronIdTranslator;
class Partition;
class SynapseLoader;

/**
 * This class provides an interface for every algorithm that is used to assign neurons to MPI processes
 */
class NeuronToSubdomainAssignment {
public:
    using position_type = RelearnTypes::position_type;
    using box_size_type = RelearnTypes::box_size_type;

    NeuronToSubdomainAssignment(std::shared_ptr<Partition> partition);

    virtual ~NeuronToSubdomainAssignment() = default;

    NeuronToSubdomainAssignment(const NeuronToSubdomainAssignment& other) = delete;
    NeuronToSubdomainAssignment(NeuronToSubdomainAssignment&& other) = delete;

    NeuronToSubdomainAssignment& operator=(const NeuronToSubdomainAssignment& other) = delete;
    NeuronToSubdomainAssignment& operator=(NeuronToSubdomainAssignment&& other) = delete;

    /**
     * @brief Initializes the assignment class, i.e., loads all neurons for the subdomains
     * @exception Can throw a RelearnException
     */
    void initialize();

    /**
     * @brief Returns the associated SynapseLoader (some type that inherites from SynapseLoader)
     * @exception Throws a RelearnException if synapse_loader is nullptr
     * @return The associated SynapseLoader
     */
    std::shared_ptr<SynapseLoader> get_synapse_loader() const {
        RelearnException::check(synapse_loader.operator bool(), "NeuronToSubdomainAssignment::get_synapse_loader: synapse_loader is empty");
        return synapse_loader;
    }

    /**
     * @brief Returns the associated NeuronIdTranslator (some type that inherites from NeuronIdTranslator)
     * @exception Throws a RelearnException if neuron_id_translator is nullptr
     * @return The associated NeuronIdTranslator
     */
    std::shared_ptr<NeuronIdTranslator> get_neuron_id_translator() const {
        RelearnException::check(neuron_id_translator.operator bool(), "NeuronToSubdomainAssignment::get_neuron_id_translator: neuron_id_translator is empty");
        return neuron_id_translator;
    }

    /**
     * @brief Returns a function object that is used to fix calculated subdomain boundaries.
     *      This might be necessary if special boundaries must be considered
     * @return A function object that corrects subdomain boundaries
     */
    virtual std::function<Vec3d(Vec3d)> get_subdomain_boundary_fix() const {
        return [](Vec3d arg) { return arg; };
    }

    /**
     * @brief Returns the total number of neurons that should be placed
     * @return The total number of neurons that should be placed
     */
    [[nodiscard]] size_t get_requested_number_neurons() const noexcept {
        return requested_number_neurons;
    }

    /**
     * @brief Returns the current number of placed neurons on this MPI rank
     * @return The current number of placed neurons
     */
    [[nodiscard]] size_t get_number_placed_neurons() const noexcept {
        return number_placed_neurons;
    }

    /**
     * @brief Returns the total number of placed neurons across all MPI ranks
     * @return The total number of placed neurons across all MPI ranks
     */
    [[nodiscard]] size_t get_total_number_placed_neurons() const {
        if (total_number_neurons == Constants::uninitialized) {
            calculate_total_number_neurons();
        }

        return total_number_neurons;
    }

    /**
     * @brief Returns the total fraction of excitatory neurons that should be placed
     * @return The total fraction of excitatory neurons that should be placed
     */
    [[nodiscard]] double get_requested_ratio_excitatory_neurons() const noexcept {
        return requested_ratio_excitatory_neurons;
    }

    /**
     * @brief Returns the current fraction of placed excitatory neurons
     * @return The total current fraction of placed excitatory neurons
     */
    [[nodiscard]] double get_ratio_placed_excitatory_neurons() const noexcept {
        return ratio_placed_excitatory_neurons;
    }

    /**
     * @brief Returns number of neurons which are in the specified subdomain
     * @param subdomain_index_1d The 1d index of the subdomain which is inquired
     * @param total_number_subdomains The total number of subdomains
     * @exception Might throw a RelearnException
     * @return The number of neurons in the subdomain
     */
    [[nodiscard]] virtual size_t get_number_neurons_in_subdomain(size_t subdomain_index_1d, size_t total_number_subdomains) const;

    /**
     * @brief Returns number of neurons which are in the specified subdomains
     * @param subdomain_index_1d_start The 1d index of the first subdomain which is inquired
     * @param subdomain_index_1d_end The 1d index of the last subdomain which is inquired
     * @param total_number_subdomains The total number of subdomains
     * @exception Might throw a RelearnException
     * @return The number of neurons in the subdomains
     */
    [[nodiscard]] size_t get_number_neurons_in_subdomains(size_t subdomain_index_1d_start, size_t subdomain_index_1d_end, size_t total_number_subdomains) const {
        size_t total_number_neurons_in_subdomains = 0;

        for (size_t subdomain_id = subdomain_index_1d_start; subdomain_id <= subdomain_index_1d_end; subdomain_id++) {
            const auto number_neurons_in_subdomain = get_number_neurons_in_subdomain(subdomain_id, total_number_subdomains);
            total_number_neurons_in_subdomains += number_neurons_in_subdomain;
        }

        return total_number_neurons_in_subdomains;
    }

    /**
     * @brief Returns the positions of the neurons which are in the specified subdomain
     * @param subdomain_index_1d The 1d index of the subdomain which is inquired
     * @param total_number_subdomains The total number of subdomains
     * @exception Might throw a RelearnException
     * @return The positions of the neurons in the subdomain
     */
    [[nodiscard]] virtual std::vector<position_type> get_neuron_positions_in_subdomain(size_t subdomain_index_1d, size_t total_number_subdomains) const;

    /**
     * @brief Returns the positions of the neurons which are in the specified subdomains
     * @param subdomain_index_1d_start The 1d index of the first subdomain which is inquired
     * @param subdomain_index_1d_end The 1d index of the last subdomain which is inquired
     * @param total_number_subdomains The total number of subdomains
     * @exception Might throw a RelearnException
     * @return The positions of the neurons in the subdomains
     */
    [[nodiscard]] std::vector<position_type> get_neuron_positions_in_subdomains(size_t subdomain_index_1d_start, size_t subdomain_index_1d_end, size_t total_number_subdomains) const {
        auto function = [this](size_t subdomain_index_1d, size_t total_number_subdomains) {
            return get_neuron_positions_in_subdomain(subdomain_index_1d, total_number_subdomains);
        };

        return get_all_values<position_type>(subdomain_index_1d_start, subdomain_index_1d_end, total_number_subdomains, function);
    }

    /**
     * @brief Returns the signal type of the neurons which are in the specified subdomain
     * @param subdomain_index_1d The 1d index of the subdomain which is inquired
     * @param total_number_subdomains The total number of subdomains
     * @exception Might throw a RelearnException
     * @return The signal types of the neurons in the subdomain
     */
    [[nodiscard]] virtual std::vector<SignalType> get_neuron_types_in_subdomain(size_t subdomain_index_1d, size_t total_number_subdomains) const;

    /**
     * @brief Returns the signal type of the neurons which are in the specified subdomains
     * @param subdomain_index_1d_start The 1d index of the first subdomain which is inquired
     * @param subdomain_index_1d_end The 1d index of the last subdomain which is inquired
     * @param total_number_subdomains The total number of subdomains
     * @exception Might throw a RelearnException
     * @return The signal types of the neurons in the subdomains
     */
    [[nodiscard]] std::vector<SignalType> get_neuron_types_in_subdomains(size_t subdomain_index_1d_start, size_t subdomain_index_1d_end, size_t total_number_subdomains) const {
        auto function = [this](size_t subdomain_index_1d, size_t total_number_subdomains) {
            return get_neuron_types_in_subdomain(subdomain_index_1d, total_number_subdomains);
        };

        return get_all_values<SignalType>(subdomain_index_1d_start, subdomain_index_1d_end, total_number_subdomains, function);
    }

    /**
     * @brief Returns the area names of the neurons which are in the specified subdomain
     * @param subdomain_index_1d The 1d index of the subdomain which is inquired
     * @param total_number_subdomains The total number of subdomains
     * @exception Might throw a RelearnException
     * @return The area names of the neurons in the subdomain
     */
    [[nodiscard]] virtual std::vector<std::string> get_neuron_area_names_in_subdomain(size_t subdomain_index_1d, size_t total_number_subdomains) const;

    /**
     * @brief Returns the area names of the neurons which are in the specified subdomains
     * @param subdomain_index_1d_start The 1d index of the first subdomain which is inquired
     * @param subdomain_index_1d_end The 1d index of the last subdomain which is inquired
     * @param total_number_subdomains The total number of subdomains
     * @exception Might throw a RelearnException
     * @return The area names of the neurons in the subdomains
     */
    [[nodiscard]] std::vector<std::string> get_neuron_area_names_in_subdomains(size_t subdomain_index_1d_start, size_t subdomain_index_1d_end, size_t total_number_subdomains) const {
        auto function = [this](size_t subdomain_index_1d, size_t total_number_subdomains) {
            return get_neuron_area_names_in_subdomain(subdomain_index_1d, total_number_subdomains);
        };

        return get_all_values<std::string>(subdomain_index_1d_start, subdomain_index_1d_end, total_number_subdomains, function);
    }

    /**
     * @brief Returns the global ids for a given subdomain.
     *      Might not be implemented
     * @param subdomain_index_1d The 1d index of the subdomain which is inquired
     * @param total_number_subdomains The total number of subdomains
     * @exception Might throw a RelearnException
     * @return The global ids for the specified subdomain
     */
    [[nodiscard]] virtual std::vector<NeuronID> get_neuron_global_ids_in_subdomain(size_t subdomain_index_1d, size_t total_number_subdomains) const = 0;

    /**
     * @brief Returns the global ids of the neurons which are in the specified subdomains.
     *      Might not be implemented
     * @param subdomain_index_1d_start The 1d index of the first subdomain which is inquired
     * @param subdomain_index_1d_end The 1d index of the last subdomain which is inquired
     * @param total_number_subdomains The total number of subdomains
     * @exception Might throw a RelearnException
     * @return The global ids of the neurons in the subdomains
     */
    [[nodiscard]] std::vector<NeuronID> get_neuron_global_ids_in_subdomains(size_t subdomain_index_1d_start, size_t subdomain_index_1d_end, size_t total_number_subdomains) const {
        auto function = [this](size_t subdomain_index_1d, size_t total_number_subdomains) {
            return get_neuron_global_ids_in_subdomain(subdomain_index_1d, total_number_subdomains);
        };

        return get_all_values<NeuronID>(subdomain_index_1d_start, subdomain_index_1d_end, total_number_subdomains, function);
    }

    /**
     * @brief Writes all loaded neurons into the specified file.
     *      The format is
     *      # ID, Position (x y z),	Area, type
     * @param file_path The filepath where to write the neurons
     * @exception Might throw a RelearnException
     */
    virtual void write_neurons_to_file(const std::filesystem::path& file_path) const;

protected:
    struct Node {
        position_type pos{ 0 };
        NeuronID id{ NeuronID::uninitialized_id() };
        SignalType signal_type{ SignalType::EXCITATORY };
        std::string area_name{ "NOT SET" };

        struct less {
            bool operator()(const Node& lhs, const Node& rhs) const {
                RelearnException::check(lhs.id.is_initialized, "Node::less::operator(): lhs id is a dummy one");
                RelearnException::check(rhs.id.is_initialized, "Node::less::operator(): rhs id is a dummy one");

                return lhs.id < rhs.id;
            }
        };
    };

    using Nodes = std::set<Node, Node::less>;

    /**
     * @brief Fills the subdomain with the given index and the boundaries. The implementation is left to the inhereting classes
     * @param subdomain_index_1d The 1d index of the subdomain which's neurons are to be filled
     * @param total_number_subdomains The total number of subdomains
     * @param min The subdomain's minimum position
     * @param max The subdomain's maximum position
     * @exception Might throw a RelearnException
     */
    virtual void fill_subdomain(size_t local_subdomain_index, size_t total_number_subdomains) = 0;

    void set_requested_ratio_excitatory_neurons(const double desired_frac_neurons_exc) noexcept {
        requested_ratio_excitatory_neurons = desired_frac_neurons_exc;
    }

    void set_requested_number_neurons(const size_t get_requested_number_neurons) noexcept {
        requested_number_neurons = get_requested_number_neurons;
    }

    void set_ratio_placed_excitatory_neurons(const double current_frac_neurons_exc) noexcept {
        ratio_placed_excitatory_neurons = current_frac_neurons_exc;
    }

    void set_number_placed_neurons(const size_t current_num_neurons) noexcept {
        number_placed_neurons = current_num_neurons;
    }

    void set_total_number_placed_neurons(const size_t total_number_placed_neurons) const {
        total_number_neurons = total_number_placed_neurons;
    }

    [[nodiscard]] const Nodes& get_nodes_for_subdomain(const size_t subdomain_index_1d) const {
        const auto subdomain_is_loaded = is_subdomain_loaded(subdomain_index_1d);
        RelearnException::check(subdomain_is_loaded, "NeuronToSubdomainAssignment::get_nodes_for_subdomain: Cannot fetch nodes for id {}", subdomain_index_1d);

        return neurons_in_subdomain.at(subdomain_index_1d);
    }

    void set_nodes_for_subdomain(const size_t subdomain_index_1d, Nodes&& nodes) {
        neurons_in_subdomain[subdomain_index_1d] = std::move(nodes);
    }

    [[nodiscard]] bool is_subdomain_loaded(const size_t subdomain_index_1d) const noexcept {
        const auto contains = neurons_in_subdomain.find(subdomain_index_1d) != neurons_in_subdomain.end();
        if (!contains) {
            return false;
        }

        return true;
    }

    virtual void calculate_total_number_neurons() const = 0;

    virtual void post_initialization() = 0;

    std::shared_ptr<Partition> partition{};

    std::shared_ptr<SynapseLoader> synapse_loader{};
    std::shared_ptr<NeuronIdTranslator> neuron_id_translator{};

private:
    template <typename T>
    std::vector<T> get_all_values(size_t subdomain_index_1d_start, size_t subdomain_index_1d_end, size_t total_number_subdomains, std::function<std::vector<T>(size_t, size_t)> subdomain_function) const {
        RelearnException::check(subdomain_index_1d_end >= subdomain_index_1d_start, "NeuronToSubdomainAssignment::get_all_values: end was smaller than start");

        std::vector<T> all_values;

        for (size_t subdomain_id = subdomain_index_1d_start; subdomain_id <= subdomain_index_1d_end; subdomain_id++) {
            const auto& partial_values = subdomain_function(subdomain_id, total_number_subdomains);
            all_values.insert(all_values.cend(), partial_values.begin(), partial_values.end());
        }

        return all_values;
    }

    std::map<size_t, Nodes> neurons_in_subdomain{};

    double requested_ratio_excitatory_neurons{ 0.0 };
    size_t requested_number_neurons{ 0 };

    double ratio_placed_excitatory_neurons{ 0.0 };
    size_t number_placed_neurons{ 0 };

    mutable size_t total_number_neurons{ Constants::uninitialized };
};
