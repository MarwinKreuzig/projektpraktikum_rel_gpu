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
     * @brief Initializes the assignment class, i.e., loads all neurons for the subdomains.
     *      This method is only virtual in case an inheriting class must do something afterwards.
     */
    virtual void initialize();

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
    [[nodiscard]] size_t get_number_requested_neurons() const noexcept {
        return desired_num_neurons_;
    }

    /**
     * @brief Returns the current number of placed neurons 
     * @return The current number of placed neurons
     */
    [[nodiscard]] size_t get_number_placed_neurons() const noexcept {
        return current_num_neurons_;
    }

    /**
     * @brief Returns the total fraction of excitatory neurons that should be placed
     * @return The total fraction of excitatory neurons that should be placed
     */
    [[nodiscard]] double get_requested_ratio_excitatory_neurons() const noexcept {
        return desired_frac_neurons_exc_;
    }

    /**
     * @brief Returns the current fraction of placed excitatory neurons
     * @return The total current fraction of placed excitatory neurons
     */
    [[nodiscard]] double get_ratio_placed_excitatory_neurons() const noexcept {
        return current_frac_neurons_exc_;
    }

    /**
     * @brief Return number of neurons which are in the specified subdomain
     * @param subdomain_idx The 1d index of the subdomain which is inquired
     * @param num_subdomains The total number of local_subdomains
     * @exception Might throw a RelearnException
     * @return The number of neurons in the subdomain
     */
    [[nodiscard]] virtual size_t get_number_neurons_in_subdomain(size_t subdomain_idx, size_t num_subdomains) const;

    /**
     * @brief Return the positions of the neurons which are in the specified subdomain
     * @param subdomain_idx The 1d index of the subdomain which is inquired
     * @param num_subdomains The total number of local_subdomains
     * @exception Might throw a RelearnException
     * @return The positions of the neurons in the subdomain
     */
    [[nodiscard]] virtual std::vector<position_type> get_neuron_positions_in_subdomain(size_t subdomain_idx, size_t num_subdomains) const;

    /**
     * @brief Return the signal type of the neurons which are in the specified subdomain
     * @param subdomain_idx The 1d index of the subdomain which is inquired
     * @param num_subdomains The total number of local_subdomains
     * @exception Might throw a RelearnException
     * @return The signal types of the neurons in the subdomain
     */
    [[nodiscard]] virtual std::vector<SignalType> get_neuron_types_in_subdomain(size_t subdomain_idx, size_t num_subdomains) const;

    /**
     * @brief Return the area names of the neurons which are in the specified subdomain
     * @param subdomain_idx The 1d index of the subdomain which is inquired
     * @param num_subdomains The total number of local_subdomains
     * @exception Might throw a RelearnException
     * @return The area names of the neurons in the subdomain
     */
    [[nodiscard]] virtual std::vector<std::string> get_neuron_area_names_in_subdomain(size_t subdomain_idx, size_t num_subdomains) const;

    /**
     * @brief Returns the global ids for a given subdomain.
     *      Might not be implemented
     * @param subdomain_idx The 1d index of the subdomain which is inquired
     * @param num_subdomains The total number of subdomains
     * @exception Might throw a RelearnException
     * @return The global ids for the specified subdomain
     */
    [[nodiscard]] virtual std::vector<size_t> get_neuron_global_ids_in_subdomain(size_t subdomain_idx, size_t num_subdomains) const = 0;

    /**
     * @brief Writes all loaded neurons into the specified file.
     *      The format is
     *      # ID, Position (x y z),	Area, type 
     * @param filename The filepath where to write the neurons
     * @exception Might throw a RelearnException
    */
    virtual void write_neurons_to_file(const std::filesystem::path& filename) const;

protected:
    struct Node {
        position_type pos{ 0 };
        size_t id{ Constants::uninitialized };
        SignalType signal_type{ SignalType::EXCITATORY };
        std::string area_name{ "NOT SET" };

        struct less {
            bool operator()(const Node& lhs, const Node& rhs) const {
                RelearnException::check(lhs.id != Constants::uninitialized, "Node::less::operator(): lhs id is a dummy one");
                RelearnException::check(rhs.id != Constants::uninitialized, "Node::less::operator(): rhs id is a dummy one");

                return lhs.id < rhs.id;
            }
        };
    };

    using Nodes = std::set<Node, Node::less>;

    /**
     * @brief Fills the subdomain with the given index and the boundaries. The implementation is left to the inhereting classes
     * @param subdomain_idx The 1d index of the subdomain which's neurons are to be filled
     * @param num_subdomains The total number of local_subdomains
     * @param min The subdomain's minimum position
     * @param max The subdomain's maximum position
     * @exception Might throw a RelearnException
     */
    virtual void fill_subdomain(size_t subdomain_idx, size_t num_subdomains, const box_size_type& min, const box_size_type& max) = 0;

    void set_desired_frac_neurons_exc(const double desired_frac_neurons_exc) noexcept {
        desired_frac_neurons_exc_ = desired_frac_neurons_exc;
    }

    void set_desired_num_neurons(const size_t get_number_requested_neurons) noexcept {
        desired_num_neurons_ = get_number_requested_neurons;
    }

    void set_current_frac_neurons_exc(const double current_frac_neurons_exc) noexcept {
        current_frac_neurons_exc_ = current_frac_neurons_exc;
    }

    void set_current_num_neurons(const size_t current_num_neurons) noexcept {
        current_num_neurons_ = current_num_neurons;
    }

    [[nodiscard]] double get_desired_frac_neurons_exc() const noexcept {
        return desired_frac_neurons_exc_;
    }

    [[nodiscard]] size_t get_desired_num_neurons() const noexcept {
        return desired_num_neurons_;
    }

    [[nodiscard]] double get_current_frac_neurons_exc() const noexcept {
        return current_frac_neurons_exc_;
    }

    [[nodiscard]] size_t get_current_num_neurons() const noexcept {
        return current_num_neurons_;
    }

    [[nodiscard]] const Nodes& get_nodes(const size_t id) const {
        const auto contains = neurons_in_subdomain.find(id) != neurons_in_subdomain.end();
        RelearnException::check(contains, "NeuronToSubdomainAssignment::get_nodes: Cannot fetch nodes for id {}", id);

        return neurons_in_subdomain.at(id);
    }

    void set_nodes(const size_t id, Nodes&& nodes) {
        neurons_in_subdomain[id] = std::move(nodes);
    }

    [[nodiscard]] bool is_loaded(const size_t id) const noexcept {
        const auto contains = neurons_in_subdomain.find(id) != neurons_in_subdomain.end();
        if (!contains) {
            return false;
        }

        return !neurons_in_subdomain.at(id).empty();
    }

    [[nodiscard]] static bool position_in_box(const position_type& pos, const box_size_type& box_min, const box_size_type& box_max) noexcept {
        return ((pos.get_x() >= box_min.get_x() && pos.get_x() <= box_max.get_x()) && (pos.get_y() >= box_min.get_y() && pos.get_y() <= box_max.get_y()) && (pos.get_z() >= box_min.get_z() && pos.get_z() <= box_max.get_z()));
    }

    std::shared_ptr<Partition> partition;

    std::shared_ptr<SynapseLoader> synapse_loader{};
    std::shared_ptr<NeuronIdTranslator> neuron_id_translator{};

private:
    std::map<size_t, Nodes> neurons_in_subdomain;

    double desired_frac_neurons_exc_{ 0.0 };
    size_t desired_num_neurons_{ 0 };

    double current_frac_neurons_exc_{ 0.0 };
    size_t current_num_neurons_{ 0 };
};
