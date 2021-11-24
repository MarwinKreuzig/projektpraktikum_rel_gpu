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

    NeuronToSubdomainAssignment(std::shared_ptr<Partition> partition)
        : partition(std::move(partition)) {
    }

    virtual ~NeuronToSubdomainAssignment() = default;

    NeuronToSubdomainAssignment(const NeuronToSubdomainAssignment& other) = delete;
    NeuronToSubdomainAssignment(NeuronToSubdomainAssignment&& other) = delete;

    NeuronToSubdomainAssignment& operator=(const NeuronToSubdomainAssignment& other) = delete;
    NeuronToSubdomainAssignment& operator=(NeuronToSubdomainAssignment&& other) = delete;

    virtual std::shared_ptr<SynapseLoader> get_synapse_loader() const noexcept = 0;
    virtual std::shared_ptr<NeuronIdTranslator> get_neuron_id_translator() const noexcept = 0;

    void initialize();

    /**
     * @brief Returns the total number of neurons that should be placed
     * @return The total number of neurons that should be placed
     */
    [[nodiscard]] size_t desired_num_neurons() const noexcept {
        return desired_num_neurons_;
    }

    /**
     * @brief Returns the current number of placed neurons 
     * @return The current number of placed neurons
     */
    [[nodiscard]] size_t placed_num_neurons() const noexcept {
        return current_num_neurons_;
    }

    /**
     * @brief Returns the total fraction of excitatory neurons that should be placed
     * @return The total fraction of excitatory neurons that should be placed
     */
    [[nodiscard]] double desired_ratio_neurons_exc() const noexcept {
        return desired_frac_neurons_exc_;
    }

    /**
     * @brief Returns the current fraction of placed excitatory neurons
     * @return The total current fraction of placed excitatory neurons
     */
    [[nodiscard]] double placed_ratio_neurons_exc() const noexcept {
        return current_frac_neurons_exc_;
    }

    /**
     * @brief Returns the size of the simulation box
     * @return The size of the simulation box
     */
    [[nodiscard]] const box_size_type& get_simulation_box_length() const noexcept {
        return simulation_box_length_;
    }

    /** 
     * @brief Returns the subdomain boundaries for a given subdomain
     * @param subdomain_3idx The 3d index of the subdomain which's boundaries are requested
     * @param num_subdomains_per_axis The number of subdomains per axis (the same for all dimensions), != 0
     * @exception Throws a RelearnException if num_subdomains_per_axis == 0
     * @return A tuple with (1) the minimum and (2) the maximum positions in the subdomain
     */
    [[nodiscard]] virtual std::tuple<box_size_type, box_size_type> get_subdomain_boundaries(const Vec3s& subdomain_3idx, size_t num_subdomains_per_axis) const;

    /** 
     * @brief Returns the subdomain boundaries for a given subdomain
     * @param subdomain_3idx The 3d index of the subdomain which's boundaries are requested
     * @param num_subdomains_per_axis The number of subdomains per axis (can have varying number per dimension)
     * @exception Might throw a RelearnException
     * @return A tuple with (1) the minimum and (2) the maximum positions in the subdomain
     */
    [[nodiscard]] virtual std::tuple<box_size_type, box_size_type> get_subdomain_boundaries(const Vec3s& subdomain_3idx, const Vec3s& num_subdomains_per_axis) const;

    /**
     * @brief Fills the subdomain with the given index and the boundaries. The implementation is left to the inhereting classes
     * @param subdomain_idx The 1d index of the subdomain which's neurons are to be filled
     * @param num_subdomains The total number of subdomains
     * @param min The subdomain's minimum position
     * @param max The subdomain's maximum position
     * @exception Might throw a RelearnException
     */
    virtual void fill_subdomain(size_t subdomain_idx, size_t num_subdomains, const box_size_type& min, const box_size_type& max) = 0;

    /**
     * @brief Return number of neurons which are in the specified subdomain and have positions in the [min, max)
     * @param subdomain_idx The 1d index of the subdomain which's neurons are to be filled
     * @param num_subdomains The total number of subdomains
     * @param min The subdomain's minimum position
     * @param max The subdomain's maximum position
     * @exception Might throw a RelearnException
     * @return The number of neurons in the subdomain
     */
    [[nodiscard]] virtual size_t num_neurons(size_t subdomain_idx, size_t num_subdomains,
        const box_size_type& min, const box_size_type& max) const;

    /**
     * @brief Return the positions of the neurons which are in the specified subdomain and have positions in the [min, max)
     * @param subdomain_idx The 1d index of the subdomain which's neurons are to be filled
     * @param num_subdomains The total number of subdomains
     * @param min The subdomain's minimum position
     * @param max The subdomain's maximum position
     * @exception Might throw a RelearnException
     * @return The positions of the neurons in the subdomain
     */
    [[nodiscard]] virtual std::vector<position_type> neuron_positions(size_t subdomain_idx, size_t num_subdomains,
        const box_size_type& min, const box_size_type& max) const;

    /**
     * @brief Return the signal type of the neurons which are in the specified subdomain and have positions in the [min, max)
     * @param subdomain_idx The 1d index of the subdomain which's neurons are to be filled
     * @param num_subdomains The total number of subdomains
     * @param min The subdomain's minimum position
     * @param max The subdomain's maximum position
     * @exception Might throw a RelearnException
     * @return The signal types of the neurons in the subdomain
     */
    [[nodiscard]] virtual std::vector<SignalType> neuron_types(size_t subdomain_idx, size_t num_subdomains,
        const box_size_type& min, const box_size_type& max) const;

    /**
     * @brief Return the area names of the neurons which are in the specified subdomain and have positions in the [min, max)
     * @param subdomain_idx The 1d index of the subdomain which's neurons are to be filled
     * @param num_subdomains The total number of subdomains
     * @param min The subdomain's minimum position
     * @param max The subdomain's maximum position
     * @exception Might throw a RelearnException
     * @return The area names of the neurons in the subdomain
     */
    [[nodiscard]] virtual std::vector<std::string> neuron_area_names(size_t subdomain_idx, size_t num_subdomains,
        const box_size_type& min, const box_size_type& max) const;

    /**
     * @brief Writes all loaded neurons into the specified file.
     *      The format is
     *      # ID, Position (x y z),	Area, type 
     * @param filename The filepath where to write the neurons
     * @exception Might throw a RelearnException
    */
    virtual void write_neurons_to_file(const std::string& filename) const;

    /**
     * @brief Returns the global ids for a given subdomain and local start and end ids.
     *      Might not be implemented
     * @param subdomain_idx The 1d index of the subdomain which's neurons are to be filled
     * @param num_subdomains The total number of subdomains
     * @param local_id_start The first local id
     * @param local_id_end The last local id
     * @exception Might throw a RelearnException
     * @return The global ids for the specified subdomain
     */
    [[nodiscard]] virtual std::vector<size_t> neuron_global_ids(size_t subdomain_idx, size_t num_subdomains) const = 0;

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

    void set_desired_frac_neurons_exc(const double desired_frac_neurons_exc) noexcept {
        desired_frac_neurons_exc_ = desired_frac_neurons_exc;
    }

    void set_desired_num_neurons(const size_t desired_num_neurons) noexcept {
        desired_num_neurons_ = desired_num_neurons;
    }

    void set_current_frac_neurons_exc(const double current_frac_neurons_exc) noexcept {
        current_frac_neurons_exc_ = current_frac_neurons_exc;
    }

    void set_current_num_neurons(const size_t current_num_neurons) noexcept {
        current_num_neurons_ = current_num_neurons;
    }

    void set_simulation_box_length(const box_size_type& simulation_box_length) noexcept;

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

    [[nodiscard]] static bool position_in_box(const position_type& pos, const box_size_type& box_min, const box_size_type& box_max) noexcept;

    std::shared_ptr<Partition> partition;

private:
    std::map<size_t, Nodes> neurons_in_subdomain;

    double desired_frac_neurons_exc_{ 0.0 };
    size_t desired_num_neurons_{ 0 };

    double current_frac_neurons_exc_{ 0.0 };
    size_t current_num_neurons_{ 0 };

    box_size_type simulation_box_length_{ 0 };
};
