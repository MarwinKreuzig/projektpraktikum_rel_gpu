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
#include <set>
#include <string>
#include <tuple>
#include <vector>

// Interface that has to be implemented by any class that
// wants to provide a neuron-to-subdomain assignment
class NeuronToSubdomainAssignment {
public:
    // Helper class to store neuron positions
    using Position = Vec3<double>;

    virtual ~NeuronToSubdomainAssignment() = default;

    NeuronToSubdomainAssignment(const NeuronToSubdomainAssignment& other) = delete;
    NeuronToSubdomainAssignment(NeuronToSubdomainAssignment&& other) = delete;

    NeuronToSubdomainAssignment& operator=(const NeuronToSubdomainAssignment& other) = delete;
    NeuronToSubdomainAssignment& operator=(NeuronToSubdomainAssignment&& other) = delete;

    // Total number of neurons
    [[nodiscard]] size_t desired_num_neurons() const noexcept {
        return desired_num_neurons_;
    }

    // Total number of neurons already placed
    [[nodiscard]] size_t placed_num_neurons() const noexcept {
        return current_num_neurons_;
    }

    // Ratio of DendriteType::EXCITATORY neurons
    [[nodiscard]] double desired_ratio_neurons_exc() const noexcept {
        return desired_frac_neurons_exc_;
    }

    // Ratio of DendriteType::EXCITATORY neurons already placed
    [[nodiscard]] double placed_ratio_neurons_exc() const noexcept {
        return current_frac_neurons_exc_;
    }

    [[nodiscard]] const Vec3d& get_simulation_box_length() const noexcept {
        return simulation_box_length_;
    }

    [[nodiscard]] virtual std::tuple<Position, Position> get_subdomain_boundaries(const Vec3s& subdomain_3idx, size_t num_subdomains_per_axis) const noexcept;

    [[nodiscard]] virtual std::tuple<Position, Position> get_subdomain_boundaries(const Vec3s& subdomain_3idx, const Vec3s& num_subdomains_per_axis) const noexcept;

    virtual void fill_subdomain(size_t subdomain_idx, size_t num_subdomains,
        const Position& min, const Position& max)
        = 0;

    // Return number of neurons which have positions in the range [min, max) in every dimension
    [[nodiscard]] virtual size_t num_neurons(size_t subdomain_idx, size_t num_subdomains,
        const Position& min, const Position& max) const;

    // Return neurons which have positions in the range [min, max) in every dimension
    [[nodiscard]] virtual std::vector<Position> neuron_positions(size_t subdomain_idx, size_t num_subdomains,
        const Position& min, const Position& max) const;

    // Return neurons which have positions in the range [min, max) in every dimension
    [[nodiscard]] virtual std::vector<SignalType> neuron_types(size_t subdomain_idx, size_t num_subdomains,
        const Position& min, const Position& max) const;

    // Return neurons which have positions in the range [min, max) in every dimension
    [[nodiscard]] virtual std::vector<std::string> neuron_area_names(size_t subdomain_idx, size_t num_subdomains,
        const Position& min, const Position& max) const;

    virtual void write_neurons_to_file(const std::string& filename) const;

    [[nodiscard]] virtual std::vector<size_t> neuron_global_ids(size_t subdomain_idx, size_t num_subdomains,
        size_t local_id_start, size_t local_id_end) const = 0;

protected:
    struct Node {
        Position pos{ 0 };
        size_t id{ Constants::uninitialized };
        SignalType signal_type{ SignalType::EXCITATORY };
        std::string area_name{ "NOT SET" };

        struct less {
            bool operator()(const Node& lhs, const Node& rhs) const /*noexcept*/ {
                RelearnException::check(lhs.id != Constants::uninitialized, "lhs id is a dummy one");
                RelearnException::check(rhs.id != Constants::uninitialized, "rhs id is a dummy one");

                return lhs.id < rhs.id;

                Position::less less;
                const bool less_struct = less(lhs.pos, rhs.pos);
                const bool less_operator = lhs.pos < rhs.pos;
                return less_struct;
            }
        };
    };

    using Nodes = std::set<Node, Node::less>;

    void set_desired_frac_neurons_exc(double desired_frac_neurons_exc) noexcept {
        desired_frac_neurons_exc_ = desired_frac_neurons_exc;
    }

    void set_desired_num_neurons(double desired_num_neurons) noexcept {
        desired_num_neurons_ = desired_num_neurons;
    }

    void set_current_frac_neurons_exc(double current_frac_neurons_exc) noexcept {
        current_frac_neurons_exc_ = current_frac_neurons_exc;
    }

    void set_current_num_neurons(double current_num_neurons) noexcept {
        current_num_neurons_ = current_num_neurons;
    }

    void set_simulation_box_length(const Vec3d& simulation_box_length) noexcept {
        simulation_box_length_ = simulation_box_length;
    }

    [[nodiscard]] double get_desired_frac_neurons_exc() const noexcept {
       return desired_frac_neurons_exc_ ;
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

    [[nodiscard]] const Nodes& get_nodes(size_t id) const {
        const auto contains = neurons_in_subdomain.find(id) != neurons_in_subdomain.end();
        RelearnException::check(contains, "Cannot fetch nodes for id");

        return neurons_in_subdomain.at(id);
    }

    void set_nodes(size_t id, Nodes nodes) {
        neurons_in_subdomain[id] = std::move(nodes);
    }

    [[nodiscard]] bool is_loaded(size_t id) const noexcept {
        const auto contains = neurons_in_subdomain.find(id) != neurons_in_subdomain.end();
        if (!contains) {
            return false;
        }

        return !neurons_in_subdomain.at(id).empty();
    }

    [[nodiscard]] static bool position_in_box(const Position& pos, const Position& box_min, const Position& box_max) noexcept;

    NeuronToSubdomainAssignment() = default;

private:
    std::map<size_t, Nodes> neurons_in_subdomain;

    double desired_frac_neurons_exc_{ 0.0 };
    size_t desired_num_neurons_{ 0 };

    double current_frac_neurons_exc_{ 0.0 };
    size_t current_num_neurons_{ 0 };

    Vec3d simulation_box_length_{ 0 };
};
