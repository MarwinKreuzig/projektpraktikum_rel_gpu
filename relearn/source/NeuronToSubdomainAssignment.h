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

#include "Commons.h"
#include "Random.h"
#include "RelearnException.h"
#include "SynapticElements.h"
#include "Vec3.h"

#include <map>
#include <set>
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
        return currently_num_neurons_;
    }

    // Ratio of DendriteType::EXCITATORY neurons
    [[nodiscard]] double desired_ratio_neurons_exc() const noexcept {
        return desired_frac_neurons_exc_;
    }

    // Ratio of DendriteType::EXCITATORY neurons already placed
    [[nodiscard]] double placed_ratio_neurons_exc() const noexcept {
        return currently_frac_neurons_exc_;
    }

    [[nodiscard]] Vec3d get_simulation_box_length() const noexcept {
        return simulation_box_length;
    }

    [[nodiscard]] virtual std::tuple<Position, Position> get_subdomain_boundaries(const Vec3<size_t>& subdomain_3idx, size_t num_subdomains_per_axis) const noexcept;

    [[nodiscard]] virtual std::tuple<Position, Position> get_subdomain_boundaries(const Vec3<size_t>& subdomain_3idx, const Vec3<size_t>& num_subdomains_per_axis) const noexcept;

    virtual void fill_subdomain(size_t subdomain_idx, size_t num_subdomains,
        const Position& min, const Position& max)
        = 0;

    // Return number of neurons which have positions in the range [min, max) in every dimension
    [[nodiscard]] virtual size_t num_neurons(size_t subdomain_idx, size_t num_subdomains,
        const Position& min, const Position& max) const;

    // Return neurons which have positions in the range [min, max) in every dimension
    virtual void neuron_positions(size_t subdomain_idx, size_t num_subdomains,
        const Position& min, const Position& max, std::vector<Position>& pos) const;

    // Return neurons which have positions in the range [min, max) in every dimension
    virtual void neuron_types(size_t subdomain_idx, size_t num_subdomains,
        const Position& min, const Position& max,
        std::vector<SynapticElements::SignalType>& types) const;

    // Return neurons which have positions in the range [min, max) in every dimension
    virtual void neuron_area_names(size_t subdomain_idx, size_t num_subdomains,
        const Position& min, const Position& max, std::vector<std::string>& areas) const;

    virtual void write_neurons_to_file(const std::string& filename) const;

    virtual void neuron_global_ids(size_t subdomain_idx, size_t num_subdomains,
        size_t local_id_start, size_t local_id_end, std::vector<size_t>& global_ids) const = 0;

protected:
    struct Node {
        Position pos{ 0 };
        size_t id{ Constants::uninitialized };
        SynapticElements::SignalType signal_type{ SynapticElements::SignalType::EXCITATORY };
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
    std::map<size_t, Nodes> neurons_in_subdomain;

    double desired_frac_neurons_exc_{ 0.0 };
    size_t desired_num_neurons_{ 0 };

    double currently_frac_neurons_exc_{ 0.0 };
    size_t currently_num_neurons_{ 0 };

    Vec3d simulation_box_length{ 0 };

    [[nodiscard]] static bool position_in_box(const Position& pos, const Position& box_min, const Position& box_max) noexcept;

    NeuronToSubdomainAssignment() = default;
};
