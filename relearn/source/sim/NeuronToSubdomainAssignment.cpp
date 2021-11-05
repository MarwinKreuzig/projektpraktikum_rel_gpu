/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "NeuronToSubdomainAssignment.h"

#include "../neurons/models/SynapticElements.h"
#include "../util/RelearnException.h"

#include <fstream>
#include <iomanip>

std::tuple<NeuronToSubdomainAssignment::box_size_type, NeuronToSubdomainAssignment::box_size_type> NeuronToSubdomainAssignment::get_subdomain_boundaries(const Vec3s& subdomain_3idx, const size_t num_subdomains_per_axis) const {
    RelearnException::check(num_subdomains_per_axis > 0, "NeuronToSubdomainAssignment::get_subdomain_boundaries: The number of subdomains per axis must be greater than 0");
    return get_subdomain_boundaries(subdomain_3idx, Vec3s{ num_subdomains_per_axis });
}

std::tuple<NeuronToSubdomainAssignment::box_size_type, NeuronToSubdomainAssignment::box_size_type> NeuronToSubdomainAssignment::get_subdomain_boundaries(const Vec3s& subdomain_3idx, const Vec3s& num_subdomains_per_axis) const {
    RelearnException::check(num_subdomains_per_axis.get_x() > 0, "NeuronToSubdomainAssignment::get_subdomain_boundaries: The number of x subdomains must be greater than 0");
    RelearnException::check(num_subdomains_per_axis.get_y() > 0, "NeuronToSubdomainAssignment::get_subdomain_boundaries: The number of y subdomains must be greater than 0");
    RelearnException::check(num_subdomains_per_axis.get_z() > 0, "NeuronToSubdomainAssignment::get_subdomain_boundaries: The number of z subdomains must be greater than 0");

    const auto& lengths = get_simulation_box_length();
    const auto x_subdomain_length = lengths.get_x() / num_subdomains_per_axis.get_x();
    const auto y_subdomain_length = lengths.get_y() / num_subdomains_per_axis.get_y();
    const auto z_subdomain_length = lengths.get_z() / num_subdomains_per_axis.get_z();

    box_size_type min{ subdomain_3idx.get_x() * x_subdomain_length, subdomain_3idx.get_y() * y_subdomain_length, subdomain_3idx.get_z() * z_subdomain_length };

    const auto next_x = static_cast<double>(subdomain_3idx.get_x() + 1) * x_subdomain_length;
    const auto next_y = static_cast<double>(subdomain_3idx.get_y() + 1) * y_subdomain_length;
    const auto next_z = static_cast<double>(subdomain_3idx.get_z() + 1) * z_subdomain_length;

    box_size_type max{ next_x, next_y, next_z };

    return std::make_tuple(min, max);
}

size_t NeuronToSubdomainAssignment::num_neurons(const size_t subdomain_idx, [[maybe_unused]] const size_t num_subdomains,
    [[maybe_unused]] const box_size_type& min, [[maybe_unused]] const box_size_type& max) const {

    const bool contains = neurons_in_subdomain.find(subdomain_idx) != neurons_in_subdomain.end();
    if (!contains) {
        RelearnException::fail("NeuronToSubdomainAssignment::num_neurons: Wanted to have num_neurons of subdomain_idx that is not present");
        return 0;
    }

    const Nodes& nodes = neurons_in_subdomain.at(subdomain_idx);
    const size_t cnt = nodes.size();
    return cnt;
}

std::vector<NeuronToSubdomainAssignment::position_type> NeuronToSubdomainAssignment::neuron_positions(const size_t subdomain_idx, [[maybe_unused]] const size_t num_subdomains,
    [[maybe_unused]] const box_size_type& min, [[maybe_unused]] const box_size_type& max) const {

    const bool contains = neurons_in_subdomain.find(subdomain_idx) != neurons_in_subdomain.end();
    if (!contains) {
        RelearnException::fail("NeuronToSubdomainAssignment::neuron_positions: Wanted to have neuron_positions of subdomain_idx that is not present: {}", subdomain_idx);
        return {};
    }

    const Nodes& nodes = neurons_in_subdomain.at(subdomain_idx);
    std::vector<position_type> pos{};
    pos.reserve(nodes.size());

    for (const Node& node : nodes) {
        pos.push_back(node.pos);
    }

    return pos;
}

std::vector<SignalType> NeuronToSubdomainAssignment::neuron_types(const size_t subdomain_idx, [[maybe_unused]] const size_t num_subdomains,
    [[maybe_unused]] const box_size_type& min, [[maybe_unused]] const box_size_type& max) const {

    const bool contains = neurons_in_subdomain.find(subdomain_idx) != neurons_in_subdomain.end();
    if (!contains) {
        RelearnException::fail("NeuronToSubdomainAssignment::neuron_types: Wanted to have neuron_types of subdomain_idx that is not present: {}", subdomain_idx);
        return {};
    }

    const Nodes& nodes = neurons_in_subdomain.at(subdomain_idx);
    std::vector<SignalType> types{};
    types.reserve(nodes.size());

    for (const Node& node : nodes) {
        types.push_back(node.signal_type);
    }

    return types;
}

std::vector<std::string> NeuronToSubdomainAssignment::neuron_area_names(const size_t subdomain_idx, [[maybe_unused]] const size_t num_subdomains,
    [[maybe_unused]] const box_size_type& min, [[maybe_unused]] const box_size_type& max) const {

    const bool contains = neurons_in_subdomain.find(subdomain_idx) != neurons_in_subdomain.end();
    if (!contains) {
        RelearnException::fail("NeuronToSubdomainAssignment::neuron_area_names: Wanted to have neuron_area_names of subdomain_idx that is not present: {}", subdomain_idx);
        return {};
    }

    const Nodes& nodes = neurons_in_subdomain.at(subdomain_idx);
    std::vector<std::string> areas;
    areas.reserve(nodes.size());

    for (const Node& node : nodes) {
        areas.push_back(node.area_name);
    }

    return areas;
}

bool NeuronToSubdomainAssignment::position_in_box(const position_type& pos, const box_size_type& box_min, const box_size_type& box_max) noexcept {
    return ((pos.get_x() >= box_min.get_x() && pos.get_x() <= box_max.get_x()) && (pos.get_y() >= box_min.get_y() && pos.get_y() <= box_max.get_y()) && (pos.get_z() >= box_min.get_z() && pos.get_z() <= box_max.get_z()));
}

void NeuronToSubdomainAssignment::write_neurons_to_file(const std::string& filename) const {
    std::ofstream of(filename, std::ios::binary | std::ios::out);

    of << std::setprecision(std::numeric_limits<double>::digits10);
    of << "# ID, Position (x y z),	Area, type \n";

    for (const auto& it : neurons_in_subdomain) {
        const Nodes& nodes = it.second;

        for (const auto& node : nodes) {
            const auto id = node.id + 1;

            of
                << id << "\t"
                << node.pos.get_x() << " "
                << node.pos.get_y() << " "
                << node.pos.get_z() << "\t"
                << node.area_name << "\t";

            if (node.signal_type == SignalType::EXCITATORY) {
                of << "ex\n";
            } else {
                of << "in\n";
            }
        }
    }

    of.flush();
    of.close();
}
