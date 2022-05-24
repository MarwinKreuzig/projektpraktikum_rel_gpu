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

#include "neurons/models/SynapticElements.h"
#include "structure/Partition.h"
#include "util/RelearnException.h"

#include <fstream>
#include <iomanip>

NeuronToSubdomainAssignment::NeuronToSubdomainAssignment(std::shared_ptr<Partition> partition)
    : partition(std::move(partition)) {
}

void NeuronToSubdomainAssignment::initialize() {
    partition->set_boundary_correction_function(get_subdomain_boundary_fix());
    partition->calculate_and_set_subdomain_boundaries();
    const auto total_number_subdomains = partition->get_number_local_subdomains();

    std::vector<size_t> number_neurons_in_subdomains(total_number_subdomains);

    for (auto i = 0; i < total_number_subdomains; i++) {
        const auto& index_1d = partition->get_1d_index_of_subdomain(i);

        fill_subdomain(i, total_number_subdomains);
        const auto number_neurons_in_subdomain = get_number_neurons_in_subdomain(index_1d, total_number_subdomains);

        number_neurons_in_subdomains[i] = number_neurons_in_subdomain;
    }

    partition->set_subdomain_number_neurons(number_neurons_in_subdomains);

    post_initialization();
}

size_t NeuronToSubdomainAssignment::get_number_neurons_in_subdomain(const size_t subdomain_index_1d, [[maybe_unused]] const size_t total_number_subdomains) const {
    const auto subdomain_is_loaded = is_subdomain_loaded(subdomain_index_1d);
    if (!subdomain_is_loaded) {
        RelearnException::fail("NeuronToSubdomainAssignment::number_neurons: Wanted to have number_neurons of subdomain_index_1d that is not present");
        return 0;
    }

    const auto& nodes = neurons_in_subdomain.at(subdomain_index_1d);
    const auto cnt = nodes.size();
    return cnt;
}

std::vector<NeuronToSubdomainAssignment::position_type> NeuronToSubdomainAssignment::get_neuron_positions_in_subdomain(const size_t subdomain_index_1d, [[maybe_unused]] const size_t total_number_subdomains) const {
    const auto subdomain_is_loaded = is_subdomain_loaded(subdomain_index_1d);
    if (!subdomain_is_loaded) {
        RelearnException::fail("NeuronToSubdomainAssignment::get_neuron_positions_in_subdomain: Wanted to have get_neuron_positions_in_subdomain of subdomain_index_1d that is not present: {}", subdomain_index_1d);
        return {};
    }

    const auto& nodes = neurons_in_subdomain.at(subdomain_index_1d);
    std::vector<position_type> pos{};
    pos.reserve(nodes.size());

    for (const auto& node : nodes) {
        pos.push_back(node.pos);
    }

    return pos;
}

std::vector<SignalType> NeuronToSubdomainAssignment::get_neuron_types_in_subdomain(const size_t subdomain_index_1d, [[maybe_unused]] const size_t total_number_subdomains) const {
    const auto subdomain_is_loaded = is_subdomain_loaded(subdomain_index_1d);
    if (!subdomain_is_loaded) {
        RelearnException::fail("NeuronToSubdomainAssignment::get_neuron_types_in_subdomain: Wanted to have get_neuron_types_in_subdomain of subdomain_index_1d that is not present: {}", subdomain_index_1d);
        return {};
    }

    const auto& nodes = neurons_in_subdomain.at(subdomain_index_1d);
    std::vector<SignalType> types{};
    types.reserve(nodes.size());

    for (const auto& node : nodes) {
        types.push_back(node.signal_type);
    }

    return types;
}

std::vector<std::string> NeuronToSubdomainAssignment::get_neuron_area_names_in_subdomain(const size_t subdomain_index_1d, [[maybe_unused]] const size_t total_number_subdomains) const {
    const auto subdomain_is_loaded = is_subdomain_loaded(subdomain_index_1d);
    if (!subdomain_is_loaded) {
        RelearnException::fail("NeuronToSubdomainAssignment::get_neuron_area_names_in_subdomain: Wanted to have get_neuron_area_names_in_subdomain of subdomain_index_1d that is not present: {}", subdomain_index_1d);
        return {};
    }

    const auto& nodes = neurons_in_subdomain.at(subdomain_index_1d);
    std::vector<std::string> areas{};
    areas.reserve(nodes.size());

    for (const auto& node : nodes) {
        areas.push_back(node.area_name);
    }

    return areas;
}

void NeuronToSubdomainAssignment::write_neurons_to_file(const std::filesystem::path& file_path) const {
    std::ofstream of(file_path, std::ios::binary | std::ios::out);

    const auto is_good = of.good();
    const auto is_bad = of.bad();

    RelearnException::check(is_good && !is_bad, "NeuronToSubdomainAssignment::write_neurons_to_file: The ofstream failed to open");

    of << std::setprecision(std::numeric_limits<double>::digits10);
    of << "# ID, Position (x y z),  Area,   type \n";

    for (const auto& [_, nodes] : neurons_in_subdomain) {
        for (const auto& node : nodes) {
            const auto id = node.id.get_global_id() + 1;
            const auto& [x, y, z] = node.pos;

            of
                << id << "\t"
                << x << " "
                << y << " "
                << z << "\t"
                << node.area_name << "\t";

            if (node.signal_type == SignalType::Excitatory) {
                of << "ex\n";
            } else {
                of << "in\n";
            }
        }
    }
}
