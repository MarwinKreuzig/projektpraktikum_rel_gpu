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
#include "../structure/NeuronIdTranslator.h"
#include "../structure/Partition.h"
#include "../util/RelearnException.h"

#include <fstream>
#include <iomanip>

NeuronToSubdomainAssignment::NeuronToSubdomainAssignment(std::shared_ptr<Partition> partition)
    : partition(std::move(partition)) {
    partition->set_boundary_correction_function(get_subdomain_boundary_fix());
}

void NeuronToSubdomainAssignment::initialize() {
    const auto num_subdomains = partition->get_number_local_subdomains();

    std::vector<size_t> number_neurons_in_subdomains(num_subdomains);

    for (auto i = 0; i < num_subdomains; i++) {
        const auto& index_1d = partition->get_1d_index_of_subdomain(i);
        const auto& index_3d = partition->get_3d_index_of_subdomain(i);

        const auto& [min, max] = partition->get_subdomain_boundaries(index_3d);
        partition->set_subdomain_boundaries(i, min, max);

        fill_subdomain(index_1d, num_subdomains, min, max);

        const auto num_neurons = get_number_neurons_in_subdomain(index_1d, num_subdomains);

        number_neurons_in_subdomains[i] = num_neurons;

        auto global_ids = get_neuron_global_ids_in_subdomain(index_1d, num_subdomains);
        std::sort(global_ids.begin(), global_ids.end());

        neuron_id_translator->set_global_ids(i, std::move(global_ids));
    }

    partition->set_subdomain_number_neurons(number_neurons_in_subdomains);
}

size_t NeuronToSubdomainAssignment::get_number_neurons_in_subdomain(const size_t subdomain_idx, [[maybe_unused]] const size_t num_subdomains) const {

    const auto contains = neurons_in_subdomain.find(subdomain_idx) != neurons_in_subdomain.end();
    if (!contains) {
        RelearnException::fail("NeuronToSubdomainAssignment::number_neurons: Wanted to have number_neurons of subdomain_idx that is not present");
        return 0;
    }

    const auto& nodes = neurons_in_subdomain.at(subdomain_idx);
    const auto cnt = nodes.size();
    return cnt;
}

std::vector<NeuronToSubdomainAssignment::position_type> NeuronToSubdomainAssignment::get_neuron_positions_in_subdomain(const size_t subdomain_idx, [[maybe_unused]] const size_t num_subdomains) const {

    const auto contains = neurons_in_subdomain.find(subdomain_idx) != neurons_in_subdomain.end();
    if (!contains) {
        RelearnException::fail("NeuronToSubdomainAssignment::get_neuron_positions_in_subdomain: Wanted to have get_neuron_positions_in_subdomain of subdomain_idx that is not present: {}", subdomain_idx);
        return {};
    }

    const auto& nodes = neurons_in_subdomain.at(subdomain_idx);
    std::vector<position_type> pos{};
    pos.reserve(nodes.size());

    for (const auto& node : nodes) {
        pos.push_back(node.pos);
    }

    return pos;
}

std::vector<SignalType> NeuronToSubdomainAssignment::get_neuron_types_in_subdomain(const size_t subdomain_idx, [[maybe_unused]] const size_t num_subdomains) const {

    const auto contains = neurons_in_subdomain.find(subdomain_idx) != neurons_in_subdomain.end();
    if (!contains) {
        RelearnException::fail("NeuronToSubdomainAssignment::get_neuron_types_in_subdomain: Wanted to have get_neuron_types_in_subdomain of subdomain_idx that is not present: {}", subdomain_idx);
        return {};
    }

    const auto& nodes = neurons_in_subdomain.at(subdomain_idx);
    std::vector<SignalType> types{};
    types.reserve(nodes.size());

    for (const auto& node : nodes) {
        types.push_back(node.signal_type);
    }

    return types;
}

std::vector<std::string> NeuronToSubdomainAssignment::get_neuron_area_names_in_subdomain(const size_t subdomain_idx, [[maybe_unused]] const size_t num_subdomains) const {

    const auto contains = neurons_in_subdomain.find(subdomain_idx) != neurons_in_subdomain.end();
    if (!contains) {
        RelearnException::fail("NeuronToSubdomainAssignment::get_neuron_area_names_in_subdomain: Wanted to have get_neuron_area_names_in_subdomain of subdomain_idx that is not present: {}", subdomain_idx);
        return {};
    }

    const auto& nodes = neurons_in_subdomain.at(subdomain_idx);
    std::vector<std::string> areas;
    areas.reserve(nodes.size());

    for (const auto& node : nodes) {
        areas.push_back(node.area_name);
    }

    return areas;
}

void NeuronToSubdomainAssignment::write_neurons_to_file(const std::filesystem::path& filename) const {
    std::ofstream of(filename, std::ios::binary | std::ios::out);

    const auto is_good = of.good();
    const auto is_bad = of.bad();

    RelearnException::check(is_good && !is_bad, "NeuronToSubdomainAssignment::write_neurons_to_file: The ofstream failed to open");

    of << std::setprecision(std::numeric_limits<double>::digits10);
    of << "# ID, Position (x y z),  Area,   type \n";

    for (const auto& [_, nodes] : neurons_in_subdomain) {
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
}
