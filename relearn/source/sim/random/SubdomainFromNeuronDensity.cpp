/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "SubdomainFromNeuronDensity.h"

#include "mpi/MPIWrapper.h"
#include "sim/SynapseLoader.h"
#include "sim/random/RandomSynapseLoader.h"
#include "structure/Partition.h"
#include "util/Random.h"
#include "util/RelearnException.h"

#include <limits>
#include <numeric>

SubdomainFromNeuronDensity::SubdomainFromNeuronDensity(const size_t number_neurons, const double fraction_excitatory_neurons, const double um_per_neuron, std::shared_ptr<Partition> partition)
    : NeuronToSubdomainAssignment(partition)
    , um_per_neuron_(um_per_neuron) {

    RelearnException::check(partition->get_my_mpi_rank() == 0 && partition->get_number_mpi_ranks() == 1, "SubdomainFromNeuronDensity::SubdomainFromNeuronDensity: Can only be used for 1 MPI rank.");

    RelearnException::check(fraction_excitatory_neurons >= 0.0 && fraction_excitatory_neurons <= 1.0,
        "SubdomainFromNeuronDensity::SubdomainFromNeuronDensity: The requested fraction of excitatory neurons is not in [0.0, 1.0]: {}", fraction_excitatory_neurons);
    RelearnException::check(um_per_neuron > 0.0, "SubdomainFromNeuronDensity::SubdomainFromNeuronDensity: The requested um per neuron is <= 0.0: {}", um_per_neuron);

    RelearnException::check(number_neurons > 0, "SubdomainFromNeuronDensity::SubdomainFromNeuronDensity: There must be more than 0 neurons.");

    RandomHolder::seed(RandomHolderKey::Subdomain, 0);

    // Calculate size of simulation box based on neuron density
    // number_neurons^(1/3) == #neurons per dimension
    const auto approx_number_of_neurons_per_dimension = ceil(pow(static_cast<double>(number_neurons), 1. / 3));
    const auto simulation_box_length_ = approx_number_of_neurons_per_dimension * um_per_neuron;

    partition->set_simulation_box_size({ 0, 0, 0 }, box_size_type(simulation_box_length_));

    set_requested_ratio_excitatory_neurons(fraction_excitatory_neurons);
    set_requested_number_neurons(number_neurons);

    set_ratio_placed_excitatory_neurons(0.0);
    set_number_placed_neurons(0);

    synapse_loader = std::make_shared<RandomSynapseLoader>(std::move(partition));
}

void SubdomainFromNeuronDensity::place_neurons_in_area(
    const box_size_type& offset,
    const box_size_type& length_of_box,
    const size_t number_neurons, const size_t subdomain_index_1d) {

    constexpr uint16_t max_short = std::numeric_limits<uint16_t>::max();

    const auto& [min, max] = partition->get_simulation_box_size();
    const auto& simulation_box_length_ = (max - min).get_maximum();

    RelearnException::check(length_of_box.get_x() <= simulation_box_length_ && length_of_box.get_y() <= simulation_box_length_ && length_of_box.get_z() <= simulation_box_length_,
        "SubdomainFromNeuronDensity::place_neurons_in_area: Requesting to fill neurons where no simulationbox is");

    const auto box = length_of_box - offset;

    const auto neurons_on_x = static_cast<size_t>(round(box.get_x() / um_per_neuron_));
    const auto neurons_on_y = static_cast<size_t>(round(box.get_y() / um_per_neuron_));
    const auto neurons_on_z = static_cast<size_t>(round(box.get_z() / um_per_neuron_));

    const auto calculated_num_neurons = neurons_on_x * neurons_on_y * neurons_on_z;
    RelearnException::check(calculated_num_neurons >= number_neurons, "SubdomainFromNeuronDensity::place_neurons_in_area: Should emplace more neurons than space in box");
    RelearnException::check(neurons_on_x <= max_short && neurons_on_y <= max_short && neurons_on_z <= max_short, "SubdomainFromNeuronDensity::place_neurons_in_area: Should emplace more neurons in a dimension than possible");

    const double desired_ex = get_requested_ratio_excitatory_neurons();

    const size_t expected_number_in = number_neurons - static_cast<size_t>(ceil(static_cast<double>(number_neurons) * desired_ex));
    const size_t expected_number_ex = number_neurons - expected_number_in;

    size_t placed_neurons = 0;

    size_t random_counter = 0;
    std::vector<size_t> positions(calculated_num_neurons);
    for (size_t x_it = 0; x_it < neurons_on_x; x_it++) {
        for (size_t y_it = 0; y_it < neurons_on_y; y_it++) {
            for (size_t z_it = 0; z_it < neurons_on_z; z_it++) {
                size_t random_position = 0;
                // NOLINTNEXTLINE
                random_position |= (z_it);
                // NOLINTNEXTLINE
                random_position |= (y_it << 16U);
                // NOLINTNEXTLINE
                random_position |= (x_it << 32U);
                positions[random_counter] = random_position;
                random_counter++;
            }
        }
    }

    RandomHolder::shuffle(RandomHolderKey::Subdomain, positions.begin(), positions.end());

    std::vector<SignalType> signal_types(expected_number_ex, SignalType::Excitatory);
    signal_types.insert(signal_types.cend(), expected_number_in, SignalType::Inhibitory);

    RandomHolder::shuffle(RandomHolderKey::Subdomain, signal_types.begin(), signal_types.end());

    Nodes nodes{};
    nodes.reserve(number_neurons);

    for (size_t i = 0; i < number_neurons; i++) {
        const size_t pos_bitmask = positions[i];
        const size_t x_it = (pos_bitmask >> 32U) & max_short;
        const size_t y_it = (pos_bitmask >> 16U) & max_short;
        const size_t z_it = pos_bitmask & max_short;

        const box_size_type::value_type x_pos_rnd = RandomHolder::get_random_uniform_double(RandomHolderKey::Subdomain, 0.0, 1.0) + static_cast<double>(x_it);
        const box_size_type::value_type y_pos_rnd = RandomHolder::get_random_uniform_double(RandomHolderKey::Subdomain, 0.0, 1.0) + static_cast<double>(y_it);
        const box_size_type::value_type z_pos_rnd = RandomHolder::get_random_uniform_double(RandomHolderKey::Subdomain, 0.0, 1.0) + static_cast<double>(z_it);

        box_size_type pos_rnd{ x_pos_rnd, y_pos_rnd, z_pos_rnd };
        pos_rnd *= um_per_neuron_;

        const box_size_type pos = pos_rnd + offset;
        const auto signal_type = signal_types[i];

        nodes.emplace_back(pos, NeuronID{ false, i }, signal_type, "random");

        placed_neurons++;
    }

    const auto current_num_neurons = get_number_placed_neurons();
    const auto former_ex_neurons = static_cast<double>(current_num_neurons) * get_ratio_placed_excitatory_neurons();

    const auto new_num_neurons = current_num_neurons + placed_neurons;

    set_number_placed_neurons(new_num_neurons);

    const auto now_ex_neurons = former_ex_neurons + static_cast<double>(expected_number_ex);
    // const auto now_in_neurons = former_in_neurons + placed_in_neurons;

    const auto current_frac_ex = static_cast<double>(now_ex_neurons) / static_cast<double>(new_num_neurons);

    set_ratio_placed_excitatory_neurons(current_frac_ex);
    set_nodes_for_subdomain(subdomain_index_1d, std::move(nodes));
}

void SubdomainFromNeuronDensity::fill_subdomain(const size_t local_subdomain_index, [[maybe_unused]] const size_t total_number_subdomains) {
    const auto number_subdomains = partition->get_total_number_subdomains();
    RelearnException::check(number_subdomains == 1, "SubdomainFromNeuronDensity::fill_subdomain: The total number of subdomains was not 1 but {}.", number_subdomains);

    const auto subdomain_index_1d = partition->get_1d_index_of_subdomain(local_subdomain_index);
    const bool subdomain_already_filled = is_subdomain_loaded(subdomain_index_1d);
    if (subdomain_already_filled) {
        RelearnException::fail("SubdomainFromNeuronDensity::fill_subdomain: Tried to fill an already filled subdomain.");
        return;
    }

    const auto& [min, max] = partition->get_subdomain_boundaries(local_subdomain_index);
    const auto requested_number_neurons = get_requested_number_neurons();
    place_neurons_in_area(min, max, requested_number_neurons, 0);
}

void SubdomainFromNeuronDensity::calculate_total_number_neurons() const {
    const auto number_local_neurons = get_number_placed_neurons();
    set_total_number_placed_neurons(number_local_neurons);
}
