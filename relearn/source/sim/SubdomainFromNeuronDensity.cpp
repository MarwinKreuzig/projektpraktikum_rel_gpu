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

#include "../mpi/MPIWrapper.h"
#include "../structure/Partition.h"
#include "../structure/SynapseLoader.h"
#include "../util/Random.h"
#include "../util/RelearnException.h"

#include <limits>
#include <numeric>

SubdomainFromNeuronDensity::SubdomainFromNeuronDensity(const size_t number_neurons, const double fraction_excitatory_neurons, const double um_per_neuron, std::shared_ptr<Partition> partition)
    : NeuronToSubdomainAssignment(partition)
    , um_per_neuron_(um_per_neuron) {

    RelearnException::check(fraction_excitatory_neurons >= 0.0 && fraction_excitatory_neurons <= 1.0,
        "SubdomainFromNeuronDensity::SubdomainFromNeuronDensity: The requested fraction of excitatory neurons is not in [0.0, 1.0]: {}", fraction_excitatory_neurons);
    RelearnException::check(um_per_neuron > 0.0, "SubdomainFromNeuronDensity::SubdomainFromNeuronDensity: The requested um per neuron is <= 0.0: {}", um_per_neuron);

    RelearnException::check(number_neurons >= partition->get_total_number_subdomains(),
        "SubdomainFromNeuronDensity::SubdomainFromNeuronDensity: There are {} subdomains but only {} neurons.", partition->get_total_number_subdomains(), number_neurons);

    const auto my_rank = static_cast<unsigned int>(MPIWrapper::get_my_rank());

    RandomHolder::seed(RandomHolderKey::Subdomain, my_rank);

    // Calculate size of simulation box based on neuron density
    // number_neurons^(1/3) == #neurons per dimension
    const auto approx_number_of_neurons_per_dimension = ceil(pow(static_cast<double>(number_neurons), 1. / 3));
    const auto simulation_box_length_ = approx_number_of_neurons_per_dimension * um_per_neuron;

    partition->set_simulation_box_size({ 0, 0, 0 }, box_size_type(simulation_box_length_));

    set_requested_ratio_excitatory_neurons(fraction_excitatory_neurons);
    set_requested_number_neurons(number_neurons);

    set_ratio_placed_excitatory_neurons(0.0);
    set_number_placed_neurons(0);
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

    Nodes nodes{};

    const double desired_ex = get_requested_ratio_excitatory_neurons();

    const size_t expected_number_in = number_neurons - static_cast<size_t>(ceil(number_neurons * desired_ex));
    const size_t expected_number_ex = number_neurons - expected_number_in;

    size_t placed_neurons = 0;
    size_t placed_in_neurons = 0;
    size_t placed_ex_neurons = 0;

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

    for (size_t i = 0; i < number_neurons; i++) {
        const size_t pos_bitmask = positions[i];
        const size_t x_it = (pos_bitmask >> 32U) & max_short;
        const size_t y_it = (pos_bitmask >> 16U) & max_short;
        const size_t z_it = pos_bitmask & max_short;

        const box_size_type::value_type x_pos_rnd = RandomHolder::get_random_uniform_double(RandomHolderKey::Subdomain, 0.0, 1.0) + x_it;
        const box_size_type::value_type y_pos_rnd = RandomHolder::get_random_uniform_double(RandomHolderKey::Subdomain, 0.0, 1.0) + y_it;
        const box_size_type::value_type z_pos_rnd = RandomHolder::get_random_uniform_double(RandomHolderKey::Subdomain, 0.0, 1.0) + z_it;

        box_size_type pos_rnd{ x_pos_rnd, y_pos_rnd, z_pos_rnd };
        pos_rnd *= um_per_neuron_;

        const box_size_type pos = pos_rnd + offset;

        const double type_indicator = RandomHolder::get_random_uniform_double(RandomHolderKey::Subdomain, 0.0, 1.0);

        if (placed_ex_neurons < expected_number_ex && (type_indicator < desired_ex || placed_in_neurons == expected_number_in)) {
            Node node{ pos, i, SignalType::EXCITATORY, "random" };
            placed_ex_neurons++;
            nodes.emplace(node);
        } else {
            Node node{ pos, i, SignalType::INHIBITORY, "random" };
            placed_in_neurons++;
            nodes.emplace(node);
        }

        placed_neurons++;

        if (placed_neurons == number_neurons) {
            const auto current_num_neurons = get_number_placed_neurons();
            const auto former_ex_neurons = current_num_neurons * get_ratio_placed_excitatory_neurons();

            const auto new_num_neurons = current_num_neurons + placed_neurons;

            set_number_placed_neurons(new_num_neurons);

            const auto now_ex_neurons = former_ex_neurons + placed_ex_neurons;
            //const auto now_in_neurons = former_in_neurons + placed_in_neurons;

            const auto current_frac_ex = static_cast<double>(now_ex_neurons) / static_cast<double>(new_num_neurons);

            set_ratio_placed_excitatory_neurons(current_frac_ex);
            set_nodes_for_subdomain(subdomain_index_1d, std::move(nodes));
            return;
        }
    }

    RelearnException::fail("SubdomainFromNeuronDensity::place_neurons_in_area: Subdomain {} does not have enough neurons: {}!", subdomain_index_1d, number_neurons);
}

void SubdomainFromNeuronDensity::fill_subdomain(const size_t local_subdomain_index, [[maybe_unused]] const size_t total_number_subdomains) {
    const auto subdomain_index_1d = partition->get_1d_index_of_subdomain(local_subdomain_index);
    const bool subdomain_already_filled = is_subdomain_loaded(subdomain_index_1d);
    if (subdomain_already_filled) {
        RelearnException::fail("SubdomainFromNeuronDensity::fill_subdomain: Tried to fill an already filled subdomain.");
        return;
    }

    const auto& [min, max] = partition->get_subdomain_boundaries(local_subdomain_index);
    const auto requested_number_neurons = get_requested_number_neurons();
    const auto number_subdomains = partition->get_total_number_subdomains();

    if (number_subdomains == 1) {
        place_neurons_in_area(min, max, requested_number_neurons, 0);
        return;
    }

    const auto volume = (max - min).get_volume();
    const auto number_boxes = volume / (um_per_neuron_ * um_per_neuron_ * um_per_neuron_);

    const auto& [sim_box_min, sim_box_max] = partition->get_simulation_box_size();
    const auto total_volume = (sim_box_max - sim_box_min).get_volume();
    const auto total_number_boxes = total_volume / (um_per_neuron_ * um_per_neuron_ * um_per_neuron_);

    const auto free_neurons = requested_number_neurons - number_subdomains;
    const auto proposed_number_local_neurons = floor(free_neurons * number_boxes / total_number_boxes);
  
    const auto adjustment_neuron = (proposed_number_local_neurons == 0) ? 2 : 1;

    const auto neurons_in_subdomain_count = proposed_number_local_neurons + adjustment_neuron;
    const auto final_neuron_count = std::min(std::round(number_boxes), neurons_in_subdomain_count);

    place_neurons_in_area(min, max, final_neuron_count, subdomain_index_1d);
}

void SubdomainFromNeuronDensity::calculate_total_number_neurons() const {
    const auto number_local_neurons = get_number_placed_neurons();
    const auto num_ranks = MPIWrapper::get_num_ranks();

    std::vector<size_t> all_number_local_neurons(num_ranks, 0);

    MPIWrapper::all_gather(number_local_neurons, all_number_local_neurons);

    const auto total_number = std::reduce(all_number_local_neurons.begin(), all_number_local_neurons.end(), 0, std::plus{});
    set_total_number_placed_neurons(total_number);
}

std::vector<size_t> SubdomainFromNeuronDensity::get_neuron_global_ids_in_subdomain([[maybe_unused]] const size_t subdomain_index_1d, [[maybe_unused]] const size_t total_number_subdomains) const {

    return {};
}

void SubdomainFromNeuronDensity::post_initialization() {
    neuron_id_translator = std::make_shared<RandomNeuronIdTranslator>(partition);
    synapse_loader = std::make_shared<RandomSynapseLoader>(partition, neuron_id_translator);
}
