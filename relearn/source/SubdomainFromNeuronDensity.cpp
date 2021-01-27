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

#include "MPIWrapper.h"
#include "RelearnException.h"

#include <limits>

SubdomainFromNeuronDensity::SubdomainFromNeuronDensity(size_t num_neurons, double desired_frac_neurons_exc, double um_per_neuron)
    : um_per_neuron_(um_per_neuron)
    , random_number_generator(RandomHolder::get_instance().get_random_generator(RandomHolder::SubdomainFromNeuronDensity))
    , random_number_distribution(0.0, 1.0) {

    unsigned int my_rank = static_cast<unsigned int>(MPIWrapper::get_my_rank());

    random_number_generator.seed(my_rank);

    // Calculate size of simulation box based on neuron density
    // num_neurons^(1/3) == #neurons per dimension
    const double approx_number_of_neurons_per_dimension = ceil(pow(static_cast<double>(num_neurons), 1. / 3));
    const double simulation_box_length_ = approx_number_of_neurons_per_dimension * um_per_neuron;

    this->simulation_box_length = Vec3d(simulation_box_length_);

    this->desired_frac_neurons_exc_ = desired_frac_neurons_exc;
    this->desired_num_neurons_ = num_neurons;

    this->currently_frac_neurons_exc_ = 0.0;
    this->currently_num_neurons_ = 0;
}

void SubdomainFromNeuronDensity::place_neurons_in_area(
    const NeuronToSubdomainAssignment::Position& offset,
    const NeuronToSubdomainAssignment::Position& length_of_box,
    size_t num_neurons, size_t subdomain_idx) {

    constexpr uint16_t max_short = std::numeric_limits<uint16_t>::max();

    const double simulation_box_length_ = get_simulation_box_length().get_maximum();

    RelearnException::check(length_of_box.get_x() <= simulation_box_length_ && length_of_box.get_y() <= simulation_box_length_ && length_of_box.get_z() <= simulation_box_length_,
        "Requesting to fill neurons where no simulationbox is");

    const auto box = length_of_box - offset;

    const auto neurons_on_x = static_cast<size_t>(round(box.get_x() / um_per_neuron_));
    const auto neurons_on_y = static_cast<size_t>(round(box.get_y() / um_per_neuron_));
    const auto neurons_on_z = static_cast<size_t>(round(box.get_z() / um_per_neuron_));

    const auto calculated_num_neurons = neurons_on_x * neurons_on_y * neurons_on_z;
    RelearnException::check(calculated_num_neurons >= num_neurons, "Should emplace more neurons than space in box");
    RelearnException::check(neurons_on_x <= max_short && neurons_on_y <= max_short && neurons_on_z <= max_short, "Should emplace more neurons in a dimension than possible");

    Nodes& nodes = neurons_in_subdomain[subdomain_idx];

    const size_t expected_number_in = num_neurons - static_cast<size_t>(ceil(num_neurons * desired_frac_neurons_exc_));
    const size_t expected_number_ex = num_neurons - expected_number_in;

    size_t placed_neurons = 0;
    size_t placed_in_neurons = 0;
    size_t placed_ex_neurons = 0;

    size_t random_counter = 0;
    std::vector<size_t> positions(calculated_num_neurons);
    for (size_t x_it = 0; x_it < neurons_on_x; x_it++) {
        for (size_t y_it = 0; y_it < neurons_on_y; y_it++) {
            for (size_t z_it = 0; z_it < neurons_on_z; z_it++) {
                size_t random_position = 0;
                random_position |= (z_it);
                random_position |= (y_it << 16U);
                random_position |= (x_it << 32U);
                positions[random_counter] = random_position;
                random_counter++;
            }
        }
    }

    std::shuffle(positions.begin(), positions.end(), random_number_generator);

    for (size_t i = 0; i < num_neurons; i++) {
        const size_t pos_bitmask = positions[i];
        const size_t x_it = (pos_bitmask >> 32U) & max_short;
        const size_t y_it = (pos_bitmask >> 16U) & max_short;
        const size_t z_it = pos_bitmask & max_short;

        const double x_pos_rnd = random_number_distribution(random_number_generator) + x_it;
        const double y_pos_rnd = random_number_distribution(random_number_generator) + y_it;
        const double z_pos_rnd = random_number_distribution(random_number_generator) + z_it;

        Position pos_rnd{ x_pos_rnd, y_pos_rnd, z_pos_rnd };
        pos_rnd *= um_per_neuron_;

        const Position pos = pos_rnd + offset;

        const double type_indicator = random_number_distribution(random_number_generator);

        if (placed_ex_neurons < expected_number_ex && (type_indicator < desired_frac_neurons_exc_ || placed_in_neurons == expected_number_in)) {
            Node node{ pos, i, SignalType::EXCITATORY, "random" };
            placed_ex_neurons++;
            nodes.emplace(node);
        } else {
            Node node{ pos, i, SignalType::INHIBITORY, "random" };
            placed_in_neurons++;
            nodes.emplace(node);
        }

        placed_neurons++;

        if (placed_neurons == num_neurons) {
            const auto former_ex_neurons = this->currently_num_neurons_ * this->currently_frac_neurons_exc_;
            //const auto former_in_neurons = this->currently_num_neurons_ - former_ex_neurons;

            this->currently_num_neurons_ += placed_neurons;

            const auto now_ex_neurons = former_ex_neurons + placed_ex_neurons;
            //const auto now_in_neurons = former_in_neurons + placed_in_neurons;

            currently_frac_neurons_exc_ = static_cast<double>(now_ex_neurons) / static_cast<double>(this->currently_num_neurons_);
            return;
        }
    }

    RelearnException::fail("In SubdomainFromNeuronDensity, shouldn't be here");
}

void SubdomainFromNeuronDensity::fill_subdomain(size_t subdomain_idx, [[maybe_unused]] size_t num_subdomains, const Position& min, const Position& max) {
    const bool subdomain_already_filled = neurons_in_subdomain.find(subdomain_idx) != neurons_in_subdomain.end();
    if (subdomain_already_filled) {
        RelearnException::fail("Tried to fill an already filled subdomain.");
        return;
    }

    const auto diff = max - min;
    const auto volume = diff.get_volume();

    const auto total_volume = get_simulation_box_length().get_volume();

    const auto neuron_portion = total_volume / volume;
    const auto neurons_in_subdomain_count = static_cast<size_t>(round(desired_num_neurons_ / neuron_portion));

    place_neurons_in_area(min, max, neurons_in_subdomain_count, subdomain_idx);
}

void SubdomainFromNeuronDensity::neuron_global_ids([[maybe_unused]] size_t subdomain_idx, [[maybe_unused]] size_t num_subdomains,
    [[maybe_unused]] size_t local_id_start, [[maybe_unused]] size_t local_id_end, [[maybe_unused]] std::vector<size_t>& global_ids) const {
}

std::tuple<SubdomainFromNeuronDensity::Position, SubdomainFromNeuronDensity::Position> SubdomainFromNeuronDensity::get_subdomain_boundaries(
    const Vec3s& subdomain_3idx,
    size_t num_subdomains_per_axis) const noexcept {
    const auto length = get_simulation_box_length().get_maximum();
    const auto one_subdomain_length = length / num_subdomains_per_axis;

    auto min = static_cast<Vec3d>(subdomain_3idx) * one_subdomain_length;
    auto max = static_cast<Vec3d>(subdomain_3idx + 1) * one_subdomain_length;

    min.round_to_larger_multiple(um_per_neuron_);
    max.round_to_larger_multiple(um_per_neuron_);

    return std::make_tuple(min, max);
}

std::tuple<SubdomainFromNeuronDensity::Position, SubdomainFromNeuronDensity::Position> SubdomainFromNeuronDensity::get_subdomain_boundaries(
    const Vec3s& subdomain_3idx,
    const Vec3s& num_subdomains_per_axis) const noexcept {

    const auto length = get_simulation_box_length().get_maximum();
    const auto x_subdomain_length = length / num_subdomains_per_axis.get_x();
    const auto y_subdomain_length = length / num_subdomains_per_axis.get_y();
    const auto z_subdomain_length = length / num_subdomains_per_axis.get_z();

    Vec3d min{ subdomain_3idx.get_x() * x_subdomain_length, subdomain_3idx.get_y() * y_subdomain_length, subdomain_3idx.get_z() * z_subdomain_length };

    const auto next_x = static_cast<double>(subdomain_3idx.get_x() + 1) * x_subdomain_length;
    const auto next_y = static_cast<double>(subdomain_3idx.get_y() + 1) * y_subdomain_length;
    const auto next_z = static_cast<double>(subdomain_3idx.get_z() + 1) * z_subdomain_length;

    Vec3d max{ next_x, next_y, next_z };

    min.round_to_larger_multiple(um_per_neuron_);
    max.round_to_larger_multiple(um_per_neuron_);

    return std::make_tuple(min, max);
}
