/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "io/NeuronIO.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <filesystem>
#include <vector>

void compute_all_distances_fixed_number_bins(std::filesystem::path neuron_file, unsigned int number_bins) {
    const auto& [ids, positions, area_names, signal_types, infos] = NeuronIO::read_neurons_componentwise(neuron_file);
    const auto number_neurons = positions.size();
    
    std::cout << "Found " << number_neurons << " neurons.\n";

    auto min = positions[0];
    auto max = positions[0];

    for (const auto& pos : positions) {
        min.calculate_componentwise_minimum(pos);
        max.calculate_componentwise_maximum(pos);
    }

    std::cout << "Minimum position is: " << min << '\n';
    std::cout << "Maximum position is: " << max << '\n';
    
    const auto max_distance = (max - min).calculate_2_norm();
    const auto bin_width = max_distance / static_cast<double>(number_bins);

    std::vector<double> upper_borders(number_bins, 0.0);
    for (auto i = 0; i < number_bins; i++) {
        const auto border = static_cast<double>(i) * bin_width;
        upper_borders[i] = border;
    }
    upper_borders[number_bins - 1] = std::numeric_limits<double>::infinity();

    std::vector<size_t> counts(number_bins, 0);

    for (auto i = 0; i < number_neurons; i++) {
        const auto& source_position = positions[i];
        const auto offset = i * number_neurons;

        for (auto j = 0; j < number_neurons; j++) {
            if (i == j) {
                continue;
            }

            const auto& target_position = positions[j];
            const auto& difference = source_position - target_position;

            const auto distance = difference.calculate_2_norm();

            for (auto boundary_id = 0; boundary_id < number_bins; boundary_id++) {
                const auto boundary = upper_borders[boundary_id];
                if (boundary > distance) {
                    counts[boundary_id]++;
                    break;
                }
            }
        }

        std::cout << "Finished " << (i + 1) << " of " << number_neurons << " neurons.\n";
    }

    auto print = [&](std::ostream& out) {
        out << std::setprecision(6);
        for (auto i = 1; i < number_bins; i++) {
            out << '[' << upper_borders[i - 1] << ", " << upper_borders[i] << "): " << counts[i - 1] << '\n';
        }
    };

    print(std::cout);
}

void compute_all_distances(std::filesystem::path neuron_file) {
    const auto& [ids, positions, area_names, signal_types, infos] = NeuronIO::read_neurons_componentwise(neuron_file);
    const auto number_neurons = positions.size();

    std::vector<double> pairwise_distances(number_neurons * number_neurons, 0.0);

    for (auto i = 0; i < number_neurons; i++) {
        const auto& source_position = positions[i];
        const auto offset = i * number_neurons;

        for (auto j = 0; j < number_neurons; j++) {
            if (i == j) {
                pairwise_distances[offset + j] = std::numeric_limits<double>::infinity();
                continue;
            }

            const auto& target_position = positions[j];
            const auto& difference = source_position - target_position;

            const auto norm = difference.calculate_2_norm();

            pairwise_distances[offset + j] = norm;
        }
    }

    std::sort(pairwise_distances.begin(), pairwise_distances.end());

    pairwise_distances.erase(pairwise_distances.end() - number_neurons, pairwise_distances.end());

    const auto number_distances = pairwise_distances.size();

    const auto quartile_25_index = 0.25 * static_cast<double>(number_distances);
    const auto quartile_75_index = 0.75 * static_cast<double>(number_distances);

    const auto quartile_25 = pairwise_distances[quartile_25_index];
    const auto quartile_75 = pairwise_distances[quartile_75_index];

    const auto interquartile_range = quartile_75 - quartile_25;
    const auto double_interquartile_range = 2 * interquartile_range;

    const auto dubios_length = std::pow(number_distances, -1.0 / 3.0);

    const auto bin_width = double_interquartile_range * dubios_length;

    const auto min_distance = pairwise_distances[0];
    const auto max_distance = pairwise_distances[number_distances - 1];

    const auto number_of_bins = (max_distance - min_distance) / bin_width;
    const auto number_of_bins_cast = static_cast<unsigned int>(number_of_bins);

    std::vector<double> upper_borders(number_of_bins_cast, 0.0);
    for (auto i = 0; i < number_of_bins_cast; i++) {
        const auto border = min_distance + static_cast<double>(i) * bin_width;
        upper_borders[i] = border;
    }
    upper_borders[number_of_bins_cast - 1] = std::numeric_limits<double>::infinity();

    std::vector<size_t> counts(number_of_bins_cast, 0);
    for (const auto distance : pairwise_distances) {
        for (auto boundary_id = 0; boundary_id < number_of_bins_cast; boundary_id++) {
            const auto boundary = upper_borders[boundary_id];
            if (boundary > distance) {
                counts[boundary_id]++;
                break;
            }
        }
    }

    auto print = [&](std::ostream& out) {
        out << std::setprecision(6);
        for (auto i = 1; i < number_of_bins_cast; i++) {
            out << '[' << upper_borders[i - 1] << ", " << upper_borders[i] << "): " << counts[i - 1] << '\n';
        }
    };

    print(std::cout);
}

int main(int argc, char** argv) {
    if (argc == 1) {
        std::cerr << "Please pass arguments!\n";
        return 1;
    }

    const auto& path_426124_nodes = std::filesystem::path{ argv[1] };

    compute_all_distances_fixed_number_bins(path_426124_nodes, 10000);
    return 0;
}
