/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "NeuronIO.h"

#include "neurons/LocalAreaTranslator.h"
#include "structure/Partition.h"
#include "util/RelearnException.h"

#include "spdlog/spdlog.h"

#include <climits>
#include <tuple>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <iostream>

std::vector<std::string> NeuronIO::read_comments(const std::filesystem::path& file_path) {
    std::ifstream file(file_path);

    const auto file_is_good = file.good();
    const auto file_is_not_good = file.fail() || file.eof();

    RelearnException::check(file_is_good && !file_is_not_good, "NeuronIO::read_comments: Opening the file was not successful");

    std::vector<std::string> comments{};

    for (std::string line{}; std::getline(file, line);) {
        if (line.empty()) {
            continue;
        }

        if (line[0] == '#') {
            comments.emplace_back(std::move(line));
            continue;
        }

        break;
    }

    return comments;
}

std::tuple<std::vector<LoadedNeuron>, std::vector<RelearnTypes::area_name>, LoadedNeuronsInfo> NeuronIO::read_neurons(const std::filesystem::path& file_path) {
    std::ifstream file(file_path);

    const auto file_is_good = file.good();
    const auto file_is_not_good = file.fail() || file.eof();

    RelearnException::check(file_is_good && !file_is_not_good, "NeuronIO::read_neurons: Opening the file was not successful");

    position_type minimum(std::numeric_limits<position_type::value_type>::max());
    position_type maximum(std::numeric_limits<position_type::value_type>::min());

    size_t found_ex_neurons = 0;
    size_t found_in_neurons = 0;

    std::vector<LoadedNeuron> nodes{};

    number_neurons_type expected_id = 0;

    std::vector<RelearnTypes::area_name> area_names{};

    for (std::string line{}; std::getline(file, line);) {
        // Skip line with comments
        if (line.empty() || '#' == line[0]) {
            continue;
        }

        NeuronID::value_type id{};
        position_type::value_type pos_x{};
        position_type::value_type pos_y{};
        position_type::value_type pos_z{};
        std::string area_name{};
        std::string signal_type{};

        std::stringstream sstream(line);
        const auto success = (sstream >> id) && (sstream >> pos_x) && (sstream >> pos_y) && (sstream >> pos_z) && (sstream >> area_name) && (sstream >> signal_type);

        if (!success) {
            spdlog::info("Skipping line: {}", line);
            continue;
        }

        RelearnException::check(pos_x >= 0, "NeuronIO::read_neurons: x position of neuron {} was negative: {}", id, pos_x);
        RelearnException::check(pos_y >= 0, "NeuronIO::read_neurons: y position of neuron {} was negative: {}", id, pos_y);
        RelearnException::check(pos_z >= 0, "NeuronIO::read_neurons: z position of neuron {} was negative: {}", id, pos_z);

        id--;

        RelearnException::check(id == expected_id, "NeuronIO::read_neurons: Loaded neuron with id {} but expected: {}", id, expected_id);

        expected_id++;

        position_type position{ pos_x, pos_y, pos_z };

        minimum.calculate_componentwise_minimum(position);
        maximum.calculate_componentwise_maximum(position);

        auto area_id_it = std::find(area_names.begin(), area_names.end(), area_name);
        RelearnTypes::area_id area_id{ 0 };
        if (area_id_it == area_names.end()) {
            // Area name not known
            area_names.emplace_back(std::move(area_name));
            area_id = area_names.size() - 1;
        } else {
            area_id = area_id_it - area_names.begin();
        }

        if (signal_type == "in") {
            found_in_neurons++;
            nodes.emplace_back(position, NeuronID{ false, id }, SignalType::Inhibitory, area_id);
        } else {
            found_ex_neurons++;
            nodes.emplace_back(position, NeuronID{ false, id }, SignalType::Excitatory, area_id);
        }
    }

    return { std::move(nodes), std::move(area_names), LoadedNeuronsInfo{ minimum, maximum, found_ex_neurons, found_in_neurons } };
}

std::tuple<std::vector<NeuronID>, std::vector<NeuronIO::position_type>, std::vector<RelearnTypes::area_id>, std::vector<RelearnTypes::area_name>, std::vector<SignalType>, LoadedNeuronsInfo>
NeuronIO::read_neurons_componentwise(const std::filesystem::path& file_path) {

    std::ifstream file(file_path);

    const auto file_is_good = file.good();
    const auto file_is_not_good = file.fail() || file.eof();

    RelearnException::check(file_is_good && !file_is_not_good, "NeuronIO::read_neurons_componentwise: Opening the file was not successful");

    position_type minimum(std::numeric_limits<position_type::value_type>::max());
    position_type maximum(std::numeric_limits<position_type::value_type>::min());

    size_t found_ex_neurons = 0;
    size_t found_in_neurons = 0;

    std::vector<NeuronID> ids{};
    std::vector<position_type> positions{};
    std::vector<RelearnTypes::area_id> area_ids{};
    std::vector<RelearnTypes::area_name> area_names{};
    std::vector<SignalType> signal_types{};

    NeuronID::value_type expected_id = 0;

    for (std::string line{}; std::getline(file, line);) {
        // Skip line with comments
        if (line.empty() || '#' == line[0]) {
            continue;
        }

        NeuronID::value_type id{};
        position_type::value_type pos_x{};
        position_type::value_type pos_y{};
        position_type::value_type pos_z{};
        RelearnTypes::area_name area_name{};
        std::string signal_type{};

        std::stringstream sstream(line);
        const auto success = (sstream >> id) && (sstream >> pos_x) && (sstream >> pos_y) && (sstream >> pos_z) && (sstream >> area_name) && (sstream >> signal_type);

        if (!success) {
            spdlog::info("Skipping line: {}", line);
            continue;
        }

        RelearnException::check(pos_x >= 0, "NeuronIO::read_neurons_componentwise: x position of neuron {} was negative: {}", id, pos_x);
        RelearnException::check(pos_y >= 0, "NeuronIO::read_neurons_componentwise: y position of neuron {} was negative: {}", id, pos_y);
        RelearnException::check(pos_z >= 0, "NeuronIO::read_neurons_componentwise: z position of neuron {} was negative: {}", id, pos_z);

        id--;

        RelearnException::check(id == expected_id, "NeuronIO::read_neurons_componentwise: Loaded neuron with id {} but expected: {}", id, expected_id);

        expected_id++;

        position_type position{ pos_x, pos_y, pos_z };

        minimum.calculate_componentwise_minimum(position);
        maximum.calculate_componentwise_maximum(position);

        ids.emplace_back(false, id);
        positions.emplace_back(position);

        auto area_id_it = std::find(area_names.begin(), area_names.end(), area_name);
        RelearnTypes::area_id area_id{ 0 };
        if (area_id_it == area_names.end()) {
            // Area name not known
            area_names.emplace_back(std::move(area_name));
            area_id = area_names.size() - 1;
        } else {
            area_id = area_id_it - area_names.begin();
        }
        area_ids.emplace_back(area_id);

        if (signal_type == "in") {
            found_in_neurons++;
            signal_types.emplace_back(SignalType::Inhibitory);
        } else {
            found_ex_neurons++;
            signal_types.emplace_back(SignalType::Excitatory);
        }
    }

    return { std::move(ids), std::move(positions), std::move(area_ids), std::move(area_names), std::move(signal_types), LoadedNeuronsInfo{ minimum, maximum, found_ex_neurons, found_in_neurons } };
}

void NeuronIO::write_neurons(const std::vector<LoadedNeuron>& neurons, const std::filesystem::path& file_path, const std::shared_ptr<LocalAreaTranslator>& local_area_translator) {
    write_neurons(neurons, file_path, local_area_translator, nullptr);
}

void NeuronIO::write_neurons(const std::vector<LoadedNeuron>& neurons, const std::filesystem::path& file_path, const std::shared_ptr<LocalAreaTranslator>& local_area_translator, std::shared_ptr<Partition> partition) {
    std::stringstream ss{};
    write_neurons(neurons, ss, local_area_translator, partition);
    std::ofstream of(file_path, std::ios::binary | std::ios::out);

    const auto is_good = of.good();
    const auto is_bad = of.bad();

    RelearnException::check(is_good && !is_bad, "NeuronIO::write_neurons_to_file: The ofstream failed to open");

    of << ss.str();
    of.close();
}

void NeuronIO::write_neurons(const std::vector<LoadedNeuron>& neurons, std::stringstream& ss, const std::shared_ptr<LocalAreaTranslator>& local_area_translator, const std::shared_ptr<Partition>& partition) {
    if (partition != nullptr) {
        const auto& total_number_neurons = partition->get_total_number_neurons();

        const auto& [simulation_box_min, simulation_box_max] = partition->get_simulation_box_size();
        const auto& [min_x, min_y, min_z] = simulation_box_min;
        const auto& [max_x, max_y, max_z] = simulation_box_max;

        ss << std::setprecision(std::numeric_limits<double>::digits10);

        // Write total number of neurons to log file
        ss << "# " << neurons.size() << " of " << total_number_neurons << '\n';
        ss << "# Minimum x: " << min_x << '\n';
        ss << "# Minimum y: " << min_y << '\n';
        ss << "# Minimum z: " << min_z << '\n';
        ss << "# Maximum x: " << max_x << '\n';
        ss << "# Maximum y: " << max_y << '\n';
        ss << "# Maximum z: " << max_z << '\n';
        ss << "# <local id> <pos x> <pos y> <pos z> <area> <type>\n";

        const auto number_local_subdomains = partition->get_number_local_subdomains();
        const auto first_local_subdomain_index = partition->get_local_subdomain_id_start();
        const auto last_local_subdomain_index = partition->get_local_subdomain_id_end();

        ss << "# Local subdomain index start: " << first_local_subdomain_index << "\n";
        ss << "# Local subdomain index end: " << last_local_subdomain_index << "\n";
        ss << "# Number of local subdomains: " << number_local_subdomains << "\n";

        for (auto local_subdomain_index = 0; local_subdomain_index < number_local_subdomains; local_subdomain_index++) {
            const auto& [subdomain_bounding_box_min, subdomain_bounding_box_max] = partition->get_subdomain_boundaries(local_subdomain_index);
            ss << "# Local subdomain " << local_subdomain_index << " boundaries (" << subdomain_bounding_box_min.get_x() << ", " << subdomain_bounding_box_min.get_y() << ", " << subdomain_bounding_box_min.get_z() << ") - (";
            ss << subdomain_bounding_box_max.get_x() << ", " << subdomain_bounding_box_max.get_y() << ", " << subdomain_bounding_box_max.get_z() << ")\n";
        }
    }

    for (const auto& neuron : neurons) {
        const auto& [x, y, z] = neuron.pos;
        const auto& signal_type_name = (neuron.signal_type == SignalType::Excitatory) ? "ex" : "in";
        const auto& area_name = local_area_translator->get_area_name_for_neuron_id(neuron.id.get_neuron_id());

        ss << fmt::format("{1:<} {2:<.{0}} {3:<.{0}} {4:<.{0}} {5:<} {6:<}",
            Constants::print_precision, (neuron.id.get_neuron_id() + 1), x, y, z, area_name, signal_type_name)
           << '\n';
    }
}

void NeuronIO::write_area_names(std::stringstream& ss, const std::shared_ptr<LocalAreaTranslator>& local_area_translator) {
    ss << "# <area id>\t<ara_name>\t<num_neurons_in_area>\n";

    const auto num_areas = local_area_translator->get_number_of_areas();

    for (size_t area_id = 0; area_id < num_areas; area_id++) {
        const auto& area_name = local_area_translator->get_area_name_for_area_id(area_id);

        ss << area_id << '\t' << area_name << '\t' << local_area_translator->get_number_neurons_in_area(area_id) << '\n';
    }
}

void NeuronIO::write_neurons_componentwise(const std::vector<NeuronID>& ids, const std::vector<position_type>& positions,
    const std::shared_ptr<LocalAreaTranslator>& local_area_translator, const std::vector<SignalType>& signal_types, std::stringstream& ss, size_t total_number_neurons, const std::tuple<Vec3<double>, Vec3<double>>& simulation_box,
    const std::vector<std::pair<Partition::box_size_type, Partition::box_size_type>>& local_subdomain_boundaries) {

    const auto size_ids = ids.size();
    const auto size_positions = positions.size();
    const auto size_signal_types = signal_types.size();

    const auto all_same_size = size_ids == size_positions && size_ids == size_signal_types;

    RelearnException::check(all_same_size, "NeuronIO::write_neurons_componentwise: The vectors had different sizes.");

    // Write total number of neurons to log file
    if (total_number_neurons > 0) {
        ss << "# " << ids.size() << " of " << total_number_neurons << '\n';
    }

    ss << std::setprecision(std::numeric_limits<double>::digits10);
    const auto& [simulation_box_min, simulation_box_max] = simulation_box;
    if (simulation_box_min.get_x() != simulation_box_max.get_x()) {
        const auto& [min_x, min_y, min_z] = simulation_box_min;
        const auto& [max_x, max_y, max_z] = simulation_box_max;

        ss << "# Minimum x: " << min_x << '\n';
        ss << "# Minimum y: " << min_y << '\n';
        ss << "# Minimum z: " << min_z << '\n';
        ss << "# Maximum x: " << max_x << '\n';
        ss << "# Maximum y: " << max_y << '\n';
        ss << "# Maximum z: " << max_z << '\n';
        ss << "# <local id> <pos x> <pos y> <pos z> <area> <type>\n";
    }

    if (!local_subdomain_boundaries.empty()) {
        const auto number_local_subdomains = local_subdomain_boundaries.size();
        ss << "# Number of local subdomains: " << number_local_subdomains << "\n";

        for (auto local_subdomain_index = 0; local_subdomain_index < number_local_subdomains; local_subdomain_index++) {
            const auto& [subdomain_bounding_box_min, subdomain_bounding_box_max] = local_subdomain_boundaries[local_subdomain_index];
            ss << "# Local subdomain " << local_subdomain_index << " boundaries (" << subdomain_bounding_box_min.get_x() << ", " << subdomain_bounding_box_min.get_y() << ", " << subdomain_bounding_box_min.get_z() << ") - (";
            ss << subdomain_bounding_box_max.get_x() << ", " << subdomain_bounding_box_max.get_y() << ", " << subdomain_bounding_box_max.get_z() << ")\n";
        }
    }

    for (const auto& neuron_id : ids) {
        RelearnException::check(neuron_id.get_neuron_id() < ids.size(), "NeuronIO::write_neurons_componentwise: Neuron id {} is too large", neuron_id);
        const auto& [x, y, z] = positions[neuron_id.get_neuron_id()];
        const auto& signal_type_name = (signal_types[neuron_id.get_neuron_id()] == SignalType::Excitatory) ? "ex" : "in";
        const auto& area_name = local_area_translator->get_area_name_for_neuron_id(neuron_id.get_neuron_id());
        ss << (neuron_id.get_neuron_id() + 1) << " " << x << " " << y << " " << z << " " << area_name << " " << signal_type_name << '\n';
    }
}

void NeuronIO::write_neurons_componentwise(const std::vector<NeuronID>& ids, const std::vector<position_type>& positions,
    const std::shared_ptr<LocalAreaTranslator>& local_area_translator, const std::vector<SignalType>& signal_types, std::filesystem::path& file_path) {
    std::stringstream ss;
    write_neurons_componentwise(ids, positions, local_area_translator, signal_types, ss, 0, std::make_tuple(RelearnTypes::position_type({ 0.0, 0.0, 0.0 }), RelearnTypes::position_type({ 0.0, 0.0, 0.0 })), {});
    std::ofstream of(file_path, std::ios::binary | std::ios::out);

    const auto is_good = of.good();
    const auto is_bad = of.bad();

    RelearnException::check(is_good && !is_bad, "NeuronIO::write_neurons_to_file: The ofstream failed to open");

    of << ss.str();
    of.close();
}

std::optional<std::vector<NeuronID>> NeuronIO::read_neuron_ids(const std::filesystem::path& file_path) {
    std::ifstream local_file(file_path);

    const bool file_is_good = local_file.good();
    const bool file_is_not_good = local_file.fail() || local_file.eof();

    if (!file_is_good || file_is_not_good) {
        return {};
    }

    std::vector<NeuronID> ids{};

    for (std::string line{}; std::getline(local_file, line);) {
        // Skip line with comments
        if (line.empty() || '#' == line[0]) {
            continue;
        }

        NeuronID::value_type id{};
        position_type::value_type pos_x{};
        position_type::value_type pos_y{};
        position_type::value_type pos_z{};
        std::string area_name{};
        std::string signal_type{};

        std::stringstream sstream(line);
        const auto success = (sstream >> id) && (sstream >> pos_x) && (sstream >> pos_y) && (sstream >> pos_z) && (sstream >> area_name) && (sstream >> signal_type);

        if (!success) {
            return {};
        }

        id--;

        if (!ids.empty()) {
            const auto last_id = ids[ids.size() - 1].get_neuron_id();

            if (last_id + 1 != id) {
                return {};
            }
        }

        ids.emplace_back(false, id);
    }

    return ids;
}

std::pair<std::tuple<LocalSynapses, DistantInSynapses>, std::tuple<LocalSynapses, DistantInSynapses>> NeuronIO::read_in_synapses(const std::filesystem::path& file_path, number_neurons_type number_local_neurons, int my_rank, int number_mpi_ranks) {
    LocalSynapses local_in_synapses_static{};
    DistantInSynapses distant_in_synapses_static{};
    LocalSynapses local_in_synapses_plastic{};
    DistantInSynapses distant_in_synapses_plastic{};

    std::ifstream file_synapses(file_path, std::ios::binary | std::ios::in);

    const auto is_good = file_synapses.good();
    const auto is_bad = file_synapses.bad();

    RelearnException::check(is_good && !is_bad, "NeuronIO::read_in_synapses: The ofstream failed to open");

    for (std::string line{}; std::getline(file_synapses, line);) {
        // Skip line with comments
        if (line.empty() || '#' == line[0]) {
            continue;
        }

        int read_target_rank = 0;
        NeuronID::value_type read_target_id = 0;
        int read_source_rank = 0;
        NeuronID::value_type read_source_id = 0;
        RelearnTypes::synapse_weight weight = 0;
        bool plastic;

        std::stringstream sstream(line);
        const bool success = (sstream >> read_target_rank) && (sstream >> read_target_id) && (sstream >> read_source_rank) && (sstream >> read_source_id) && (sstream >> weight) && (sstream >> plastic);

        if (!success) {
            spdlog::info("Skipping line: {}", line);
            continue;
        }

        RelearnException::check(read_target_rank == my_rank, "NeuronIO::read_in_synapses: target_rank is not equal to my_rank: {} vs {}", read_target_rank, my_rank);
        RelearnException::check(read_target_id > 0 && read_target_id <= number_local_neurons, "NeuronIO::read_in_synapses: target_id was not from [1, {}]: {}", number_local_neurons, read_target_id);

        RelearnException::check(read_source_rank < number_mpi_ranks, "NeuronIO::read_in_synapses: source rank is not smaller than the number of mpi ranks: {} vs {}", read_source_rank, number_mpi_ranks);

        RelearnException::check(weight != 0, "NeuronIO::read_in_synapses: weight was 0");

        // The neurons start with 1
        --read_source_id;
        --read_target_id;

        auto source_id = NeuronID{ false, read_source_id };
        auto target_id = NeuronID{ false, read_target_id };

        if (read_source_rank != my_rank) {
            if (plastic) {
                distant_in_synapses_plastic.emplace_back(target_id, RankNeuronId{ read_source_rank, source_id }, weight);
            } else {
                distant_in_synapses_static.emplace_back(target_id, RankNeuronId{ read_source_rank, source_id }, weight);
            }
        } else {
            if (plastic) {
                local_in_synapses_plastic.emplace_back(target_id, source_id, weight);
            } else {
                local_in_synapses_static.emplace_back(target_id, source_id, weight);
            }
        }
    }

    return { { local_in_synapses_static, distant_in_synapses_static }, { local_in_synapses_plastic, distant_in_synapses_plastic } };
}

void NeuronIO::write_in_synapses(const LocalSynapses& local_in_synapses_static, const DistantInSynapses& distant_in_synapses_static, const LocalSynapses& local_in_synapses_plastic, const DistantInSynapses& distant_in_synapses_plastic, int my_rank, const std::filesystem::path& file_path) {
    std::ofstream file_synapses(file_path, std::ios::binary | std::ios::out);

    const auto is_good = file_synapses.good();
    const auto is_bad = file_synapses.bad();

    RelearnException::check(is_good && !is_bad, "NeuronIO::write_in_synapses: The ofstream failed to open");

    file_synapses << "# Number local in-sypases static: " << local_in_synapses_static.size() << '\n';
    file_synapses << "# Number distant in-sypases static: " << distant_in_synapses_static.size() << '\n';
    file_synapses << "# Number local in-sypases plastic: " << local_in_synapses_plastic.size() << '\n';
    file_synapses << "# Number distant in-sypases plastic: " << distant_in_synapses_plastic.size() << '\n';
    file_synapses << "# <target rank> <target neuron id>\t<source rank> <source neuron id>\t<weight>\n";

    for (const auto& [target_id, source_id, weight] : local_in_synapses_static) {
        const auto target_neuron_id = target_id.get_neuron_id();
        const auto source_neuron_id = source_id.get_neuron_id();

        file_synapses << my_rank << ' ' << (target_neuron_id + 1) << '\t' << my_rank << ' ' << (source_neuron_id + 1) << '\t' << weight << '\t' << '0' << '\n';
    }
    for (const auto& [target_id, source_id, weight] : local_in_synapses_plastic) {
        const auto target_neuron_id = target_id.get_neuron_id();
        const auto source_neuron_id = source_id.get_neuron_id();

        file_synapses << my_rank << ' ' << (target_neuron_id + 1) << '\t' << my_rank << ' ' << (source_neuron_id + 1) << '\t' << weight << '\t' << '1' << '\n';
    }

    for (const auto& [target_id, source_rni, weight] : distant_in_synapses_static) {
        const auto target_neuron_id = target_id.get_neuron_id();

        const auto& [source_rank, source_id] = source_rni;
        const auto source_neuron_id = source_id.get_neuron_id();

        RelearnException::check(source_rank != my_rank, "NeuronIO::write_distant_in_synapses: source rank was equal to my_rank: {}", my_rank);

        file_synapses << my_rank << ' ' << (target_neuron_id + 1) << '\t' << source_rank << ' ' << (source_neuron_id + 1) << '\t' << weight << '\t' << '0' << '\n';
    }
    for (const auto& [target_id, source_rni, weight] : distant_in_synapses_plastic) {
        const auto target_neuron_id = target_id.get_neuron_id();

        const auto& [source_rank, source_id] = source_rni;
        const auto source_neuron_id = source_id.get_neuron_id();

        RelearnException::check(source_rank != my_rank, "NeuronIO::write_distant_in_synapses: source rank was equal to my_rank: {}", my_rank);

        file_synapses << my_rank << ' ' << (target_neuron_id + 1) << '\t' << source_rank << ' ' << (source_neuron_id + 1) << '\t' << weight << '\t' << '1' << '\n';
    }
}

std::pair<std::tuple<LocalSynapses, DistantOutSynapses>, std::tuple<LocalSynapses, DistantOutSynapses>> NeuronIO::read_out_synapses(const std::filesystem::path& file_path, number_neurons_type number_local_neurons, int my_rank, int number_mpi_ranks) {
    LocalSynapses local_out_synapses_static{};
    DistantOutSynapses distant_out_synapses_static{};
    LocalSynapses local_out_synapses_plastic{};
    DistantOutSynapses distant_out_synapses_plastic{};

    std::ifstream file_synapses(file_path, std::ios::binary | std::ios::in);

    const auto is_good = file_synapses.good();
    const auto is_bad = file_synapses.bad();

    RelearnException::check(is_good && !is_bad, "NeuronIO::read_out_synapses: The ofstream failed to open");

    for (std::string line{}; std::getline(file_synapses, line);) {
        // Skip line with comments
        if (line.empty() || '#' == line[0]) {
            continue;
        }

        int read_target_rank = 0;
        NeuronID::value_type read_target_id = 0;
        int read_source_rank = 0;
        NeuronID::value_type read_source_id = 0;
        RelearnTypes::synapse_weight weight = 0;

        std::stringstream sstream(line);
        bool plastic;
        const bool success = (sstream >> read_target_rank) && (sstream >> read_target_id) && (sstream >> read_source_rank) && (sstream >> read_source_id) && (sstream >> weight) && (sstream >> plastic);

        if (!success) {
            spdlog::info("Skipping line: {}", line);
            continue;
        }

        RelearnException::check(read_source_rank == my_rank, "NeuronIO::read_out_synapses: source_rank is not equal to my_rank: {} vs {}", read_target_rank, my_rank);
        RelearnException::check(read_source_id > 0 && read_source_id <= number_local_neurons, "NeuronIO::read_out_synapses: source_id was not from [1, {}]: {}", number_local_neurons, read_source_id);

        RelearnException::check(read_target_rank < number_mpi_ranks, "NeuronIO::read_out_synapses: target rank is not smaller than the number of mpi ranks: {} vs {}", read_source_rank, number_mpi_ranks);

        RelearnException::check(weight != 0, "NeuronIO::read_out_synapses: weight was 0");

        // The neurons start with 1
        --read_source_id;
        --read_target_id;

        auto source_id = NeuronID{ false, read_source_id };
        auto target_id = NeuronID{ false, read_target_id };

        if (read_target_rank != my_rank) {
            if (plastic) {
                distant_out_synapses_plastic.emplace_back(RankNeuronId{ read_target_rank, target_id }, source_id, weight);
            } else {
                distant_out_synapses_static.emplace_back(RankNeuronId{ read_target_rank, target_id }, source_id, weight);
            }
        } else {
            if (plastic) {
                local_out_synapses_plastic.emplace_back(target_id, source_id, weight);
            } else {
                local_out_synapses_static.emplace_back(target_id, source_id, weight);
            }
        }
    }

    return { { local_out_synapses_static, distant_out_synapses_static }, { local_out_synapses_plastic, distant_out_synapses_plastic } };
}

void NeuronIO::write_out_synapses(const LocalSynapses& local_out_synapses_static, const DistantOutSynapses& distant_out_synapses_static, const LocalSynapses& local_out_synapses_plastic, const DistantOutSynapses& distant_out_synapses_plastic, int my_rank, const std::filesystem::path& file_path) {
    std::ofstream file_synapses(file_path, std::ios::binary | std::ios::out);

    const auto is_good = file_synapses.good();
    const auto is_bad = file_synapses.bad();

    RelearnException::check(is_good && !is_bad, "NeuronIO::write_distant_out_synapses: The ofstream failed to open");

    file_synapses << "# Number local out-sypases static: " << local_out_synapses_static.size() << '\n';
    file_synapses << "# Number distant out-sypases static: " << distant_out_synapses_static.size() << '\n';
    file_synapses << "# Number local out-sypases plastic: " << local_out_synapses_plastic.size() << '\n';
    file_synapses << "# Number distant out-sypases plastic: " << distant_out_synapses_plastic.size() << '\n';
    file_synapses << "# <target rank> <target neuron id>\t<source rank> <source neuron id>\t<weight>\n";

    for (const auto& [target_id, source_id, weight] : local_out_synapses_static) {
        const auto target_neuron_id = target_id.get_neuron_id();
        const auto source_neuron_id = source_id.get_neuron_id();

        file_synapses << my_rank << ' ' << (target_neuron_id + 1) << '\t' << my_rank << ' ' << (source_neuron_id + 1) << '\t' << weight << '\t' << '0' << '\n';
    }
    for (const auto& [target_id, source_id, weight] : local_out_synapses_plastic) {
        const auto target_neuron_id = target_id.get_neuron_id();
        const auto source_neuron_id = source_id.get_neuron_id();

        file_synapses << my_rank << ' ' << (target_neuron_id + 1) << '\t' << my_rank << ' ' << (source_neuron_id + 1) << '\t' << weight << '\t' << '1' << '\n';
    }

    for (const auto& [target_rni, source_id, weight] : distant_out_synapses_static) {
        const auto& [target_rank, target_id] = target_rni;
        const auto target_neuron_id = target_id.get_neuron_id();

        const auto source_neuron_id = source_id.get_neuron_id();

        RelearnException::check(target_rank != my_rank, "NeuronIO::write_distant_out_synapses: target rank was equal to my_rank: {}", my_rank);

        file_synapses << target_rank << ' ' << (target_neuron_id + 1) << '\t' << my_rank << ' ' << (source_neuron_id + 1) << '\t' << weight << '\t' << '0' << '\n';
    }
    for (const auto& [target_rni, source_id, weight] : distant_out_synapses_plastic) {
        const auto& [target_rank, target_id] = target_rni;
        const auto target_neuron_id = target_id.get_neuron_id();

        const auto source_neuron_id = source_id.get_neuron_id();

        RelearnException::check(target_rank != my_rank, "NeuronIO::write_distant_out_synapses: target rank was equal to my_rank: {}", my_rank);

        file_synapses << target_rank << ' ' << (target_neuron_id + 1) << '\t' << my_rank << ' ' << (source_neuron_id + 1) << '\t' << weight << '\t' << '1' << '\n';
    }
}
