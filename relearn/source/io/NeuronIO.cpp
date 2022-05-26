#include "NeuronIO.h"

#include "util/RelearnException.h"

#include "spdlog/spdlog.h"

#include <cmath>
#include <climits>
#include <fstream>
#include <iomanip>
#include <sstream>

std::tuple<std::vector<LoadedNeuron>, LoadedNeuronsInfo> NeuronIO::read_neurons(const std::filesystem::path& file_path) {
    std::ifstream file(file_path);

    const auto file_is_good = file.good();
    const auto file_is_not_good = file.fail() || file.eof();

    RelearnException::check(file_is_good && !file_is_not_good, "NeuronIO::read_neurons: Opening the file was not successful");

    position_type minimum(std::numeric_limits<position_type::value_type>::max());
    position_type maximum(std::numeric_limits<position_type::value_type>::min());

    size_t found_ex_neurons = 0;
    size_t found_in_neurons = 0;

    std::vector<LoadedNeuron> nodes{};

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

        if (signal_type == "in") {
            found_in_neurons++;
            nodes.emplace_back(position, NeuronID{ false, id }, SignalType::Inhibitory, std::move(area_name));
        } else {
            found_ex_neurons++;
            nodes.emplace_back(position, NeuronID{ false, id }, SignalType::Excitatory, std::move(area_name));
        }
    }

    const auto new_max_x = std::nextafter(maximum.get_x(), maximum.get_x() + Constants::eps);
    const auto new_max_y = std::nextafter(maximum.get_y(), maximum.get_y() + Constants::eps);
    const auto new_max_z = std::nextafter(maximum.get_z(), maximum.get_z() + Constants::eps);

    return { std::move(nodes), LoadedNeuronsInfo{ minimum, { new_max_x, new_max_y, new_max_z }, found_ex_neurons, found_in_neurons } };
}

std::tuple<std::vector<NeuronID>, std::vector<NeuronIO::position_type>, std::vector<std::string>, std::vector<SignalType>, LoadedNeuronsInfo>
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
    std::vector<std::string> area_names{};
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
        std::string area_name{};
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
        area_names.emplace_back(std::move(area_name));

        if (signal_type == "in") {
            found_in_neurons++;
            signal_types.emplace_back(SignalType::Inhibitory);
        } else {
            found_ex_neurons++;
            signal_types.emplace_back(SignalType::Excitatory);
        }
    }

    const auto new_max_x = std::nextafter(maximum.get_x(), maximum.get_x() + Constants::eps);
    const auto new_max_y = std::nextafter(maximum.get_y(), maximum.get_y() + Constants::eps);
    const auto new_max_z = std::nextafter(maximum.get_z(), maximum.get_z() + Constants::eps);

    return { std::move(ids), std::move(positions), std::move(area_names), std::move(signal_types), LoadedNeuronsInfo{ minimum, { new_max_x, new_max_y, new_max_z }, found_ex_neurons, found_in_neurons } };
}

void NeuronIO::write_neurons(const std::vector<LoadedNeuron>& neurons, const std::filesystem::path& file_path) {
    std::ofstream of(file_path, std::ios::binary | std::ios::out);

    const auto is_good = of.good();
    const auto is_bad = of.bad();

    RelearnException::check(is_good && !is_bad, "NeuronToSubdomainAssignment::write_neurons_to_file: The ofstream failed to open");

    of << std::setprecision(std::numeric_limits<double>::digits10);
    of << "# ID, Position (x y z),  Area,   type \n";

    for (const auto& node : neurons) {
        const auto id = node.id.get_local_id() + 1;
        const auto& [x, y, z] = node.pos;

        of << id << "\t"
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

void NeuronIO::write_neurons_componentwise(const std::vector<NeuronID>& ids, const std::vector<position_type>& positions,
    const std::vector<std::string>& area_names, const std::vector<SignalType>& signal_types, const std::filesystem::path& file_path) {

    const auto size_ids = ids.size();
    const auto size_positions = positions.size();
    const auto size_area_names = area_names.size();
    const auto size_signal_types = signal_types.size();

    const auto all_same_size = size_ids == size_positions && size_ids == size_area_names && size_ids == size_signal_types;

    RelearnException::check(all_same_size, "NeuronToSubdomainAssignment::write_neurons_componentwise: The vectors had different sizes.");

    std::ofstream of(file_path, std::ios::binary | std::ios::out);

    const auto is_good = of.good();
    const auto is_bad = of.bad();

    RelearnException::check(is_good && !is_bad, "NeuronToSubdomainAssignment::write_neurons_componentwise: The ofstream failed to open");

    of << std::setprecision(std::numeric_limits<double>::digits10);
    of << "# ID, Position (x y z),  Area,   type \n";

    for (size_t i = 0; i < size_ids; i++) {
        const auto id = ids[i].get_local_id() + 1;
        const auto& [x, y, z] = positions[i];

        of << id << "\t"
           << x << " "
           << y << " "
           << z << "\t"
           << area_names[i] << "\t";

        if (signal_types[i] == SignalType::Excitatory) {
            of << "ex\n";
        } else {
            of << "in\n";
        }
    }
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

        if (!ids.empty()) {
            const auto last_id = ids[ids.size() - 1].get_local_id();

            if (last_id >= id) {
                return {};
            }
        }

        ids.emplace_back(false, id);
    }

    return ids;
}

LocalSynapses NeuronIO::read_local_synapses(const std::filesystem::path& file_path, std::uint64_t number_local_neurons) {
    LocalSynapses local_synapses{};

    std::ifstream file_synapses(file_path, std::ios::binary | std::ios::in);

    for (std::string line{}; std::getline(file_synapses, line);) {
        // Skip line with comments
        if (line.empty() || '#' == line[0]) {
            continue;
        }

        NeuronID::value_type read_target_id = 0;
        NeuronID::value_type read_source_id = 0;
        RelearnTypes::synapse_weight weight = 0;

        std::stringstream sstream(line);
        const bool success = (sstream >> read_target_id) && (sstream >> read_source_id) && (sstream >> weight);

        if (!success) {
            spdlog::info("Skipping line: {}", line);
            continue;
        }

        RelearnException::check(read_target_id > 0 && read_target_id <= number_local_neurons, "NeuronIO::read_local_synapses: target_id was not from [1, {}]: {}", number_local_neurons, read_target_id);
        RelearnException::check(read_source_id > 0 && read_source_id <= number_local_neurons, "NeuronIO::read_local_synapses: source_id was not from [1, {}]: {}", number_local_neurons, read_source_id);

        RelearnException::check(weight != 0, "NeuronIO::read_local_synapses: weight was 0");

        // The neurons start with 1
        --read_source_id;
        --read_target_id;
        auto source_id = NeuronID{ false, read_source_id };
        auto target_id = NeuronID{ false, read_target_id };

        local_synapses.emplace_back(target_id, source_id, weight);
    }

    return local_synapses;
}
