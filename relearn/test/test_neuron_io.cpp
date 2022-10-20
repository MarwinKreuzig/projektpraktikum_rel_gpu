#include "../googletest/include/gtest/gtest.h"

#include "RelearnTest.hpp"

#include "io/NeuronIO.h"

TEST_F(IOTest, testNeuronIOWriteComponentwiseSizeExceptions) {
    const auto number_neurons = get_random_number_neurons();

    const auto correct_ids = std::vector<NeuronID>{ number_neurons };
    const auto correct_position = std::vector<RelearnTypes::position_type>{ number_neurons };
    const auto correct_area_names = std::vector<std::string>{ number_neurons };
    const auto correct_signal_types = std::vector<SignalType>{ number_neurons };

    const auto faulty_ids = std::vector<NeuronID>{ number_neurons + 1 };
    const auto faulty_position = std::vector<RelearnTypes::position_type>{ number_neurons + 1 };
    const auto faulty_area_names = std::vector<std::string>{ number_neurons + 1 };
    const auto faulty_signal_types = std::vector<SignalType>{ number_neurons + 1 };

    std::filesystem::path path{ "./neurons.tmp" };

    ASSERT_THROW(NeuronIO::write_neurons_componentwise(correct_ids, correct_position, correct_area_names, faulty_signal_types, path), RelearnException);
    ASSERT_THROW(NeuronIO::write_neurons_componentwise(correct_ids, correct_position, faulty_area_names, correct_signal_types, path), RelearnException);
    ASSERT_THROW(NeuronIO::write_neurons_componentwise(correct_ids, faulty_position, correct_area_names, correct_signal_types, path), RelearnException);
    ASSERT_THROW(NeuronIO::write_neurons_componentwise(faulty_ids, correct_position, correct_area_names, correct_signal_types, path), RelearnException);

    ASSERT_THROW(NeuronIO::write_neurons_componentwise(correct_ids, correct_position, faulty_area_names, faulty_signal_types, path), RelearnException);
    ASSERT_THROW(NeuronIO::write_neurons_componentwise(correct_ids, faulty_position, correct_area_names, faulty_signal_types, path), RelearnException);
    ASSERT_THROW(NeuronIO::write_neurons_componentwise(faulty_ids, correct_position, correct_area_names, faulty_signal_types, path), RelearnException);
    ASSERT_THROW(NeuronIO::write_neurons_componentwise(correct_ids, faulty_position, faulty_area_names, correct_signal_types, path), RelearnException);
    ASSERT_THROW(NeuronIO::write_neurons_componentwise(faulty_ids, correct_position, faulty_area_names, correct_signal_types, path), RelearnException);
    ASSERT_THROW(NeuronIO::write_neurons_componentwise(faulty_ids, faulty_position, correct_area_names, correct_signal_types, path), RelearnException);

    ASSERT_THROW(NeuronIO::write_neurons_componentwise(faulty_ids, faulty_position, faulty_area_names, correct_signal_types, path), RelearnException);
    ASSERT_THROW(NeuronIO::write_neurons_componentwise(faulty_ids, faulty_position, correct_area_names, faulty_signal_types, path), RelearnException);
    ASSERT_THROW(NeuronIO::write_neurons_componentwise(faulty_ids, correct_position, faulty_area_names, faulty_signal_types, path), RelearnException);
    ASSERT_THROW(NeuronIO::write_neurons_componentwise(correct_ids, faulty_position, faulty_area_names, faulty_signal_types, path), RelearnException);
}

TEST_F(IOTest, testNeuronIOWriteComponentwiseFileNotFound) {
    const auto number_neurons = get_random_number_neurons();

    auto preliminary_ids = std::vector<NeuronID>{};
    auto preliminary_position = std::vector<RelearnTypes::position_type>{};
    auto preliminary_area_names = std::vector<std::string>{};
    auto preliminary_signal_types = std::vector<SignalType>{};

    for (auto i = 0; i < number_neurons; i++) {
        preliminary_ids.emplace_back(false, i);
        preliminary_position.emplace_back(get_random_position());
        preliminary_area_names.emplace_back("area_" + std::to_string(i));
        preliminary_signal_types.emplace_back(get_random_signal_type());
    }

    std::filesystem::path path{ "" };

    ASSERT_THROW(NeuronIO::write_neurons_componentwise(preliminary_ids, preliminary_position, preliminary_area_names, preliminary_signal_types, path), RelearnException);
}

TEST_F(IOTest, testNeuronIOWriteComponentwise) {
    const auto number_neurons = get_random_number_neurons();

    auto preliminary_ids = std::vector<NeuronID>{};
    auto preliminary_position = std::vector<RelearnTypes::position_type>{};
    auto preliminary_area_names = std::vector<std::string>{};
    auto preliminary_signal_types = std::vector<SignalType>{};

    for (auto i = 0; i < number_neurons; i++) {
        preliminary_ids.emplace_back(false, i);
        preliminary_position.emplace_back(get_random_position());
        preliminary_area_names.emplace_back("area_" + std::to_string(i));
        preliminary_signal_types.emplace_back(get_random_signal_type());
    }

    std::filesystem::path path{ "./neurons.tmp" };

    ASSERT_NO_THROW(NeuronIO::write_neurons_componentwise(preliminary_ids, preliminary_position, preliminary_area_names, preliminary_signal_types, path));
}

TEST_F(IOTest, testNeuronIOReadComponentwise) {
    const auto number_neurons = get_random_number_neurons();

    auto preliminary_ids = std::vector<NeuronID>{};
    auto preliminary_position = std::vector<RelearnTypes::position_type>{};
    auto preliminary_area_names = std::vector<std::string>{};
    auto preliminary_signal_types = std::vector<SignalType>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = get_maximum_position();

    for (auto i = 0; i < number_neurons; i++) {
        preliminary_ids.emplace_back(false, i);
        preliminary_position.emplace_back(get_random_position_in_box(min_pos, max_pos));
        preliminary_area_names.emplace_back("area_" + std::to_string(i));
        preliminary_signal_types.emplace_back(get_random_signal_type());
    }

    std::filesystem::path path{ "./neurons.tmp" };

    NeuronIO::write_neurons_componentwise(preliminary_ids, preliminary_position, preliminary_area_names, preliminary_signal_types, path);

    const auto& [read_ids, read_positions, read_area_names, read_signal_types, additional_infos]
        = NeuronIO::read_neurons_componentwise(path);

    ASSERT_EQ(preliminary_ids, read_ids);
    ASSERT_EQ(preliminary_area_names, read_area_names);
    ASSERT_EQ(preliminary_signal_types, read_signal_types);

    ASSERT_EQ(preliminary_position.size(), read_positions.size());

    for (auto i = 0; i < number_neurons; i++) {
        const auto& diff = preliminary_position[i] - read_positions[i];
        const auto norm = diff.calculate_2_norm();

        ASSERT_NEAR(0.0, norm, eps);
    }

    const auto& [read_min_position, read_max_position, read_excitatory_neurons, read_inhibitory_neurons] = additional_infos;

    const auto number_excitatory = std::count(preliminary_signal_types.begin(), preliminary_signal_types.end(), SignalType::Excitatory);
    const auto number_inhibitory = std::count(preliminary_signal_types.begin(), preliminary_signal_types.end(), SignalType::Inhibitory);

    ASSERT_EQ(number_excitatory, read_excitatory_neurons);
    ASSERT_EQ(number_inhibitory, read_inhibitory_neurons);

    RelearnTypes::position_type minimum(std::numeric_limits<RelearnTypes::position_type::value_type>::max());
    RelearnTypes::position_type maximum(std::numeric_limits<RelearnTypes::position_type::value_type>::min());

    for (const auto& position : read_positions) {
        minimum.calculate_componentwise_minimum(position);
        maximum.calculate_componentwise_maximum(position);
    }

    ASSERT_EQ(minimum, read_min_position);
    ASSERT_EQ(maximum, read_max_position);
}

TEST_F(IOTest, testNeuronIOReadComponentwiseFileNotFound) {
    std::filesystem::path path{ "" };
    ASSERT_THROW(const auto& val = NeuronIO::read_neurons_componentwise(path), RelearnException);
}

TEST_F(IOTest, testNeuronIOReadComponentwiseIDException) {
    const auto number_neurons = get_random_number_neurons() + 1;

    auto preliminary_ids = std::vector<NeuronID>{};
    auto preliminary_position = std::vector<RelearnTypes::position_type>{};
    auto preliminary_area_names = std::vector<std::string>{};
    auto preliminary_signal_types = std::vector<SignalType>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = get_maximum_position();

    for (auto i = 0; i < number_neurons; i++) {
        preliminary_ids.emplace_back(false, i);
        preliminary_position.emplace_back(get_random_position_in_box(min_pos, max_pos));
        preliminary_area_names.emplace_back("area_" + std::to_string(i));
        preliminary_signal_types.emplace_back(get_random_signal_type());
    }

    const auto idx1 = get_random_integer<size_t>(0, number_neurons - 1);
    auto idx2 = get_random_integer<size_t>(0, number_neurons - 2);

    if (idx1 <= idx2) {
        idx2++;
    }

    std::swap(preliminary_ids[idx1], preliminary_ids[idx2]);

    std::filesystem::path path{ "./neurons.tmp" };

    NeuronIO::write_neurons_componentwise(preliminary_ids, preliminary_position, preliminary_area_names, preliminary_signal_types, path);

    ASSERT_THROW(const auto& val = NeuronIO::read_neurons_componentwise(path);, RelearnException);
}

TEST_F(IOTest, testNeuronIOReadComponentwisePositionXException) {
    const auto number_neurons = get_random_number_neurons();

    auto preliminary_ids = std::vector<NeuronID>{};
    auto preliminary_position = std::vector<RelearnTypes::position_type>{};
    auto preliminary_area_names = std::vector<std::string>{};
    auto preliminary_signal_types = std::vector<SignalType>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = get_maximum_position();

    for (auto i = 0; i < number_neurons; i++) {
        preliminary_ids.emplace_back(false, i);
        preliminary_position.emplace_back(get_random_position_in_box(min_pos, max_pos));
        preliminary_area_names.emplace_back("area_" + std::to_string(i));
        preliminary_signal_types.emplace_back(get_random_signal_type());
    }

    const auto idx1 = get_random_integer<size_t>(0, number_neurons - 1);

    preliminary_position[idx1].set_x(-preliminary_position[idx1].get_x());

    std::filesystem::path path{ "./neurons.tmp" };
    NeuronIO::write_neurons_componentwise(preliminary_ids, preliminary_position, preliminary_area_names, preliminary_signal_types, path);
    ASSERT_THROW(const auto& val = NeuronIO::read_neurons_componentwise(path);, RelearnException);
}

TEST_F(IOTest, testNeuronIOReadComponentwisePositionYException) {
    const auto number_neurons = get_random_number_neurons();

    auto preliminary_ids = std::vector<NeuronID>{};
    auto preliminary_position = std::vector<RelearnTypes::position_type>{};
    auto preliminary_area_names = std::vector<std::string>{};
    auto preliminary_signal_types = std::vector<SignalType>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = get_maximum_position();

    for (auto i = 0; i < number_neurons; i++) {
        preliminary_ids.emplace_back(false, i);
        preliminary_position.emplace_back(get_random_position_in_box(min_pos, max_pos));
        preliminary_area_names.emplace_back("area_" + std::to_string(i));
        preliminary_signal_types.emplace_back(get_random_signal_type());
    }

    const auto idx1 = get_random_integer<size_t>(0, number_neurons - 1);

    preliminary_position[idx1].set_y(-preliminary_position[idx1].get_y());

    std::filesystem::path path{ "./neurons.tmp" };
    NeuronIO::write_neurons_componentwise(preliminary_ids, preliminary_position, preliminary_area_names, preliminary_signal_types, path);
    ASSERT_THROW(const auto& val = NeuronIO::read_neurons_componentwise(path);, RelearnException);
}

TEST_F(IOTest, testNeuronIOReadComponentwisePositionZException) {
    const auto number_neurons = get_random_number_neurons();

    auto preliminary_ids = std::vector<NeuronID>{};
    auto preliminary_position = std::vector<RelearnTypes::position_type>{};
    auto preliminary_area_names = std::vector<std::string>{};
    auto preliminary_signal_types = std::vector<SignalType>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = get_maximum_position();

    for (auto i = 0; i < number_neurons; i++) {
        preliminary_ids.emplace_back(false, i);
        preliminary_position.emplace_back(get_random_position_in_box(min_pos, max_pos));
        preliminary_area_names.emplace_back("area_" + std::to_string(i));
        preliminary_signal_types.emplace_back(get_random_signal_type());
    }

    const auto idx1 = get_random_integer<size_t>(0, number_neurons - 1);

    preliminary_position[idx1].set_z(-preliminary_position[idx1].get_z());

    std::filesystem::path path{ "./neurons.tmp" };
    NeuronIO::write_neurons_componentwise(preliminary_ids, preliminary_position, preliminary_area_names, preliminary_signal_types, path);
    ASSERT_THROW(const auto& val = NeuronIO::read_neurons_componentwise(path);, RelearnException);
}

TEST_F(IOTest, testNeuronIOWrite1) {
    const auto number_neurons = get_random_number_neurons();

    auto preliminary_ids = std::vector<NeuronID>{};
    auto preliminary_position = std::vector<RelearnTypes::position_type>{};
    auto preliminary_area_names = std::vector<std::string>{};
    auto preliminary_signal_types = std::vector<SignalType>{};

    auto preliminary_neurons = std::vector<LoadedNeuron>{};

    for (auto i = 0; i < number_neurons; i++) {
        preliminary_ids.emplace_back(false, i);
        preliminary_position.emplace_back(get_random_position());
        preliminary_area_names.emplace_back("area_" + std::to_string(i));
        preliminary_signal_types.emplace_back(get_random_signal_type());

        preliminary_neurons.emplace_back(preliminary_position[i], preliminary_ids[i], preliminary_signal_types[i], preliminary_area_names[i]);
    }

    std::filesystem::path path{ "./neurons.tmp" };

    ASSERT_NO_THROW(NeuronIO::write_neurons(preliminary_neurons, path));
}

TEST_F(IOTest, testNeuronIOWrite2) {
    const auto number_neurons = get_random_number_neurons();

    auto preliminary_ids = std::vector<NeuronID>{};
    auto preliminary_position = std::vector<RelearnTypes::position_type>{};
    auto preliminary_area_names = std::vector<std::string>{};
    auto preliminary_signal_types = std::vector<SignalType>{};

    auto preliminary_neurons = std::vector<LoadedNeuron>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = get_maximum_position();

    for (auto i = 0; i < number_neurons; i++) {
        preliminary_ids.emplace_back(false, i);
        preliminary_position.emplace_back(get_random_position_in_box(min_pos, max_pos));
        preliminary_area_names.emplace_back("area_" + std::to_string(i));
        preliminary_signal_types.emplace_back(get_random_signal_type());

        preliminary_neurons.emplace_back(preliminary_position[i], preliminary_ids[i], preliminary_signal_types[i], preliminary_area_names[i]);
    }

    std::filesystem::path path{ "./neurons.tmp" };

    ASSERT_NO_THROW(NeuronIO::write_neurons(preliminary_neurons, path));

    const auto& [read_ids, read_positions, read_area_names, read_signal_types, additional_infos]
        = NeuronIO::read_neurons_componentwise(path);

    ASSERT_EQ(preliminary_ids, read_ids);
    ASSERT_EQ(preliminary_area_names, read_area_names);
    ASSERT_EQ(preliminary_signal_types, read_signal_types);

    ASSERT_EQ(preliminary_position.size(), read_positions.size());

    for (auto i = 0; i < number_neurons; i++) {
        const auto& diff = preliminary_position[i] - read_positions[i];
        const auto norm = diff.calculate_2_norm();

        ASSERT_NEAR(0.0, norm, eps);
    }

    const auto& [read_min_position, read_max_position, read_excitatory_neurons, read_inhibitory_neurons] = additional_infos;

    const auto number_excitatory = std::count(preliminary_signal_types.begin(), preliminary_signal_types.end(), SignalType::Excitatory);
    const auto number_inhibitory = std::count(preliminary_signal_types.begin(), preliminary_signal_types.end(), SignalType::Inhibitory);

    ASSERT_EQ(number_excitatory, read_excitatory_neurons);
    ASSERT_EQ(number_inhibitory, read_inhibitory_neurons);

    RelearnTypes::position_type minimum(std::numeric_limits<RelearnTypes::position_type::value_type>::max());
    RelearnTypes::position_type maximum(std::numeric_limits<RelearnTypes::position_type::value_type>::min());

    for (const auto& position : read_positions) {
        minimum.calculate_componentwise_minimum(position);
        maximum.calculate_componentwise_maximum(position);
    }

    ASSERT_EQ(minimum, read_min_position);
    ASSERT_EQ(maximum, read_max_position);
}

TEST_F(IOTest, testNeuronIOWriteFileNotFound) {
    const auto number_neurons = get_random_number_neurons();

    auto preliminary_ids = std::vector<NeuronID>{};
    auto preliminary_position = std::vector<RelearnTypes::position_type>{};
    auto preliminary_area_names = std::vector<std::string>{};
    auto preliminary_signal_types = std::vector<SignalType>{};

    auto preliminary_neurons = std::vector<LoadedNeuron>{};

    for (auto i = 0; i < number_neurons; i++) {
        preliminary_ids.emplace_back(false, i);
        preliminary_position.emplace_back(get_random_position());
        preliminary_area_names.emplace_back("area_" + std::to_string(i));
        preliminary_signal_types.emplace_back(get_random_signal_type());

        preliminary_neurons.emplace_back(preliminary_position[i], preliminary_ids[i], preliminary_signal_types[i], preliminary_area_names[i]);
    }

    std::filesystem::path path{ "" };

    ASSERT_THROW(NeuronIO::write_neurons(preliminary_neurons, path), RelearnException);
}

TEST_F(IOTest, testNeuronIORead) {
    const auto number_neurons = get_random_number_neurons();

    auto preliminary_neurons = std::vector<LoadedNeuron>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = get_maximum_position();

    for (auto i = 0; i < number_neurons; i++) {
        preliminary_neurons.emplace_back(get_random_position_in_box(min_pos, max_pos), NeuronID{ false, i }, get_random_signal_type(), "area_" + std::to_string(i));
    }

    std::filesystem::path path{ "./neurons.tmp" };

    ASSERT_NO_THROW(NeuronIO::write_neurons(preliminary_neurons, path));

    const auto& [read_neurons, additional_infos]
        = NeuronIO::read_neurons(path);

    ASSERT_EQ(read_neurons.size(), preliminary_neurons.size());

    RelearnTypes::position_type minimum(std::numeric_limits<RelearnTypes::position_type::value_type>::max());
    RelearnTypes::position_type maximum(std::numeric_limits<RelearnTypes::position_type::value_type>::min());

    auto number_excitatory = 0;
    auto number_inhibitory = 0;

    for (auto i = 0; i < number_neurons; i++) {
        const auto& preliminary_neuron = preliminary_neurons[i];
        const auto& read_neuron = read_neurons[i];

        const auto& diff = preliminary_neuron.pos - read_neuron.pos;
        const auto norm = diff.calculate_2_norm();

        ASSERT_NEAR(0.0, norm, eps);

        ASSERT_EQ(read_neuron.id, preliminary_neuron.id);
        ASSERT_EQ(read_neuron.signal_type, preliminary_neuron.signal_type);
        ASSERT_EQ(read_neuron.area_name, preliminary_neuron.area_name);

        minimum.calculate_componentwise_minimum(read_neuron.pos);
        maximum.calculate_componentwise_maximum(read_neuron.pos);

        if (read_neuron.signal_type == SignalType::Excitatory) {
            number_excitatory++;
        } else {
            number_inhibitory++;
        }
    }

    const auto& [read_min_position, read_max_position, read_excitatory_neurons, read_inhibitory_neurons] = additional_infos;

    ASSERT_EQ(number_excitatory, read_excitatory_neurons);
    ASSERT_EQ(number_inhibitory, read_inhibitory_neurons);

    ASSERT_EQ(minimum, read_min_position);
    ASSERT_EQ(maximum, read_max_position);
}

TEST_F(IOTest, testNeuronIOReadFileNotFound) {
    std::filesystem::path path{ "" };
    ASSERT_THROW(const auto& val = NeuronIO::read_neurons_componentwise(path), RelearnException);
}

TEST_F(IOTest, testNeuronIOReadIDException) {
    const auto number_neurons = get_random_number_neurons() + 3;

    auto preliminary_ids = std::vector<NeuronID>{};
    auto preliminary_position = std::vector<RelearnTypes::position_type>{};
    auto preliminary_area_names = std::vector<std::string>{};
    auto preliminary_signal_types = std::vector<SignalType>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = get_maximum_position();

    for (auto i = 0; i < number_neurons; i++) {
        preliminary_ids.emplace_back(false, i);
        preliminary_position.emplace_back(get_random_position_in_box(min_pos, max_pos));
        preliminary_area_names.emplace_back("area_" + std::to_string(i));
        preliminary_signal_types.emplace_back(get_random_signal_type());
    }

    const auto idx1 = get_random_integer<size_t>(0, number_neurons - 1);
    auto idx2 = get_random_integer<size_t>(0, number_neurons - 2);

    if (idx1 <= idx2) {
        idx2++;
    }

    std::swap(preliminary_ids[idx1], preliminary_ids[idx2]);

    std::filesystem::path path{ "./neurons.tmp" };

    NeuronIO::write_neurons_componentwise(preliminary_ids, preliminary_position, preliminary_area_names, preliminary_signal_types, path);

    ASSERT_THROW(const auto& val = NeuronIO::read_neurons(path);, RelearnException);
}

TEST_F(IOTest, testNeuronIOReadPositionXException) {
    const auto number_neurons = get_random_number_neurons();

    auto preliminary_ids = std::vector<NeuronID>{};
    auto preliminary_position = std::vector<RelearnTypes::position_type>{};
    auto preliminary_area_names = std::vector<std::string>{};
    auto preliminary_signal_types = std::vector<SignalType>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = get_maximum_position();

    for (auto i = 0; i < number_neurons; i++) {
        preliminary_ids.emplace_back(false, i);
        preliminary_position.emplace_back(get_random_position_in_box(min_pos, max_pos));
        preliminary_area_names.emplace_back("area_" + std::to_string(i));
        preliminary_signal_types.emplace_back(get_random_signal_type());
    }

    const auto idx1 = get_random_integer<size_t>(0, number_neurons - 1);

    preliminary_position[idx1].set_x(-preliminary_position[idx1].get_x());

    std::filesystem::path path{ "./neurons.tmp" };
    NeuronIO::write_neurons_componentwise(preliminary_ids, preliminary_position, preliminary_area_names, preliminary_signal_types, path);
    ASSERT_THROW(const auto& val = NeuronIO::read_neurons(path);, RelearnException);
}

TEST_F(IOTest, testNeuronIOReadPositionYException) {
    const auto number_neurons = get_random_number_neurons();

    auto preliminary_ids = std::vector<NeuronID>{};
    auto preliminary_position = std::vector<RelearnTypes::position_type>{};
    auto preliminary_area_names = std::vector<std::string>{};
    auto preliminary_signal_types = std::vector<SignalType>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = get_maximum_position();

    for (auto i = 0; i < number_neurons; i++) {
        preliminary_ids.emplace_back(false, i);
        preliminary_position.emplace_back(get_random_position_in_box(min_pos, max_pos));
        preliminary_area_names.emplace_back("area_" + std::to_string(i));
        preliminary_signal_types.emplace_back(get_random_signal_type());
    }

    const auto idx1 = get_random_integer<size_t>(0, number_neurons - 1);

    preliminary_position[idx1].set_y(-preliminary_position[idx1].get_y());

    std::filesystem::path path{ "./neurons.tmp" };
    NeuronIO::write_neurons_componentwise(preliminary_ids, preliminary_position, preliminary_area_names, preliminary_signal_types, path);
    ASSERT_THROW(const auto& val = NeuronIO::read_neurons(path);, RelearnException);
}

TEST_F(IOTest, testNeuronIOReadPositionZException) {
    const auto number_neurons = get_random_number_neurons();

    auto preliminary_ids = std::vector<NeuronID>{};
    auto preliminary_position = std::vector<RelearnTypes::position_type>{};
    auto preliminary_area_names = std::vector<std::string>{};
    auto preliminary_signal_types = std::vector<SignalType>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = get_maximum_position();

    for (auto i = 0; i < number_neurons; i++) {
        preliminary_ids.emplace_back(false, i);
        preliminary_position.emplace_back(get_random_position_in_box(min_pos, max_pos));
        preliminary_area_names.emplace_back("area_" + std::to_string(i));
        preliminary_signal_types.emplace_back(get_random_signal_type());
    }

    const auto idx1 = get_random_integer<size_t>(0, number_neurons - 1);

    preliminary_position[idx1].set_z(-preliminary_position[idx1].get_z());

    std::filesystem::path path{ "./neurons.tmp" };
    NeuronIO::write_neurons_componentwise(preliminary_ids, preliminary_position, preliminary_area_names, preliminary_signal_types, path);
    ASSERT_THROW(const auto& val = NeuronIO::read_neurons(path);, RelearnException);
}

TEST_F(IOTest, testNeuronIOReadIDs) {
    const auto number_neurons = get_random_number_neurons();

    auto preliminary_ids = std::vector<NeuronID>{};
    auto preliminary_position = std::vector<RelearnTypes::position_type>{};
    auto preliminary_area_names = std::vector<std::string>{};
    auto preliminary_signal_types = std::vector<SignalType>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = get_maximum_position();

    for (auto i = 0; i < number_neurons; i++) {
        preliminary_ids.emplace_back(false, i);
        preliminary_position.emplace_back(get_random_position_in_box(min_pos, max_pos));
        preliminary_area_names.emplace_back("area_" + std::to_string(i));
        preliminary_signal_types.emplace_back(get_random_signal_type());
    }

    std::filesystem::path path{ "./neurons.tmp" };

    NeuronIO::write_neurons_componentwise(preliminary_ids, preliminary_position, preliminary_area_names, preliminary_signal_types, path);

    const auto& read_ids = NeuronIO::read_neuron_ids(path);

    ASSERT_EQ(read_ids, preliminary_ids);
}

TEST_F(IOTest, testNeuronIOReadIDsEmpty1) {
    const auto number_neurons = get_random_integer(2, upper_bound_num_neurons);

    auto preliminary_ids = std::vector<NeuronID>{};
    auto preliminary_position = std::vector<RelearnTypes::position_type>{};
    auto preliminary_area_names = std::vector<std::string>{};
    auto preliminary_signal_types = std::vector<SignalType>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = get_maximum_position();

    for (auto i = 0; i < number_neurons; i++) {
        preliminary_ids.emplace_back(false, i);
        preliminary_position.emplace_back(get_random_position_in_box(min_pos, max_pos));
        preliminary_area_names.emplace_back("area_" + std::to_string(i));
        preliminary_signal_types.emplace_back(get_random_signal_type());
    }

    const auto idx1 = get_random_integer<size_t>(0, number_neurons - 1);
    auto idx2 = get_random_integer<size_t>(0, number_neurons - 2);

    if (idx1 <= idx2) {
        idx2++;
    }

    std::swap(preliminary_ids[idx1], preliminary_ids[idx2]);

    std::filesystem::path path{ "./neurons.tmp" };

    NeuronIO::write_neurons_componentwise(preliminary_ids, preliminary_position, preliminary_area_names, preliminary_signal_types, path);

    const auto& ids = NeuronIO::read_neuron_ids(path);
    ASSERT_FALSE(ids.has_value());
}

TEST_F(IOTest, testNeuronIOReadIDsEmpty2) {
    const auto number_neurons = get_random_number_neurons();

    auto preliminary_ids = std::vector<NeuronID>{};
    auto preliminary_position = std::vector<RelearnTypes::position_type>{};
    auto preliminary_area_names = std::vector<std::string>{};
    auto preliminary_signal_types = std::vector<SignalType>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = get_maximum_position();

    for (auto i = 0; i < number_neurons; i++) {
        preliminary_ids.emplace_back(false, i);
        preliminary_position.emplace_back(get_random_position_in_box(min_pos, max_pos));
        preliminary_area_names.emplace_back("area_" + std::to_string(i));
        preliminary_signal_types.emplace_back(get_random_signal_type());
    }

    preliminary_ids.emplace_back(false, number_neurons + 1);
    preliminary_position.emplace_back(get_random_position_in_box(min_pos, max_pos));
    preliminary_area_names.emplace_back("area_" + std::to_string(number_neurons + 1));
    preliminary_signal_types.emplace_back(get_random_signal_type());

    std::filesystem::path path{ "./neurons.tmp" };

    NeuronIO::write_neurons_componentwise(preliminary_ids, preliminary_position, preliminary_area_names, preliminary_signal_types, path);

    const auto& ids = NeuronIO::read_neuron_ids(path);
    ASSERT_FALSE(ids.has_value());
}

TEST_F(IOTest, testNeuronIOReadIDsFileNotFound) {
    std::filesystem::path path{ "" };
    const auto& ids = NeuronIO::read_neuron_ids(path);
    ASSERT_FALSE(ids.has_value());
}

TEST_F(IOTest, testNeuronIOReadCommentsFileNotFound) {
    std::filesystem::path path{ "" };
    ASSERT_THROW(const auto& comments = NeuronIO::read_comments(path);, RelearnException);
}

TEST_F(IOTest, testNeuronIOReadComments1) {
    std::filesystem::path path{ "./comments.tmp" };

    {
        std::ofstream out_file{ path };
        out_file << "# 1\n# 2\n# #\n#";
    }

    const auto& comments = NeuronIO::read_comments(path);

    ASSERT_EQ(comments.size(), 4);
    ASSERT_EQ(comments[0], std::string("# 1"));
    ASSERT_EQ(comments[1], std::string("# 2"));
    ASSERT_EQ(comments[2], std::string("# #"));
    ASSERT_EQ(comments[3], std::string("#"));
}

TEST_F(IOTest, testNeuronIOReadComments2) {
    std::filesystem::path path{ "./comments.tmp" };

    {
        std::ofstream out_file{ path };
        out_file << "Hallo\n# 1\n# 2\n# #\n#";
    }

    const auto& comments = NeuronIO::read_comments(path);

    ASSERT_TRUE(comments.empty());
}

TEST_F(IOTest, testNeuronIOReadComments3) {
    std::filesystem::path path{ "./comments.tmp" };

    {
        std::ofstream out_file{ path };
        for (auto i = 0; i < 10; i++) {
            out_file << "# 1\n";
            out_file << "Hallo\n";
        }
    }

    const auto& comments = NeuronIO::read_comments(path);

    ASSERT_EQ(comments.size(), 1);
    ASSERT_EQ(comments[0], std::string("# 1"));
}

TEST_F(IOTest, testReadInSynapsesFileNotFound) {
    std::filesystem::path path{ "" };
    ASSERT_THROW(const auto& val = NeuronIO::read_in_synapses(path, 1, 1, 2), RelearnException);
}

TEST_F(IOTest, testReadInSynapses) {
    const auto number_ranks = get_random_number_ranks();
    const auto my_rank = get_random_rank(number_ranks);

    const auto number_synapses = get_random_number_synapses();
    const auto number_neurons = get_random_number_neurons();

    LocalSynapses preliminary_local_synapses{};
    DistantInSynapses preliminary_distant_synapses{};

    std::filesystem::path path{ "./in_network.tmp" };
    std::ofstream ofstream(path);

    for (auto synapse_id = 0; synapse_id < number_synapses; synapse_id++) {
        const auto& source_id = get_random_neuron_id(number_neurons);
        const auto& target_id = get_random_neuron_id(number_neurons);

        const auto source_rank = get_random_rank(number_ranks);

        const auto weight = get_random_synapse_weight();

        if (source_rank == my_rank) {
            preliminary_local_synapses.emplace_back(target_id, source_id, weight);
        } else {
            preliminary_distant_synapses.emplace_back(target_id, RankNeuronId(source_rank, source_id), weight);
        }

        ofstream << my_rank << ' ' << (target_id.get_neuron_id() + 1) << '\t' << source_rank << ' ' << (source_id.get_neuron_id() + 1) << ' ' << weight << '\n';
    }

    ofstream.flush();
    ofstream.close();

    auto [read_local_synapses, read_distant_synapses] = NeuronIO::read_in_synapses(path, number_neurons, my_rank, number_ranks);

    std::ranges::sort(preliminary_local_synapses);
    std::ranges::sort(preliminary_distant_synapses);

    std::ranges::sort(read_local_synapses);
    std::ranges::sort(read_distant_synapses);

    ASSERT_EQ(preliminary_local_synapses, read_local_synapses);
    ASSERT_EQ(preliminary_distant_synapses, read_distant_synapses);
}

TEST_F(IOTest, testReadOutSynapsesFileNotFound) {
    std::filesystem::path path{ "" };
    ASSERT_THROW(const auto& val = NeuronIO::read_out_synapses(path, 1, 1, 2), RelearnException);
}

TEST_F(IOTest, testReadOutSynapses) {
    const auto number_ranks = get_random_number_ranks();
    const auto my_rank = get_random_rank(number_ranks);

    const auto number_synapses = get_random_number_synapses();
    const auto number_neurons = get_random_number_neurons();

    LocalSynapses preliminary_local_synapses{};
    DistantOutSynapses preliminary_distant_synapses{};

    std::filesystem::path path{ "./out_network.tmp" };
    std::ofstream ofstream(path);

    for (auto synapse_id = 0; synapse_id < number_synapses; synapse_id++) {
        const auto& source_id = get_random_neuron_id(number_neurons);
        const auto& target_id = get_random_neuron_id(number_neurons);

        const auto target_rank = get_random_rank(number_ranks);

        const auto weight = get_random_synapse_weight();

        if (target_rank == my_rank) {
            preliminary_local_synapses.emplace_back(target_id, source_id, weight);
        } else {
            preliminary_distant_synapses.emplace_back(RankNeuronId(target_rank, target_id), source_id, weight);
        }

        ofstream << target_rank << ' ' << (target_id.get_neuron_id() + 1) << '\t' << my_rank << ' ' << (source_id.get_neuron_id() + 1) << ' ' << weight << '\n';
    }

    ofstream.flush();
    ofstream.close();

    auto [read_local_synapses, read_distant_synapses] = NeuronIO::read_out_synapses(path, number_neurons, my_rank, number_ranks);

    std::ranges::sort(preliminary_local_synapses);
    std::ranges::sort(preliminary_distant_synapses);

    std::ranges::sort(read_local_synapses);
    std::ranges::sort(read_distant_synapses);

    ASSERT_EQ(preliminary_local_synapses, read_local_synapses);
    ASSERT_EQ(preliminary_distant_synapses, read_distant_synapses);
}

TEST_F(IOTest, testWriteInSynapsesFileNotFound) {
    std::filesystem::path path{ "" };
    ASSERT_THROW(NeuronIO::write_in_synapses({}, {}, 0, path);, RelearnException);
}

TEST_F(IOTest, testWriteInSynapses) {
    const auto number_ranks = get_random_number_ranks();
    const auto my_rank = get_random_rank(number_ranks);

    const auto number_synapses = get_random_number_synapses();
    const auto number_neurons = get_random_number_neurons();

    LocalSynapses preliminary_local_synapses{};
    DistantInSynapses preliminary_distant_synapses{};

    std::filesystem::path path{ "./in_network.tmp" };

    for (auto synapse_id = 0; synapse_id < number_synapses; synapse_id++) {
        const auto& source_id = get_random_neuron_id(number_neurons);
        const auto& target_id = get_random_neuron_id(number_neurons);

        const auto source_rank = get_random_rank(number_ranks);

        const auto weight = get_random_synapse_weight();

        if (source_rank == my_rank) {
            preliminary_local_synapses.emplace_back(target_id, source_id, weight);
        } else {
            preliminary_distant_synapses.emplace_back(target_id, RankNeuronId(source_rank, source_id), weight);
        }
    }

    NeuronIO::write_in_synapses(preliminary_local_synapses, preliminary_distant_synapses, my_rank, path);

    auto [read_local_synapses, read_distant_synapses] = NeuronIO::read_in_synapses(path, number_neurons, my_rank, number_ranks);

    std::ranges::sort(preliminary_local_synapses);
    std::ranges::sort(preliminary_distant_synapses);

    std::ranges::sort(read_local_synapses);
    std::ranges::sort(read_distant_synapses);

    ASSERT_EQ(preliminary_local_synapses, read_local_synapses);
    ASSERT_EQ(preliminary_distant_synapses, read_distant_synapses);
}

TEST_F(IOTest, testWriteOutSynapsesFileNotFound) {
    std::filesystem::path path{ "" };
    ASSERT_THROW(NeuronIO::write_out_synapses({}, {}, 0, path);, RelearnException);
}

TEST_F(IOTest, testWriteOutSynapses) {
    const auto number_ranks = get_random_number_ranks();
    const auto my_rank = get_random_rank(number_ranks);

    const auto number_synapses = get_random_number_synapses();
    const auto number_neurons = get_random_number_neurons();

    LocalSynapses preliminary_local_synapses{};
    DistantOutSynapses preliminary_distant_synapses{};

    std::filesystem::path path{ "./out_network.tmp" };

    for (auto synapse_id = 0; synapse_id < number_synapses; synapse_id++) {
        const auto& source_id = get_random_neuron_id(number_neurons);
        const auto& target_id = get_random_neuron_id(number_neurons);

        const auto target_rank = get_random_rank(number_ranks);

        const auto weight = get_random_synapse_weight();

        if (target_rank == my_rank) {
            preliminary_local_synapses.emplace_back(target_id, source_id, weight);
        } else {
            preliminary_distant_synapses.emplace_back(RankNeuronId(target_rank, target_id), source_id, weight);
        }
    }

    NeuronIO::write_out_synapses(preliminary_local_synapses, preliminary_distant_synapses, my_rank, path);

    auto [read_local_synapses, read_distant_synapses] = NeuronIO::read_out_synapses(path, number_neurons, my_rank, number_ranks);

    std::ranges::sort(preliminary_local_synapses);
    std::ranges::sort(preliminary_distant_synapses);

    std::ranges::sort(read_local_synapses);
    std::ranges::sort(read_distant_synapses);

    ASSERT_EQ(preliminary_local_synapses, read_local_synapses);
    ASSERT_EQ(preliminary_distant_synapses, read_distant_synapses);
}

TEST_F(IOTest, testReadSynapsesInteractionNetworkGraph) {
    const auto number_ranks = get_random_number_ranks();
    const auto my_rank = get_random_rank(number_ranks);

    const auto number_synapses = get_random_number_synapses() * 0 + 5;
    const auto number_neurons = get_random_number_neurons();

    std::vector<std::map<NeuronID, RelearnTypes::synapse_weight>> local_synapses(number_neurons, std::map<NeuronID, RelearnTypes::synapse_weight>{});
    std::vector<std::map<RankNeuronId, RelearnTypes::synapse_weight>> distant_in_synapses(number_neurons, std::map<RankNeuronId, RelearnTypes::synapse_weight>{});
    std::vector<std::map<RankNeuronId, RelearnTypes::synapse_weight>> distant_out_synapses(number_neurons, std::map<RankNeuronId, RelearnTypes::synapse_weight>{});

    for (auto synapse_id = 0; synapse_id < number_synapses; synapse_id++) {
        const auto& source_id = get_random_neuron_id(number_neurons);
        const auto& target_id = get_random_neuron_id(number_neurons);

        const auto target_rank = get_random_rank(number_ranks);

        const auto weight = get_random_synapse_weight();

        if (target_rank == my_rank) {
            local_synapses[target_id.get_neuron_id()][source_id] += weight;
        } else {
            distant_in_synapses[source_id.get_neuron_id()][RankNeuronId(target_rank, target_id)] += weight;
        }
    }

    for (auto synapse_id = 0; synapse_id < number_synapses; synapse_id++) {
        const auto& source_id = get_random_neuron_id(number_neurons);
        const auto& target_id = get_random_neuron_id(number_neurons);

        const auto source_rank = get_random_rank(number_ranks);

        const auto weight = get_random_synapse_weight();

        if (source_rank == my_rank) {
            local_synapses[target_id.get_neuron_id()][source_id] += weight;
        } else {
            distant_out_synapses[target_id.get_neuron_id()][RankNeuronId(source_rank, source_id)] += weight;
        }
    }

    LocalSynapses golden_local_synapses{};
    golden_local_synapses.reserve(number_synapses * 2);

    DistantInSynapses golden_distant_in_synapses{};
    golden_distant_in_synapses.reserve(number_synapses * 2);

    DistantOutSynapses golden_distant_out_synapses{};
    golden_distant_out_synapses.reserve(number_synapses * 2);

    for (size_t neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
        for (const auto& [source_id, weight] : local_synapses[neuron_id]) {
            golden_local_synapses.emplace_back(NeuronID(neuron_id), source_id, weight);
        }

        for (const auto& [source_rni, weight] : distant_in_synapses[neuron_id]) {
            golden_distant_in_synapses.emplace_back(NeuronID(neuron_id), source_rni, weight);
        }

        for (const auto& [target_rni, weight] : distant_out_synapses[neuron_id]) {
            golden_distant_out_synapses.emplace_back(target_rni, NeuronID(neuron_id), weight);
        }
    }

    std::ranges::sort(golden_local_synapses);
    std::ranges::sort(golden_distant_in_synapses);
    std::ranges::sort(golden_distant_out_synapses);

    NetworkGraph ng(number_neurons, my_rank);
    ng.add_edges(golden_local_synapses, golden_distant_in_synapses, golden_distant_out_synapses);

    std::filesystem::path in_path{ "./in_network.tmp" };
    std::filesystem::path out_path{ "./out_network.tmp" };

    std::ofstream in_ofstream{ in_path };
    std::ofstream out_ofstream{ out_path };

    ng.print_with_ranks(out_ofstream, in_ofstream);

    in_ofstream.flush();
    in_ofstream.close();

    out_ofstream.flush();
    out_ofstream.close();

    auto [read_local_in_synapses, read_distant_in_synapses] = NeuronIO::read_in_synapses(in_path, number_neurons, my_rank, number_ranks);
    auto [read_local_out_synapses, read_distant_out_synapses] = NeuronIO::read_out_synapses(out_path, number_neurons, my_rank, number_ranks);
       
    std::ranges::sort(read_local_in_synapses);
    std::ranges::sort(read_distant_in_synapses);
    std::ranges::sort(read_local_out_synapses);
    std::ranges::sort(read_distant_out_synapses);

    ASSERT_EQ(golden_local_synapses, read_local_in_synapses);
    ASSERT_EQ(golden_distant_in_synapses, read_distant_in_synapses);
    ASSERT_EQ(golden_local_synapses, read_local_out_synapses);
    ASSERT_EQ(golden_distant_out_synapses, read_distant_out_synapses);
}
