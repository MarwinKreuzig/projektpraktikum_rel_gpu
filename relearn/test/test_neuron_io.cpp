#include "../googletest/include/gtest/gtest.h"

#include "RelearnTest.hpp"

#include "../source/io/NeuronIO.h"

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

    auto golden_ids = std::vector<NeuronID>{};
    auto golden_position = std::vector<RelearnTypes::position_type>{};
    auto golden_area_names = std::vector<std::string>{};
    auto golden_signal_types = std::vector<SignalType>{};

    for (auto i = 0; i < number_neurons; i++) {
        golden_ids.emplace_back(false, i);
        golden_position.emplace_back(get_random_position());
        golden_area_names.emplace_back("area_" + std::to_string(i));
        golden_signal_types.emplace_back(get_random_signal_type());
    }

    std::filesystem::path path{ "" };

    ASSERT_THROW(NeuronIO::write_neurons_componentwise(golden_ids, golden_position, golden_area_names, golden_signal_types, path), RelearnException);
}

TEST_F(IOTest, testNeuronIOWriteComponentwise) {
    const auto number_neurons = get_random_number_neurons();

    auto golden_ids = std::vector<NeuronID>{};
    auto golden_position = std::vector<RelearnTypes::position_type>{};
    auto golden_area_names = std::vector<std::string>{};
    auto golden_signal_types = std::vector<SignalType>{};

    for (auto i = 0; i < number_neurons; i++) {
        golden_ids.emplace_back(false, i);
        golden_position.emplace_back(get_random_position());
        golden_area_names.emplace_back("area_" + std::to_string(i));
        golden_signal_types.emplace_back(get_random_signal_type());
    }

    std::filesystem::path path{ "./neurons.tmp" };

    ASSERT_NO_THROW(NeuronIO::write_neurons_componentwise(golden_ids, golden_position, golden_area_names, golden_signal_types, path));
}

TEST_F(IOTest, testNeuronIOReadComponentwise) {
    const auto number_neurons = get_random_number_neurons();

    auto golden_ids = std::vector<NeuronID>{};
    auto golden_position = std::vector<RelearnTypes::position_type>{};
    auto golden_area_names = std::vector<std::string>{};
    auto golden_signal_types = std::vector<SignalType>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = get_maximum_position();

    for (auto i = 0; i < number_neurons; i++) {
        golden_ids.emplace_back(false, i);
        golden_position.emplace_back(get_random_position_in_box(min_pos, max_pos));
        golden_area_names.emplace_back("area_" + std::to_string(i));
        golden_signal_types.emplace_back(get_random_signal_type());
    }

    std::filesystem::path path{ "./neurons.tmp" };

    NeuronIO::write_neurons_componentwise(golden_ids, golden_position, golden_area_names, golden_signal_types, path);

    const auto& [read_ids, read_positions, read_area_names, read_signal_types, additional_infos]
        = NeuronIO::read_neurons_componentwise(path);

    ASSERT_EQ(golden_ids, read_ids);
    ASSERT_EQ(golden_area_names, read_area_names);
    ASSERT_EQ(golden_signal_types, read_signal_types);

    ASSERT_EQ(golden_position.size(), read_positions.size());

    for (auto i = 0; i < number_neurons; i++) {
        const auto& diff = golden_position[i] - read_positions[i];
        const auto norm = diff.calculate_2_norm();

        ASSERT_NEAR(0.0, norm, eps);
    }

    const auto& [read_min_position, read_max_position, read_excitatory_neurons, read_inhibitory_neurons] = additional_infos;

    const auto number_excitatory = std::count(golden_signal_types.begin(), golden_signal_types.end(), SignalType::Excitatory);
    const auto number_inhibitory = std::count(golden_signal_types.begin(), golden_signal_types.end(), SignalType::Inhibitory);

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
    const auto number_neurons = get_random_number_neurons();

    auto golden_ids = std::vector<NeuronID>{};
    auto golden_position = std::vector<RelearnTypes::position_type>{};
    auto golden_area_names = std::vector<std::string>{};
    auto golden_signal_types = std::vector<SignalType>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = get_maximum_position();

    for (auto i = 0; i < number_neurons; i++) {
        golden_ids.emplace_back(false, i);
        golden_position.emplace_back(get_random_position_in_box(min_pos, max_pos));
        golden_area_names.emplace_back("area_" + std::to_string(i));
        golden_signal_types.emplace_back(get_random_signal_type());
    }

    const auto idx1 = get_random_integer<size_t>(0, number_neurons - 1);
    auto idx2 = get_random_integer<size_t>(0, number_neurons - 2);

    if (idx1 <= idx2) {
        idx2++;
    }

    std::swap(golden_ids[idx1], golden_ids[idx2]);

    std::filesystem::path path{ "./neurons.tmp" };

    NeuronIO::write_neurons_componentwise(golden_ids, golden_position, golden_area_names, golden_signal_types, path);

    ASSERT_THROW(const auto& val = NeuronIO::read_neurons_componentwise(path);, RelearnException);
}

TEST_F(IOTest, testNeuronIOReadComponentwisePositionXException) {
    const auto number_neurons = get_random_number_neurons();

    auto golden_ids = std::vector<NeuronID>{};
    auto golden_position = std::vector<RelearnTypes::position_type>{};
    auto golden_area_names = std::vector<std::string>{};
    auto golden_signal_types = std::vector<SignalType>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = get_maximum_position();

    for (auto i = 0; i < number_neurons; i++) {
        golden_ids.emplace_back(false, i);
        golden_position.emplace_back(get_random_position_in_box(min_pos, max_pos));
        golden_area_names.emplace_back("area_" + std::to_string(i));
        golden_signal_types.emplace_back(get_random_signal_type());
    }

    const auto idx1 = get_random_integer<size_t>(0, number_neurons - 1);

    golden_position[idx1].set_x(-golden_position[idx1].get_x());

    std::filesystem::path path{ "./neurons.tmp" };
    NeuronIO::write_neurons_componentwise(golden_ids, golden_position, golden_area_names, golden_signal_types, path);
    ASSERT_THROW(const auto& val = NeuronIO::read_neurons_componentwise(path);, RelearnException);
}

TEST_F(IOTest, testNeuronIOReadComponentwisePositionYException) {
    const auto number_neurons = get_random_number_neurons();

    auto golden_ids = std::vector<NeuronID>{};
    auto golden_position = std::vector<RelearnTypes::position_type>{};
    auto golden_area_names = std::vector<std::string>{};
    auto golden_signal_types = std::vector<SignalType>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = get_maximum_position();

    for (auto i = 0; i < number_neurons; i++) {
        golden_ids.emplace_back(false, i);
        golden_position.emplace_back(get_random_position_in_box(min_pos, max_pos));
        golden_area_names.emplace_back("area_" + std::to_string(i));
        golden_signal_types.emplace_back(get_random_signal_type());
    }

    const auto idx1 = get_random_integer<size_t>(0, number_neurons - 1);

    golden_position[idx1].set_y(-golden_position[idx1].get_y());

    std::filesystem::path path{ "./neurons.tmp" };
    NeuronIO::write_neurons_componentwise(golden_ids, golden_position, golden_area_names, golden_signal_types, path);
    ASSERT_THROW(const auto& val = NeuronIO::read_neurons_componentwise(path);, RelearnException);
}

TEST_F(IOTest, testNeuronIOReadComponentwisePositionZException) {
    const auto number_neurons = get_random_number_neurons();

    auto golden_ids = std::vector<NeuronID>{};
    auto golden_position = std::vector<RelearnTypes::position_type>{};
    auto golden_area_names = std::vector<std::string>{};
    auto golden_signal_types = std::vector<SignalType>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = get_maximum_position();

    for (auto i = 0; i < number_neurons; i++) {
        golden_ids.emplace_back(false, i);
        golden_position.emplace_back(get_random_position_in_box(min_pos, max_pos));
        golden_area_names.emplace_back("area_" + std::to_string(i));
        golden_signal_types.emplace_back(get_random_signal_type());
    }

    const auto idx1 = get_random_integer<size_t>(0, number_neurons - 1);

    golden_position[idx1].set_z(-golden_position[idx1].get_z());

    std::filesystem::path path{ "./neurons.tmp" };
    NeuronIO::write_neurons_componentwise(golden_ids, golden_position, golden_area_names, golden_signal_types, path);
    ASSERT_THROW(const auto& val = NeuronIO::read_neurons_componentwise(path);, RelearnException);
}

TEST_F(IOTest, testNeuronIOWrite1) {
    const auto number_neurons = get_random_number_neurons();

    auto golden_ids = std::vector<NeuronID>{};
    auto golden_position = std::vector<RelearnTypes::position_type>{};
    auto golden_area_names = std::vector<std::string>{};
    auto golden_signal_types = std::vector<SignalType>{};

    auto golden_neurons = std::vector<LoadedNeuron>{};

    for (auto i = 0; i < number_neurons; i++) {
        golden_ids.emplace_back(false, i);
        golden_position.emplace_back(get_random_position());
        golden_area_names.emplace_back("area_" + std::to_string(i));
        golden_signal_types.emplace_back(get_random_signal_type());

        golden_neurons.emplace_back(golden_position[i], golden_ids[i], golden_signal_types[i], golden_area_names[i]);
    }

    std::filesystem::path path{ "./neurons.tmp" };

    ASSERT_NO_THROW(NeuronIO::write_neurons(golden_neurons, path));
}

TEST_F(IOTest, testNeuronIOWrite2) {
    const auto number_neurons = get_random_number_neurons();

    auto golden_ids = std::vector<NeuronID>{};
    auto golden_position = std::vector<RelearnTypes::position_type>{};
    auto golden_area_names = std::vector<std::string>{};
    auto golden_signal_types = std::vector<SignalType>{};

    auto golden_neurons = std::vector<LoadedNeuron>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = get_maximum_position();

    for (auto i = 0; i < number_neurons; i++) {
        golden_ids.emplace_back(false, i);
        golden_position.emplace_back(get_random_position_in_box(min_pos, max_pos));
        golden_area_names.emplace_back("area_" + std::to_string(i));
        golden_signal_types.emplace_back(get_random_signal_type());

        golden_neurons.emplace_back(golden_position[i], golden_ids[i], golden_signal_types[i], golden_area_names[i]);
    }

    std::filesystem::path path{ "./neurons.tmp" };

    ASSERT_NO_THROW(NeuronIO::write_neurons(golden_neurons, path));

    const auto& [read_ids, read_positions, read_area_names, read_signal_types, additional_infos]
        = NeuronIO::read_neurons_componentwise(path);

    ASSERT_EQ(golden_ids, read_ids);
    ASSERT_EQ(golden_area_names, read_area_names);
    ASSERT_EQ(golden_signal_types, read_signal_types);

    ASSERT_EQ(golden_position.size(), read_positions.size());

    for (auto i = 0; i < number_neurons; i++) {
        const auto& diff = golden_position[i] - read_positions[i];
        const auto norm = diff.calculate_2_norm();

        ASSERT_NEAR(0.0, norm, eps);
    }

    const auto& [read_min_position, read_max_position, read_excitatory_neurons, read_inhibitory_neurons] = additional_infos;

    const auto number_excitatory = std::count(golden_signal_types.begin(), golden_signal_types.end(), SignalType::Excitatory);
    const auto number_inhibitory = std::count(golden_signal_types.begin(), golden_signal_types.end(), SignalType::Inhibitory);

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

    auto golden_ids = std::vector<NeuronID>{};
    auto golden_position = std::vector<RelearnTypes::position_type>{};
    auto golden_area_names = std::vector<std::string>{};
    auto golden_signal_types = std::vector<SignalType>{};

    auto golden_neurons = std::vector<LoadedNeuron>{};

    for (auto i = 0; i < number_neurons; i++) {
        golden_ids.emplace_back(false, i);
        golden_position.emplace_back(get_random_position());
        golden_area_names.emplace_back("area_" + std::to_string(i));
        golden_signal_types.emplace_back(get_random_signal_type());

        golden_neurons.emplace_back(golden_position[i], golden_ids[i], golden_signal_types[i], golden_area_names[i]);
    }

    std::filesystem::path path{ "" };

    ASSERT_THROW(NeuronIO::write_neurons(golden_neurons, path), RelearnException);
}

TEST_F(IOTest, testNeuronIORead) {
    const auto number_neurons = get_random_number_neurons();

    auto golden_neurons = std::vector<LoadedNeuron>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = get_maximum_position();

    for (auto i = 0; i < number_neurons; i++) {
        golden_neurons.emplace_back(get_random_position_in_box(min_pos, max_pos), NeuronID{ false, i }, get_random_signal_type(), "area_" + std::to_string(i));
    }

    std::filesystem::path path{ "./neurons.tmp" };

    ASSERT_NO_THROW(NeuronIO::write_neurons(golden_neurons, path));

    const auto& [read_neurons, additional_infos]
        = NeuronIO::read_neurons(path);

    ASSERT_EQ(read_neurons.size(), golden_neurons.size());

    RelearnTypes::position_type minimum(std::numeric_limits<RelearnTypes::position_type::value_type>::max());
    RelearnTypes::position_type maximum(std::numeric_limits<RelearnTypes::position_type::value_type>::min());

    auto number_excitatory = 0;
    auto number_inhibitory = 0;

    for (auto i = 0; i < number_neurons; i++) {
        const auto& golden_neuron = golden_neurons[i];
        const auto& read_neuron = read_neurons[i];

        const auto& diff = golden_neuron.pos - read_neuron.pos;
        const auto norm = diff.calculate_2_norm();

        ASSERT_NEAR(0.0, norm, eps);

        ASSERT_EQ(read_neuron.id, golden_neuron.id);
        ASSERT_EQ(read_neuron.signal_type, golden_neuron.signal_type);
        ASSERT_EQ(read_neuron.area_name, golden_neuron.area_name);

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
    const auto number_neurons = get_random_number_neurons();

    auto golden_ids = std::vector<NeuronID>{};
    auto golden_position = std::vector<RelearnTypes::position_type>{};
    auto golden_area_names = std::vector<std::string>{};
    auto golden_signal_types = std::vector<SignalType>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = get_maximum_position();

    for (auto i = 0; i < number_neurons; i++) {
        golden_ids.emplace_back(false, i);
        golden_position.emplace_back(get_random_position_in_box(min_pos, max_pos));
        golden_area_names.emplace_back("area_" + std::to_string(i));
        golden_signal_types.emplace_back(get_random_signal_type());
    }

    const auto idx1 = get_random_integer<size_t>(0, number_neurons - 1);
    auto idx2 = get_random_integer<size_t>(0, number_neurons - 2);

    if (idx1 <= idx2) {
        idx2++;
    }

    std::swap(golden_ids[idx1], golden_ids[idx2]);

    std::filesystem::path path{ "./neurons.tmp" };

    NeuronIO::write_neurons_componentwise(golden_ids, golden_position, golden_area_names, golden_signal_types, path);

    ASSERT_THROW(const auto& val = NeuronIO::read_neurons(path);, RelearnException);
}

TEST_F(IOTest, testNeuronIOReadPositionXException) {
    const auto number_neurons = get_random_number_neurons();

    auto golden_ids = std::vector<NeuronID>{};
    auto golden_position = std::vector<RelearnTypes::position_type>{};
    auto golden_area_names = std::vector<std::string>{};
    auto golden_signal_types = std::vector<SignalType>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = get_maximum_position();

    for (auto i = 0; i < number_neurons; i++) {
        golden_ids.emplace_back(false, i);
        golden_position.emplace_back(get_random_position_in_box(min_pos, max_pos));
        golden_area_names.emplace_back("area_" + std::to_string(i));
        golden_signal_types.emplace_back(get_random_signal_type());
    }

    const auto idx1 = get_random_integer<size_t>(0, number_neurons - 1);

    golden_position[idx1].set_x(-golden_position[idx1].get_x());

    std::filesystem::path path{ "./neurons.tmp" };
    NeuronIO::write_neurons_componentwise(golden_ids, golden_position, golden_area_names, golden_signal_types, path);
    ASSERT_THROW(const auto& val = NeuronIO::read_neurons(path);, RelearnException);
}

TEST_F(IOTest, testNeuronIOReadPositionYException) {
    const auto number_neurons = get_random_number_neurons();

    auto golden_ids = std::vector<NeuronID>{};
    auto golden_position = std::vector<RelearnTypes::position_type>{};
    auto golden_area_names = std::vector<std::string>{};
    auto golden_signal_types = std::vector<SignalType>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = get_maximum_position();

    for (auto i = 0; i < number_neurons; i++) {
        golden_ids.emplace_back(false, i);
        golden_position.emplace_back(get_random_position_in_box(min_pos, max_pos));
        golden_area_names.emplace_back("area_" + std::to_string(i));
        golden_signal_types.emplace_back(get_random_signal_type());
    }

    const auto idx1 = get_random_integer<size_t>(0, number_neurons - 1);

    golden_position[idx1].set_y(-golden_position[idx1].get_y());

    std::filesystem::path path{ "./neurons.tmp" };
    NeuronIO::write_neurons_componentwise(golden_ids, golden_position, golden_area_names, golden_signal_types, path);
    ASSERT_THROW(const auto& val = NeuronIO::read_neurons(path);, RelearnException);
}

TEST_F(IOTest, testNeuronIOReadPositionZException) {
    const auto number_neurons = get_random_number_neurons();

    auto golden_ids = std::vector<NeuronID>{};
    auto golden_position = std::vector<RelearnTypes::position_type>{};
    auto golden_area_names = std::vector<std::string>{};
    auto golden_signal_types = std::vector<SignalType>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = get_maximum_position();

    for (auto i = 0; i < number_neurons; i++) {
        golden_ids.emplace_back(false, i);
        golden_position.emplace_back(get_random_position_in_box(min_pos, max_pos));
        golden_area_names.emplace_back("area_" + std::to_string(i));
        golden_signal_types.emplace_back(get_random_signal_type());
    }

    const auto idx1 = get_random_integer<size_t>(0, number_neurons - 1);

    golden_position[idx1].set_z(-golden_position[idx1].get_z());

    std::filesystem::path path{ "./neurons.tmp" };
    NeuronIO::write_neurons_componentwise(golden_ids, golden_position, golden_area_names, golden_signal_types, path);
    ASSERT_THROW(const auto& val = NeuronIO::read_neurons(path);, RelearnException);
}

TEST_F(IOTest, testNeuronIOReadIDs) {
    const auto number_neurons = get_random_number_neurons();

    auto golden_ids = std::vector<NeuronID>{};
    auto golden_position = std::vector<RelearnTypes::position_type>{};
    auto golden_area_names = std::vector<std::string>{};
    auto golden_signal_types = std::vector<SignalType>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = get_maximum_position();

    for (auto i = 0; i < number_neurons; i++) {
        golden_ids.emplace_back(false, i);
        golden_position.emplace_back(get_random_position_in_box(min_pos, max_pos));
        golden_area_names.emplace_back("area_" + std::to_string(i));
        golden_signal_types.emplace_back(get_random_signal_type());
    }

    std::filesystem::path path{ "./neurons.tmp" };

    NeuronIO::write_neurons_componentwise(golden_ids, golden_position, golden_area_names, golden_signal_types, path);

    const auto& read_ids = NeuronIO::read_neuron_ids(path);

    ASSERT_EQ(read_ids, golden_ids);
}

TEST_F(IOTest, testNeuronIOReadIDsEmpty1) {
    const auto number_neurons = get_random_number_neurons();

    auto golden_ids = std::vector<NeuronID>{};
    auto golden_position = std::vector<RelearnTypes::position_type>{};
    auto golden_area_names = std::vector<std::string>{};
    auto golden_signal_types = std::vector<SignalType>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = get_maximum_position();

    for (auto i = 0; i < number_neurons; i++) {
        golden_ids.emplace_back(false, i);
        golden_position.emplace_back(get_random_position_in_box(min_pos, max_pos));
        golden_area_names.emplace_back("area_" + std::to_string(i));
        golden_signal_types.emplace_back(get_random_signal_type());
    }

    const auto idx1 = get_random_integer<size_t>(0, number_neurons - 1);
    auto idx2 = get_random_integer<size_t>(0, number_neurons - 2);

    if (idx1 <= idx2) {
        idx2++;
    }

    std::swap(golden_ids[idx1], golden_ids[idx2]);

    std::filesystem::path path{ "./neurons.tmp" };

    NeuronIO::write_neurons_componentwise(golden_ids, golden_position, golden_area_names, golden_signal_types, path);

    const auto& ids = NeuronIO::read_neuron_ids(path);
    ASSERT_FALSE(ids.has_value());
}

TEST_F(IOTest, testNeuronIOReadIDsEmpty2) {
    const auto number_neurons = get_random_number_neurons();

    auto golden_ids = std::vector<NeuronID>{};
    auto golden_position = std::vector<RelearnTypes::position_type>{};
    auto golden_area_names = std::vector<std::string>{};
    auto golden_signal_types = std::vector<SignalType>{};

    const auto& min_pos = RelearnTypes::position_type{ 0.0, 0.0, 0.0 };
    const auto& max_pos = get_maximum_position();

    for (auto i = 0; i < number_neurons; i++) {
        golden_ids.emplace_back(false, i);
        golden_position.emplace_back(get_random_position_in_box(min_pos, max_pos));
        golden_area_names.emplace_back("area_" + std::to_string(i));
        golden_signal_types.emplace_back(get_random_signal_type());
    }

    golden_ids.emplace_back(false, number_neurons + 1);
    golden_position.emplace_back(get_random_position_in_box(min_pos, max_pos));
    golden_area_names.emplace_back("area_" + std::to_string(number_neurons + 1));
    golden_signal_types.emplace_back(get_random_signal_type());

    std::filesystem::path path{ "./neurons.tmp" };

    NeuronIO::write_neurons_componentwise(golden_ids, golden_position, golden_area_names, golden_signal_types, path);

    const auto& ids = NeuronIO::read_neuron_ids(path);
    ASSERT_FALSE(ids.has_value());
}

TEST_F(IOTest, testNeuronIOReadIDsFileNotFound) {
    std::filesystem::path path{ "" };
    const auto& ids = NeuronIO::read_neuron_ids(path);
    ASSERT_FALSE(ids.has_value());
}
