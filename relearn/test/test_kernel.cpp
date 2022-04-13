#include "gtest/gtest.h"

#include "RelearnTest.hpp"

#include "algorithm/Kernel/Gaussian.h"
#include "algorithm/Kernel/Kernel.h"
#include "algorithm/Cells.h"
#include "util/Random.h"

#include <array>
#include <iostream>
#include <tuple>

TEST_F(KernelTest, testGaussianSamePosition) {
    const auto sigma = get_random_double(0.001, 100000);
    const auto number_elements = get_random_integer<unsigned int>(0, 10000);
    const auto converted_double = static_cast<double>(number_elements);

    const auto& position = get_random_position();

    GaussianKernel::set_sigma(sigma);
    const auto attractiveness = GaussianKernel::calculate_attractiveness_to_connect(position, position, number_elements);

    ASSERT_NEAR(attractiveness, converted_double, eps);
}

TEST_F(KernelTest, testGaussianNoFreeElements) {
    const auto sigma = get_random_double(0.001, 100000);

    const auto& source_position = get_random_position();
    const auto& target_position = get_random_position();

    GaussianKernel::set_sigma(sigma);
    const auto attractiveness = GaussianKernel::calculate_attractiveness_to_connect(source_position, target_position, 0);

    ASSERT_EQ(attractiveness, 0.0);
}

TEST_F(KernelTest, testGaussianConstantSigma) {
    const auto sigma = get_random_double(0.001, 100000);

    const auto& source_position = get_random_position();
    const auto& target_position = get_random_position();

    GaussianKernel::set_sigma(sigma);
    const auto attractiveness_one = GaussianKernel::calculate_attractiveness_to_connect(source_position, target_position, 1);

    for (auto number_free_elements = 0U; number_free_elements < 10000U; number_free_elements++) {
        const auto attractiveness = GaussianKernel::calculate_attractiveness_to_connect(source_position, target_position, number_free_elements);

        const auto expected_attractiveness = attractiveness_one * number_free_elements;
        ASSERT_NEAR(attractiveness, expected_attractiveness, eps);
    }
}

TEST_F(KernelTest, testGaussianVariableSigma) {
    const auto number_elements = get_random_integer<unsigned int>(0, 10000);
    const auto converted_double = static_cast<double>(number_elements);

    const auto& source_position = get_random_position();
    const auto& target_position = get_random_position();

    std::vector<double> sigmas{};
    for (auto i = 0; i < 100; i++) {
        sigmas.emplace_back(get_random_double(0.001, 100000));
    }

    std::sort(sigmas.begin(), sigmas.end());

    std::vector<double> attractivenesses{};
    for (auto i = 0; i < 100; i++) {
        GaussianKernel::set_sigma(sigmas[i]);
        attractivenesses.emplace_back(GaussianKernel::calculate_attractiveness_to_connect(source_position, target_position, number_elements));
    }

    for (auto i = 1; i < 100; i++) {
        const auto attractiveness_a = attractivenesses[i - 1];
        const auto attractiveness_b = attractivenesses[i];

        ASSERT_LE(attractiveness_a, attractiveness_b);
    }
}

TEST_F(KernelTest, testGaussianVariablePosition) {
    const auto sigma = get_random_double(0.001, 100000);
    GaussianKernel::set_sigma(sigma);

    const auto& source_position = get_random_position();

    const auto number_elements = get_random_integer<unsigned int>(0, 10000);
    const auto converted_double = static_cast<double>(number_elements);

    std::vector<Vec3d> positions{};
    for (auto i = 0; i < 100; i++) {
        positions.emplace_back(get_random_position());
    }

    std::sort(positions.begin(), positions.end(), [&](const Vec3d& pos_a, const Vec3d& pos_b) {
        const auto& dist_a = (pos_a - source_position).calculate_squared_2_norm();
        const auto& dist_b = (pos_b - source_position).calculate_squared_2_norm();
        return dist_a < dist_b;
    });

    std::vector<double> attractivenesses{};
    for (auto i = 0; i < 100; i++) {
        const auto attr_a = GaussianKernel::calculate_attractiveness_to_connect(source_position, positions[i], number_elements);
        const auto attr_b = GaussianKernel::calculate_attractiveness_to_connect(positions[i], source_position, number_elements);

        ASSERT_NEAR(attr_a, attr_b, eps);
        attractivenesses.emplace_back(attr_a);
    }

    for (auto i = 1; i < 100; i++) {
        const auto attractiveness_a = attractivenesses[i - 1];
        const auto attractiveness_b = attractivenesses[i];

        ASSERT_GE(attractiveness_a, attractiveness_b);
    }
}

TEST_F(KernelTest, testGaussianConstantDistance) {
    const auto sigma = get_random_double(0.001, 100000);
    GaussianKernel::set_sigma(sigma);

    const auto& source_position = get_random_position();
    const auto& [x, y, z] = source_position;

    const auto number_elements = get_random_integer<unsigned int>(0, 10000);

    const auto distance = get_random_position_element();

    const auto sqrt3 = std::sqrt(3);

    const Vec3d target_position_1{ x + distance, y + distance, z + distance };
    const Vec3d target_position_2{ x + (sqrt3 * distance), y, z };
    const Vec3d target_position_3{ x, y + (sqrt3 * distance), z };
    const Vec3d target_position_4{ x, y, z + (sqrt3 * distance) };

    const auto attr_1 = GaussianKernel::calculate_attractiveness_to_connect(source_position, target_position_1, number_elements);
    const auto attr_2 = GaussianKernel::calculate_attractiveness_to_connect(source_position, target_position_2, number_elements);
    const auto attr_3 = GaussianKernel::calculate_attractiveness_to_connect(source_position, target_position_3, number_elements);
    const auto attr_4 = GaussianKernel::calculate_attractiveness_to_connect(source_position, target_position_4, number_elements);

    ASSERT_NEAR(attr_1, attr_2, eps);
    ASSERT_NEAR(attr_1, attr_3, eps);
    ASSERT_NEAR(attr_1, attr_4, eps);
}

TEST_F(KernelTest, testGaussianPrecalculatedValues) {
    std::array<std::tuple<double, double, double>, 5> precalculated_values{
        {
            { 100.0, 250.0, 0.85214378896621133 },
            { 20.0, 100.0, 0.96078943915232320 },
            { 10.0, 0.3, 0.0 },
            { 10.0, 20.3, 0.784533945772685 },
            { 15.0, 175, 0.992679984005486 },
        }
    };

    const auto sqrt3 = std::sqrt(3);

    for (const auto& [position_difference, sigma, golden_attractiveness] : precalculated_values) {
        const auto& source_position = get_random_position();
        const auto& target_position = source_position + (position_difference / sqrt3);

        GaussianKernel::set_sigma(sigma);
        const auto attractiveness = GaussianKernel::calculate_attractiveness_to_connect(source_position, target_position, 1);
        ASSERT_NEAR(attractiveness, golden_attractiveness, eps);
    }
}

TEST_F(KernelTest, testGaussianSameNode) {
    const auto& neuron_id = get_random_neuron_id(1000);

    const auto& position = get_random_position();

    const auto element_type = get_random_element_type();
    const auto signal_type = get_random_signal_type();

    const auto sigma = get_random_double(0.001, 100000);
    GaussianKernel::set_sigma(sigma);

    OctreeNode<BarnesHutCell> node{};
    node.set_cell_neuron_id(neuron_id);

    const auto attractiveness = Kernel<BarnesHutCell, GaussianKernel>::calculate_attractiveness_to_connect(neuron_id, position, &node, element_type, signal_type);
    ASSERT_EQ(attractiveness, 0.0);
}

TEST_F(KernelTest, testGaussianDifferentNode) {
    const auto& neuron_id_1 = get_random_neuron_id(1000);
    const auto& neuron_id_2 = get_random_neuron_id(1000, 1000);

    const auto& source_position = get_random_position();

    const auto sigma = get_random_double(0.001, 100000);
    GaussianKernel::set_sigma(sigma);

    const auto& target_excitatory_axon_position = get_random_position();
    const auto& target_inhibitory_axon_position = get_random_position();
    const auto& target_excitatory_dendrite_position = get_random_position();
    const auto& target_inhibitory_dendrite_position = get_random_position();

    const auto& number_vacant_excitatory_axons = get_random_synaptic_element_count();
    const auto& number_vacant_inhibitory_axons = get_random_synaptic_element_count();
    const auto& number_vacant_excitatory_dendrites = get_random_synaptic_element_count();
    const auto& number_vacant_inhibitory_dendrites = get_random_synaptic_element_count();

    OctreeNode<FastMultipoleMethodsCell> node{};
    node.set_cell_neuron_id(neuron_id_1);
    node.set_cell_size(get_minimum_position(), get_maximum_position());

    node.set_cell_excitatory_axons_position(target_excitatory_axon_position);
    node.set_cell_inhibitory_axons_position(target_inhibitory_axon_position);
    node.set_cell_excitatory_dendrites_position(target_excitatory_dendrite_position);
    node.set_cell_inhibitory_dendrites_position(target_inhibitory_dendrite_position);

    node.set_cell_number_axons(number_vacant_excitatory_axons, number_vacant_inhibitory_axons);
    node.set_cell_number_dendrites(number_vacant_excitatory_dendrites, number_vacant_inhibitory_dendrites);

    const auto attr_exc_axons = Kernel<FastMultipoleMethodsCell, GaussianKernel>::calculate_attractiveness_to_connect(neuron_id_2, source_position, &node, ElementType::Axon, SignalType::Excitatory);
    const auto attr_inh_axons = Kernel<FastMultipoleMethodsCell, GaussianKernel>::calculate_attractiveness_to_connect(neuron_id_2, source_position, &node, ElementType::Axon, SignalType::Inhibitory);
    const auto attr_exc_dendrites = Kernel<FastMultipoleMethodsCell, GaussianKernel>::calculate_attractiveness_to_connect(neuron_id_2, source_position, &node, ElementType::Dendrite, SignalType::Excitatory);
    const auto attr_inh_dendrites = Kernel<FastMultipoleMethodsCell, GaussianKernel>::calculate_attractiveness_to_connect(neuron_id_2, source_position, &node, ElementType::Dendrite, SignalType::Inhibitory);

    const auto golden_attr_exc_axons = GaussianKernel::calculate_attractiveness_to_connect(source_position, target_excitatory_axon_position, number_vacant_excitatory_axons);
    const auto golden_attr_inh_axons = GaussianKernel::calculate_attractiveness_to_connect(source_position, target_inhibitory_axon_position, number_vacant_inhibitory_axons);
    const auto golden_attr_exc_dendrites = GaussianKernel::calculate_attractiveness_to_connect(source_position, target_excitatory_dendrite_position, number_vacant_excitatory_dendrites);
    const auto golden_attr_inh_dendrites = GaussianKernel::calculate_attractiveness_to_connect(source_position, target_inhibitory_dendrite_position, number_vacant_inhibitory_dendrites);

    ASSERT_EQ(attr_exc_axons, golden_attr_exc_axons);
    ASSERT_EQ(attr_inh_axons, golden_attr_inh_axons);
    ASSERT_EQ(attr_exc_dendrites, golden_attr_exc_dendrites);
    ASSERT_EQ(attr_inh_dendrites, golden_attr_inh_dendrites);
}

TEST_F(KernelTest, testGaussianException) {
    const auto& neuron_id_1 = get_random_neuron_id(1000);
    const auto& neuron_id_2 = get_random_neuron_id(1000, 1000);

    const auto& source_position = get_random_position();

    const auto sigma = get_random_double(0.001, 100000);
    GaussianKernel::set_sigma(sigma);

    const auto& number_vacant_excitatory_axons = get_random_synaptic_element_count();
    const auto& number_vacant_inhibitory_axons = get_random_synaptic_element_count();
    const auto& number_vacant_excitatory_dendrites = get_random_synaptic_element_count();
    const auto& number_vacant_inhibitory_dendrites = get_random_synaptic_element_count();

    OctreeNode<FastMultipoleMethodsCell> node{};
    node.set_cell_neuron_id(neuron_id_1);
    node.set_cell_size(get_minimum_position(), get_maximum_position());

    node.set_cell_excitatory_axons_position({});
    node.set_cell_inhibitory_axons_position({});
    node.set_cell_excitatory_dendrites_position({});
    node.set_cell_inhibitory_dendrites_position({});

    node.set_cell_number_axons(number_vacant_excitatory_axons, number_vacant_inhibitory_axons);
    node.set_cell_number_dendrites(number_vacant_excitatory_dendrites, number_vacant_inhibitory_dendrites);

    using type = Kernel<FastMultipoleMethodsCell, GaussianKernel>;
    ASSERT_THROW(const auto attr_exc_axons = type::calculate_attractiveness_to_connect(neuron_id_2, source_position, &node, ElementType::Axon, SignalType::Excitatory);, RelearnException);
    ASSERT_THROW(const auto attr_inh_axons = type::calculate_attractiveness_to_connect(neuron_id_2, source_position, &node, ElementType::Axon, SignalType::Inhibitory);, RelearnException);
    ASSERT_THROW(const auto attr_exc_dendrites = type::calculate_attractiveness_to_connect(neuron_id_2, source_position, &node, ElementType::Dendrite, SignalType::Excitatory);, RelearnException);
    ASSERT_THROW(const auto attr_inh_dendrites = type::calculate_attractiveness_to_connect(neuron_id_2, source_position, &node, ElementType::Dendrite, SignalType::Inhibitory);, RelearnException);
}

TEST_F(KernelTest, testGaussianEmptyVector) {
    const auto& neuron_id = get_random_neuron_id(1000);

    const auto& position = get_random_position();

    const auto element_type = get_random_element_type();
    const auto signal_type = get_random_signal_type();

    const auto sigma = get_random_double(0.001, 100000);
    GaussianKernel::set_sigma(sigma);

    const auto number_nodes = get_random_number_neurons();

    const auto& [sum, attrs] = Kernel<FastMultipoleMethodsCell, GaussianKernel>::create_probability_interval(
        neuron_id, position, {}, element_type, signal_type);

    ASSERT_EQ(sum, 0.0);
    ASSERT_EQ(0, attrs.size());
}

TEST_F(KernelTest, testGaussianAutapseVector) {
    const auto& neuron_id = get_random_neuron_id(1000);

    const auto& position = get_random_position();

    const auto element_type = get_random_element_type();
    const auto signal_type = get_random_signal_type();

    const auto sigma = get_random_double(0.001, 100000);
    GaussianKernel::set_sigma(sigma);

    const auto number_nodes = get_random_number_neurons();

    std::vector<OctreeNode<FastMultipoleMethodsCell>> nodes{ number_nodes, OctreeNode<FastMultipoleMethodsCell>{} };
    std::vector<OctreeNode<FastMultipoleMethodsCell>*> node_pointers{ number_nodes, nullptr };

    for (auto i = 0; i < number_nodes; i++) {
        nodes[i].set_cell_neuron_id(neuron_id);
        nodes[i].set_cell_size(get_minimum_position(), get_maximum_position());

        const auto& target_excitatory_axon_position = get_random_position();
        const auto& target_inhibitory_axon_position = get_random_position();
        const auto& target_excitatory_dendrite_position = get_random_position();
        const auto& target_inhibitory_dendrite_position = get_random_position();

        const auto& number_vacant_excitatory_axons = get_random_synaptic_element_count();
        const auto& number_vacant_inhibitory_axons = get_random_synaptic_element_count();
        const auto& number_vacant_excitatory_dendrites = get_random_synaptic_element_count();
        const auto& number_vacant_inhibitory_dendrites = get_random_synaptic_element_count();

        nodes[i].set_cell_excitatory_axons_position(target_excitatory_axon_position);
        nodes[i].set_cell_inhibitory_axons_position(target_inhibitory_axon_position);
        nodes[i].set_cell_excitatory_dendrites_position(target_excitatory_dendrite_position);
        nodes[i].set_cell_inhibitory_dendrites_position(target_inhibitory_dendrite_position);

        nodes[i].set_cell_number_axons(number_vacant_excitatory_axons, number_vacant_inhibitory_axons);
        nodes[i].set_cell_number_dendrites(number_vacant_excitatory_dendrites, number_vacant_inhibitory_dendrites);

        node_pointers[i] = &nodes[i];
    }

    const auto& [sum, attrs] = Kernel<FastMultipoleMethodsCell, GaussianKernel>::create_probability_interval(
        neuron_id, position, node_pointers, element_type, signal_type);

    ASSERT_EQ(sum, 0.0);
    ASSERT_EQ(0, attrs.size());
}

TEST_F(KernelTest, testGaussianVectorException) {
    const auto& neuron_id = get_random_neuron_id(1000);

    const auto& position = get_random_position();

    const auto element_type = get_random_element_type();
    const auto signal_type = get_random_signal_type();

    const auto sigma = get_random_double(0.001, 100000);
    GaussianKernel::set_sigma(sigma);

    const auto number_nodes = get_random_number_neurons();

    std::vector<OctreeNode<FastMultipoleMethodsCell>> nodes{ number_nodes, OctreeNode<FastMultipoleMethodsCell>{} };
    std::vector<OctreeNode<FastMultipoleMethodsCell>*> node_pointers{ number_nodes, nullptr };

    for (auto i = 0; i < number_nodes; i++) {
        nodes[i].set_cell_neuron_id(get_random_neuron_id(1000, 1000));
        nodes[i].set_cell_size(get_minimum_position(), get_maximum_position());

        const auto& target_excitatory_axon_position = get_random_position();
        const auto& target_inhibitory_axon_position = get_random_position();
        const auto& target_excitatory_dendrite_position = get_random_position();
        const auto& target_inhibitory_dendrite_position = get_random_position();

        const auto& number_vacant_excitatory_axons = get_random_synaptic_element_count();
        const auto& number_vacant_inhibitory_axons = get_random_synaptic_element_count();
        const auto& number_vacant_excitatory_dendrites = get_random_synaptic_element_count();
        const auto& number_vacant_inhibitory_dendrites = get_random_synaptic_element_count();

        nodes[i].set_cell_excitatory_axons_position(target_excitatory_axon_position);
        nodes[i].set_cell_inhibitory_axons_position(target_inhibitory_axon_position);
        nodes[i].set_cell_excitatory_dendrites_position(target_excitatory_dendrite_position);
        nodes[i].set_cell_inhibitory_dendrites_position(target_inhibitory_dendrite_position);

        nodes[i].set_cell_number_axons(number_vacant_excitatory_axons, number_vacant_inhibitory_axons);
        nodes[i].set_cell_number_dendrites(number_vacant_excitatory_dendrites, number_vacant_inhibitory_dendrites);

        node_pointers[i] = &nodes[i];
    }

    const auto nullptr_index = get_random_integer<size_t>(0, number_nodes - 1);
    node_pointers[nullptr_index] = nullptr;

    using TT = Kernel<FastMultipoleMethodsCell, GaussianKernel>;

    ASSERT_THROW(const auto& val = TT::create_probability_interval(neuron_id, position, node_pointers, element_type, signal_type);, RelearnException);
}

TEST_F(KernelTest, testGaussianRandomVector) {
    const auto& neuron_id = get_random_neuron_id(1000);

    const auto& position = get_random_position();

    const auto element_type = get_random_element_type();
    const auto signal_type = get_random_signal_type();

    const auto sigma = get_random_double(0.001, 100000);
    GaussianKernel::set_sigma(sigma);

    const auto number_nodes = get_random_number_neurons();

    std::vector<OctreeNode<FastMultipoleMethodsCell>> nodes{ number_nodes, OctreeNode<FastMultipoleMethodsCell>{} };
    std::vector<OctreeNode<FastMultipoleMethodsCell>*> node_pointers{ number_nodes, nullptr };

    for (auto i = 0; i < number_nodes; i++) {
        nodes[i].set_cell_neuron_id(get_random_neuron_id(1000, 1000));
        nodes[i].set_cell_size(get_minimum_position(), get_maximum_position());

        const auto& target_excitatory_axon_position = get_random_position();
        const auto& target_inhibitory_axon_position = get_random_position();
        const auto& target_excitatory_dendrite_position = get_random_position();
        const auto& target_inhibitory_dendrite_position = get_random_position();

        const auto& number_vacant_excitatory_axons = get_random_synaptic_element_count();
        const auto& number_vacant_inhibitory_axons = get_random_synaptic_element_count();
        const auto& number_vacant_excitatory_dendrites = get_random_synaptic_element_count();
        const auto& number_vacant_inhibitory_dendrites = get_random_synaptic_element_count();

        nodes[i].set_cell_excitatory_axons_position(target_excitatory_axon_position);
        nodes[i].set_cell_inhibitory_axons_position(target_inhibitory_axon_position);
        nodes[i].set_cell_excitatory_dendrites_position(target_excitatory_dendrite_position);
        nodes[i].set_cell_inhibitory_dendrites_position(target_inhibitory_dendrite_position);

        nodes[i].set_cell_number_axons(number_vacant_excitatory_axons, number_vacant_inhibitory_axons);
        nodes[i].set_cell_number_dendrites(number_vacant_excitatory_dendrites, number_vacant_inhibitory_dendrites);

        node_pointers[i] = &nodes[i];
    }

    auto total_attractiveness = 0.0;
    std::vector<double> attractivenesses{};
    for (auto i = 0; i < number_nodes; i++) {
        const auto attr = Kernel<FastMultipoleMethodsCell, GaussianKernel>::calculate_attractiveness_to_connect(neuron_id, position, &nodes[i], element_type, signal_type);
        attractivenesses.emplace_back(attr);
        total_attractiveness += attr;
    }

    const auto& [sum, attrs] = Kernel<FastMultipoleMethodsCell, GaussianKernel>::create_probability_interval(
        neuron_id, position, node_pointers, element_type, signal_type);

    ASSERT_NEAR(sum, total_attractiveness, eps);
    ASSERT_EQ(attractivenesses.size(), attrs.size());

    for (auto i = 0; i < number_nodes; i++) {
        ASSERT_NEAR(attrs[i], attractivenesses[i], eps);
    }
}

TEST_F(KernelTest, testPickTargetEmpty) {
    const std::vector<OctreeNode<BarnesHutCell>*> targets{};
    const std::vector<double> probabilities{};

    const auto random_number = get_random_double(0.0, 100.0);

    using TT = Kernel<BarnesHutCell, GaussianKernel>;

    ASSERT_THROW(const auto& val = TT::pick_target(targets, probabilities, random_number);, RelearnException);
}

TEST_F(KernelTest, testPickTargetMismatchSize) {
    const auto number_targets = get_random_integer<size_t>(1, 1000);
    auto number_probabilities = get_random_integer<size_t>(1, 1000);
    while (number_probabilities == number_targets) {
        number_probabilities = get_random_integer<size_t>(1, 1000);
    }

    std::vector<OctreeNode<BarnesHutCell>*> targets{ number_targets };
    std::vector<double> probabilities = std::vector<double>(number_probabilities, 0.0);

    const auto random_number = get_random_double(0.0, 100.0);

    using TT = Kernel<BarnesHutCell, GaussianKernel>;

    ASSERT_THROW(const auto& val = TT::pick_target(targets, probabilities, random_number);, RelearnException);
}

TEST_F(KernelTest, testPickTargetNegativeRandomNumber) {
    const auto number_targets = get_random_integer<size_t>(1, 1000);

    std::vector<OctreeNode<BarnesHutCell>*> targets{ number_targets };
    std::vector<double> probabilities = std::vector<double>(number_targets, 0.0);

    const auto random_number = -get_random_double(0.001, 100.0);

    using TT = Kernel<BarnesHutCell, GaussianKernel>;

    ASSERT_THROW(const auto& val = TT::pick_target(targets, probabilities, random_number);, RelearnException);
}

TEST_F(KernelTest, testPickTargetRandom) {
    const auto number_nodes = get_random_number_neurons();

    std::vector<OctreeNode<BarnesHutCell>> nodes{ number_nodes, OctreeNode<BarnesHutCell>{} };
    std::vector<OctreeNode<BarnesHutCell>*> node_pointers{ number_nodes, nullptr };
    std::vector<double> probabilities{};

    for (auto i = 0; i < number_nodes; i++) {
        node_pointers[i] = &nodes[i];
        probabilities.emplace_back(get_random_percentage());
    }

    const auto total_probability = std::reduce(probabilities.begin(), probabilities.end(), 0.0);

    for (auto it = 0; it < number_nodes; it++) {
        const auto random_number = get_random_double(0.0, total_probability);

        auto current_probability = random_number;
        auto index = -1;

        for (auto i = 0; i < number_nodes; i++) {
            current_probability -= probabilities[i];
            if (current_probability <= 0.0) {
                index = i;
                break;
            }
        }

        if (index == -1) {
            std::cerr << "testPickTargetRandom: the index was -1\n";
            index = number_nodes - 1;
        }

        const auto* chosen_node = Kernel<BarnesHutCell, GaussianKernel>::pick_target(node_pointers, probabilities, random_number);

        ASSERT_EQ(chosen_node, node_pointers[index]);
    }
}

TEST_F(KernelTest, testPickTargetTooLarge) {
    const auto number_nodes = get_random_number_neurons();

    std::vector<OctreeNode<BarnesHutCell>> nodes{ number_nodes, OctreeNode<BarnesHutCell>{} };
    std::vector<OctreeNode<BarnesHutCell>*> node_pointers{ number_nodes, nullptr };
    std::vector<double> probabilities{};

    for (auto i = 0; i < number_nodes; i++) {
        node_pointers[i] = &nodes[i];
        probabilities.emplace_back(get_random_percentage());
    }

    const auto total_probability = std::reduce(probabilities.begin(), probabilities.end(), 0.0);
    

    for (auto it = 0; it < number_nodes; it++) {
        const auto random_number = get_random_double(total_probability + eps, (total_probability + eps + 1) * 2);
        const auto* chosen_node = Kernel<BarnesHutCell, GaussianKernel>::pick_target(node_pointers, probabilities, random_number);

        ASSERT_EQ(chosen_node, node_pointers[number_nodes - 1]);
    }
}

TEST_F(KernelTest, testPickTargetEmpty2) {
    const auto& neuron_id = get_random_neuron_id(1000);

    const auto& position = get_random_position();

    const auto element_type = get_random_element_type();
    const auto signal_type = get_random_signal_type();

    const auto sigma = get_random_double(0.001, 100000);
    GaussianKernel::set_sigma(sigma);

    auto* result = Kernel<BarnesHutCell, GaussianKernel>::pick_target(neuron_id, position, {}, element_type, signal_type);

    ASSERT_EQ(result, nullptr);
}

TEST_F(KernelTest, testPickTargetException) {
    const auto number_nodes = get_random_number_neurons();
    const auto& neuron_id = get_random_neuron_id(1000);

    const auto& position = get_random_position();

    const auto element_type = get_random_element_type();
    const auto signal_type = get_random_signal_type();

    const auto sigma = get_random_double(0.001, 100000);
    GaussianKernel::set_sigma(sigma);

    std::vector<OctreeNode<FastMultipoleMethodsCell>> nodes{ number_nodes, OctreeNode<FastMultipoleMethodsCell>{} };
    std::vector<OctreeNode<FastMultipoleMethodsCell>*> node_pointers{ number_nodes, nullptr };

    for (auto i = 0; i < number_nodes; i++) {
        nodes[i].set_cell_neuron_id(get_random_neuron_id(1000, 1000));
        nodes[i].set_cell_size(get_minimum_position(), get_maximum_position());

        const auto& target_excitatory_axon_position = get_random_position();
        const auto& target_inhibitory_axon_position = get_random_position();
        const auto& target_excitatory_dendrite_position = get_random_position();
        const auto& target_inhibitory_dendrite_position = get_random_position();

        const auto& number_vacant_excitatory_axons = get_random_synaptic_element_count();
        const auto& number_vacant_inhibitory_axons = get_random_synaptic_element_count();
        const auto& number_vacant_excitatory_dendrites = get_random_synaptic_element_count();
        const auto& number_vacant_inhibitory_dendrites = get_random_synaptic_element_count();

        nodes[i].set_cell_excitatory_axons_position(target_excitatory_axon_position);
        nodes[i].set_cell_inhibitory_axons_position(target_inhibitory_axon_position);
        nodes[i].set_cell_excitatory_dendrites_position(target_excitatory_dendrite_position);
        nodes[i].set_cell_inhibitory_dendrites_position(target_inhibitory_dendrite_position);

        nodes[i].set_cell_number_axons(number_vacant_excitatory_axons, number_vacant_inhibitory_axons);
        nodes[i].set_cell_number_dendrites(number_vacant_excitatory_dendrites, number_vacant_inhibitory_dendrites);

        node_pointers[i] = &nodes[i];
    }

    const auto nullptr_index = get_random_integer<size_t>(0, number_nodes - 1);
    node_pointers[nullptr_index] = nullptr;

    using TT = Kernel<FastMultipoleMethodsCell, GaussianKernel>;
    ASSERT_THROW(auto* result = TT::pick_target(neuron_id, position, node_pointers, element_type, signal_type);, RelearnException);
}

TEST_F(KernelTest, testPickTargetRandom2) {
    const auto number_nodes = get_random_number_neurons();
    const auto& neuron_id = get_random_neuron_id(1000);

    const auto& position = get_random_position();

    const auto element_type = get_random_element_type();
    const auto signal_type = get_random_signal_type();

    const auto sigma = get_random_double(0.001, 100000);
    GaussianKernel::set_sigma(sigma);

    std::vector<OctreeNode<FastMultipoleMethodsCell>> nodes{ number_nodes, OctreeNode<FastMultipoleMethodsCell>{} };
    std::vector<OctreeNode<FastMultipoleMethodsCell>*> node_pointers{ number_nodes, nullptr };

    for (auto i = 0; i < number_nodes; i++) {
        nodes[i].set_cell_neuron_id(get_random_neuron_id(1000, 1000));
        nodes[i].set_cell_size(get_minimum_position(), get_maximum_position());

        const auto& target_excitatory_axon_position = get_random_position();
        const auto& target_inhibitory_axon_position = get_random_position();
        const auto& target_excitatory_dendrite_position = get_random_position();
        const auto& target_inhibitory_dendrite_position = get_random_position();

        const auto& number_vacant_excitatory_axons = get_random_synaptic_element_count();
        const auto& number_vacant_inhibitory_axons = get_random_synaptic_element_count();
        const auto& number_vacant_excitatory_dendrites = get_random_synaptic_element_count();
        const auto& number_vacant_inhibitory_dendrites = get_random_synaptic_element_count();

        nodes[i].set_cell_excitatory_axons_position(target_excitatory_axon_position);
        nodes[i].set_cell_inhibitory_axons_position(target_inhibitory_axon_position);
        nodes[i].set_cell_excitatory_dendrites_position(target_excitatory_dendrite_position);
        nodes[i].set_cell_inhibitory_dendrites_position(target_inhibitory_dendrite_position);

        nodes[i].set_cell_number_axons(number_vacant_excitatory_axons, number_vacant_inhibitory_axons);
        nodes[i].set_cell_number_dendrites(number_vacant_excitatory_dendrites, number_vacant_inhibitory_dendrites);

        node_pointers[i] = &nodes[i];
    }

    for (auto i = 0; i < number_nodes; i++) {
        auto* result = Kernel<FastMultipoleMethodsCell, GaussianKernel>::
            pick_target(neuron_id, position, node_pointers, element_type, signal_type);

        auto pos = std::find(node_pointers.begin(), node_pointers.end(), result);
        ASSERT_NE(pos, node_pointers.end());
    }
}
