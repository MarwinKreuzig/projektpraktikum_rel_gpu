#include "gtest/gtest.h"

#include "RelearnTest.hpp"

#include "algorithm/Kernel/Gaussian.h"
#include "algorithm/Kernel/Kernel.h"
#include "algorithm/Cells.h"
#include "util/Random.h"

#include <array>
#include <iostream>
#include <tuple>

TEST_F(ProbabilityKernelTest, testLinearGetterSetter) {
    LinearDistributionKernel::set_cutoff(LinearDistributionKernel::default_cutoff);

    const auto cutoff_point = get_random_linear_cutoff();

    ASSERT_EQ(LinearDistributionKernel::get_cutoff(), LinearDistributionKernel::default_cutoff);
    ASSERT_NO_THROW(LinearDistributionKernel::set_cutoff(cutoff_point));
    ASSERT_EQ(LinearDistributionKernel::get_cutoff(), cutoff_point);
}

TEST_F(ProbabilityKernelTest, testLinearGetterSetterInf) {
    LinearDistributionKernel::set_cutoff(LinearDistributionKernel::default_cutoff);

    constexpr auto cutoff_point_inf = std::numeric_limits<double>::infinity();

    ASSERT_NO_THROW(LinearDistributionKernel::set_cutoff(cutoff_point_inf));
    ASSERT_EQ(LinearDistributionKernel::get_cutoff(), cutoff_point_inf);
}

TEST_F(ProbabilityKernelTest, testLinearGetterSetterException) {
    LinearDistributionKernel::set_cutoff(LinearDistributionKernel::default_cutoff);

    const auto cutoff_point = -get_random_linear_cutoff();

    ASSERT_THROW(LinearDistributionKernel::set_cutoff(cutoff_point), RelearnException);
    ASSERT_EQ(LinearDistributionKernel::get_cutoff(), LinearDistributionKernel::default_cutoff);
}

TEST_F(ProbabilityKernelTest, testLinearNoFreeElements) {
    const auto cutoff_point = get_random_linear_cutoff();
    LinearDistributionKernel::set_cutoff(cutoff_point);

    const auto& source_position = get_random_position();
    const auto& target_position = get_random_position();

    const auto attractiveness = LinearDistributionKernel::calculate_attractiveness_to_connect(source_position, target_position, 0);

    ASSERT_EQ(attractiveness, 0.0);
}

TEST_F(ProbabilityKernelTest, testLinearLinearFreeElements) {
    const auto cutoff_point = get_random_linear_cutoff();
    LinearDistributionKernel::set_cutoff(cutoff_point);

    const auto& source_position = get_random_position();
    const auto& target_position = get_random_position();

    const auto attractiveness_one = LinearDistributionKernel::calculate_attractiveness_to_connect(source_position, target_position, 1);

    for (auto number_free_elements = 0U; number_free_elements < 10000U; number_free_elements++) {
        const auto attractiveness = LinearDistributionKernel::calculate_attractiveness_to_connect(source_position, target_position, number_free_elements);

        const auto expected_attractiveness = attractiveness_one * number_free_elements;
        ASSERT_NEAR(attractiveness, expected_attractiveness, eps);
    }
}

TEST_F(ProbabilityKernelTest, testLinearSamePosition) {
    const auto cutoff_point = get_random_linear_cutoff();
    LinearDistributionKernel::set_cutoff(cutoff_point);

    const auto number_elements = get_random_integer<unsigned int>(0, 10000);
    const auto converted_double = static_cast<double>(number_elements);

    const auto& position = get_random_position();

    const auto attractiveness = LinearDistributionKernel::calculate_attractiveness_to_connect(position, position, number_elements);

    ASSERT_NEAR(attractiveness, converted_double, eps);
}

TEST_F(ProbabilityKernelTest, testLinearInf) {
    constexpr auto cutoff_point_inf = std::numeric_limits<double>::infinity();
    LinearDistributionKernel::set_cutoff(cutoff_point_inf);

    const auto& source = get_random_position();
    const auto& target = get_random_position();

    const auto number_elements = get_random_integer<unsigned int>(0, 10000);

    const auto attraction = LinearDistributionKernel::calculate_attractiveness_to_connect(source, target, number_elements);

    ASSERT_EQ(attraction, static_cast<double>(number_elements));
}

TEST_F(ProbabilityKernelTest, testLinearFinite) {
    const auto cutoff_point = get_random_linear_cutoff();
    LinearDistributionKernel::set_cutoff(cutoff_point);

    for (auto i = 0; i < 100; i++) {
        const auto number_elements = get_random_integer<unsigned int>(0, 10000);

        const auto& source = get_random_position();
        const auto& target = get_random_position();

        const auto attraction = LinearDistributionKernel::calculate_attractiveness_to_connect(source, target, number_elements);

        const auto difference = (source - target).calculate_2_norm();

        if (difference > cutoff_point) {
            ASSERT_EQ(attraction, 0.0);
            continue;
        }

        const auto fixed_attraction = attraction * cutoff_point / difference;

        ASSERT_NEAR(attraction, fixed_attraction, eps);
    }
}

TEST_F(KernelTest, testLinearKernelIntegration) {
    const auto& neuron_id_1 = get_random_neuron_id(1000);
    const auto& neuron_id_2 = get_random_neuron_id(1000, 1000);

    const auto& source_position = get_random_position();

    Kernel<FastMultipoleMethodsCell>::set_kernel_type(KernelType::Linear);

    const auto cutoff_point = get_random_linear_cutoff();
    LinearDistributionKernel::set_cutoff(cutoff_point);

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

    const auto attr_exc_axons = Kernel<FastMultipoleMethodsCell>::calculate_attractiveness_to_connect(neuron_id_2, source_position, &node, ElementType::Axon, SignalType::Excitatory);
    const auto attr_inh_axons = Kernel<FastMultipoleMethodsCell>::calculate_attractiveness_to_connect(neuron_id_2, source_position, &node, ElementType::Axon, SignalType::Inhibitory);
    const auto attr_exc_dendrites = Kernel<FastMultipoleMethodsCell>::calculate_attractiveness_to_connect(neuron_id_2, source_position, &node, ElementType::Dendrite, SignalType::Excitatory);
    const auto attr_inh_dendrites = Kernel<FastMultipoleMethodsCell>::calculate_attractiveness_to_connect(neuron_id_2, source_position, &node, ElementType::Dendrite, SignalType::Inhibitory);

    const auto golden_attr_exc_axons = LinearDistributionKernel::calculate_attractiveness_to_connect(source_position, target_excitatory_axon_position, number_vacant_excitatory_axons);
    const auto golden_attr_inh_axons = LinearDistributionKernel::calculate_attractiveness_to_connect(source_position, target_inhibitory_axon_position, number_vacant_inhibitory_axons);
    const auto golden_attr_exc_dendrites = LinearDistributionKernel::calculate_attractiveness_to_connect(source_position, target_excitatory_dendrite_position, number_vacant_excitatory_dendrites);
    const auto golden_attr_inh_dendrites = LinearDistributionKernel::calculate_attractiveness_to_connect(source_position, target_inhibitory_dendrite_position, number_vacant_inhibitory_dendrites);

    ASSERT_EQ(attr_exc_axons, golden_attr_exc_axons);
    ASSERT_EQ(attr_inh_axons, golden_attr_inh_axons);
    ASSERT_EQ(attr_exc_dendrites, golden_attr_exc_dendrites);
    ASSERT_EQ(attr_inh_dendrites, golden_attr_inh_dendrites);
}
