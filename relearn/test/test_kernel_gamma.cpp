#include "gtest/gtest.h"

#include "RelearnTest.hpp"

#include "algorithm/Kernel/Gaussian.h"
#include "algorithm/Kernel/Kernel.h"
#include "algorithm/Cells.h"
#include "util/Random.h"

#include <array>
#include <iostream>
#include <tuple>

TEST_F(ProbabilityKernelTest, testGammaSetterGetter) {
    GammaDistributionKernel::set_k(GammaDistributionKernel::default_k);
    GammaDistributionKernel::set_theta(GammaDistributionKernel::default_theta);

    ASSERT_EQ(GammaDistributionKernel::get_k(), GammaDistributionKernel::default_k);
    ASSERT_EQ(GammaDistributionKernel::get_theta(), GammaDistributionKernel::default_theta);

    const auto k = get_random_gamma_k();
    const auto theta = get_random_gamma_theta();

    GammaDistributionKernel::set_k(k);
    GammaDistributionKernel::set_theta(theta);

    ASSERT_EQ(GammaDistributionKernel::get_k(), k);
    ASSERT_EQ(GammaDistributionKernel::get_theta(), theta);
}

TEST_F(ProbabilityKernelTest, testGammaSetterGetterException) {
    const auto k = get_random_gamma_k();
    const auto theta = get_random_gamma_theta();

    GammaDistributionKernel::set_k(k);
    GammaDistributionKernel::set_theta(theta);

    ASSERT_THROW(GammaDistributionKernel::set_k(0.0), RelearnException);
    ASSERT_THROW(GammaDistributionKernel::set_k(-k), RelearnException);
    ASSERT_THROW(GammaDistributionKernel::set_theta(0.0), RelearnException);
    ASSERT_THROW(GammaDistributionKernel::set_theta(-theta), RelearnException);

    ASSERT_EQ(GammaDistributionKernel::get_k(), k);
    ASSERT_EQ(GammaDistributionKernel::get_theta(), theta);
}

TEST_F(ProbabilityKernelTest, testGammaNoFreeElements) {
    GammaDistributionKernel::set_k(GammaDistributionKernel::default_k);
    GammaDistributionKernel::set_theta(GammaDistributionKernel::default_theta);

    const auto k = get_random_gamma_k();
    const auto theta = get_random_gamma_theta();

    GammaDistributionKernel::set_k(k);
    GammaDistributionKernel::set_theta(theta);

    const auto& source_position = get_random_position();
    const auto& target_position = get_random_position();

    const auto attractiveness = GammaDistributionKernel::calculate_attractiveness_to_connect(source_position, target_position, 0);

    ASSERT_EQ(attractiveness, 0.0);
}

TEST_F(ProbabilityKernelTest, testGammaLinearElements) {
    GammaDistributionKernel::set_k(GammaDistributionKernel::default_k);
    GammaDistributionKernel::set_theta(GammaDistributionKernel::default_theta);

    const auto k = get_random_gamma_k();
    const auto theta = get_random_gamma_theta();

    GammaDistributionKernel::set_k(k);
    GammaDistributionKernel::set_theta(theta);

    const auto& source_position = get_random_position();
    const auto& target_position = get_random_position();

    const auto attractiveness_one = GammaDistributionKernel::calculate_attractiveness_to_connect(source_position, target_position, 1);

    for (auto number_free_elements = 0U; number_free_elements < 10000U; number_free_elements++) {
        const auto attractiveness = GammaDistributionKernel::calculate_attractiveness_to_connect(source_position, target_position, number_free_elements);

        const auto expected_attractiveness = attractiveness_one * number_free_elements;
        ASSERT_NEAR(attractiveness, expected_attractiveness, eps);
    }
}

TEST_F(ProbabilityKernelTest, testGammaSamePosition) {
    // Unfortunately, this test does not make sense for a gamma distribution

    //GammaDistributionKernel::set_k(GammaDistributionKernel::default_k);
    //GammaDistributionKernel::set_theta(GammaDistributionKernel::default_theta);

    //const auto k = get_random_gamma_k();
    //const auto theta = get_random_gamma_theta();

    //GammaDistributionKernel::set_k(k);
    //GammaDistributionKernel::set_theta(theta);

    //const auto number_elements = get_random_integer<unsigned int>(0, 10000);
    //const auto& position = get_random_position();
    //const auto attractiveness = GammaDistributionKernel::calculate_attractiveness_to_connect(position, position, number_elements);

    //ASSERT_NEAR(attractiveness, 0.0, eps);
}

TEST_F(ProbabilityKernelTest, testGammaPrecalculatedValues) {
    GammaDistributionKernel::set_k(GammaDistributionKernel::default_k);
    GammaDistributionKernel::set_theta(GammaDistributionKernel::default_theta);

    std::array<std::tuple<double, double, double, double>, 5> precalculated_values{
        {
            { 100.0, 10.0, 10.0, 0.012511 },
            { 20.0, 1.0, 10.0, 0.0135335 },
            { 0.15, 1.0, 1.0, 0.860708 },
            { 7.0, 1.0, 1 / 1.4, 0.0000776322 },
            { 6.25, 3.0, 11.4, 0.00761931 },
        }
    };

    const auto sqrt3 = std::sqrt(3);

    for (const auto& [position_difference, k, theta, expected] : precalculated_values) {
        const auto& source_position = get_random_position();
        const auto& target_position = source_position + (position_difference / sqrt3);

        GammaDistributionKernel::set_k(k);
        GammaDistributionKernel::set_theta(theta);

        const auto attractiveness = GammaDistributionKernel::calculate_attractiveness_to_connect(source_position, target_position, 1);
        ASSERT_NEAR(attractiveness, expected, eps);
    }
}


TEST_F(KernelTest, testGammaKernelIntegration) {
    const auto& neuron_id_1 = get_random_neuron_id(1000);
    const auto& neuron_id_2 = get_random_neuron_id(1000, 1000);

    const auto& source_position = get_random_position();

    Kernel<FastMultipoleMethodsCell>::set_kernel_type(KernelType::Gamma);

    const auto k = get_random_gamma_k();
    const auto theta = get_random_gamma_theta();

    GammaDistributionKernel::set_k(k);
    GammaDistributionKernel::set_theta(theta);

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

    const auto golden_attr_exc_axons = GammaDistributionKernel::calculate_attractiveness_to_connect(source_position, target_excitatory_axon_position, number_vacant_excitatory_axons);
    const auto golden_attr_inh_axons = GammaDistributionKernel::calculate_attractiveness_to_connect(source_position, target_inhibitory_axon_position, number_vacant_inhibitory_axons);
    const auto golden_attr_exc_dendrites = GammaDistributionKernel::calculate_attractiveness_to_connect(source_position, target_excitatory_dendrite_position, number_vacant_excitatory_dendrites);
    const auto golden_attr_inh_dendrites = GammaDistributionKernel::calculate_attractiveness_to_connect(source_position, target_inhibitory_dendrite_position, number_vacant_inhibitory_dendrites);

    ASSERT_EQ(attr_exc_axons, golden_attr_exc_axons);
    ASSERT_EQ(attr_inh_axons, golden_attr_inh_axons);
    ASSERT_EQ(attr_exc_dendrites, golden_attr_exc_dendrites);
    ASSERT_EQ(attr_inh_dendrites, golden_attr_inh_dendrites);
}
