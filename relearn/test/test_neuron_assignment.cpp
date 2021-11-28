#include "../googletest/include/gtest/gtest.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "RelearnTest.hpp"

#include "../source/sim/NeuronToSubdomainAssignment.h"
#include "../source/structure/Partition.h"
#include "../source/sim/SubdomainFromNeuronDensity.h"
#include "../source/sim/SubdomainFromFile.h"
#include "../source/util/RelearnException.h"

void sort_indices(const std::vector<Vec3d>& vectors, std::vector<int>& sequence) {
    sequence = std::vector<int>(vectors.size());
    std::iota(sequence.begin(), sequence.end(), 0);

    std::stable_sort(sequence.begin(), sequence.end(),
        [&vectors](int i1, int i2) { return vectors[i1] < vectors[i2]; });
}

void check_types_fraction(std::vector<SignalType>& types, double& frac_ex, unsigned long long total_subdomains, const size_t& number_neurons) {
    size_t neurons_ex = 0;
    size_t neurons_in = 0;

    for (auto& type : types) {
        if (type == SignalType::EXCITATORY) {
            neurons_ex++;
        } else if (type == SignalType::INHIBITORY) {
            neurons_in++;
        }
    }

    auto frac_ex_ = (static_cast<double>(neurons_ex) / static_cast<double>(neurons_ex + neurons_in));
    ASSERT_NEAR(frac_ex, frac_ex_, static_cast<double>(total_subdomains) / static_cast<double>(number_neurons));
}

void check_positions(std::vector<NeuronToSubdomainAssignment::position_type>& pos, double um_per_neuron, size_t& expected_neurons_per_dimension, bool* flags) {
    std::vector<Vec3<size_t>> pos_fixed;
    for (auto& p : pos) {
        pos_fixed.emplace_back((Vec3<size_t>)(p * (1 / um_per_neuron)));
    }

    for (auto& p : pos_fixed) {
        auto x = p.get_x();
        auto y = p.get_y();
        auto z = p.get_z();

        ASSERT_LE(x, expected_neurons_per_dimension);
        ASSERT_LE(y, expected_neurons_per_dimension);
        ASSERT_LE(z, expected_neurons_per_dimension);

        auto idx = x * expected_neurons_per_dimension * expected_neurons_per_dimension + y * expected_neurons_per_dimension + z;
        auto& val = flags[idx];
        ASSERT_FALSE(val);
        val = true;
    }
}

void generate_neuron_positions(std::vector<Vec3d>& positions,
    std::vector<std::string>& area_names, std::vector<SignalType>& types, std::mt19937& mt) {

    std::uniform_int_distribution<size_t> uid(1, 1000);
    std::uniform_real_distribution<double> urd(0.0, 1.0);

    auto number_neurons = uid(mt);
    auto frac_ex = urd(mt);
    auto um_per_neuron = urd(mt) * 100;

    auto part = std::make_shared<Partition>(1, 0);
    SubdomainFromNeuronDensity sfnd{ number_neurons, frac_ex, um_per_neuron, part };

    auto num_neurons_ = sfnd.get_requested_number_neurons();

    ASSERT_EQ(number_neurons, num_neurons_);

    const auto& [sim_box_min, sim_box_max] = part->get_simulation_box_size();
    auto box_length = (sim_box_max - sim_box_min).get_maximum();

    sfnd.initialize();

    positions = sfnd.get_neuron_positions_in_subdomain(0, 1);
    area_names = sfnd.get_neuron_area_names_in_subdomain(0, 1);
    types = sfnd.get_neuron_types_in_subdomain(0, 1);

    ASSERT_EQ(number_neurons, positions.size());
    ASSERT_EQ(number_neurons, area_names.size());
    ASSERT_EQ(number_neurons, types.size());

    sfnd.write_neurons_to_file("neurons.tmp");
}

double calculate_excitatory_fraction(const std::vector<SignalType>& types) {
    auto number_excitatory = 0;
    auto number_inhibitory = 0;

    for (const auto& type : types) {
        if (type == SignalType::EXCITATORY) {
            number_excitatory++;
        } else {
            number_inhibitory++;
        }
    }

    const auto ratio = static_cast<double>(number_excitatory) / static_cast<double>(number_excitatory + number_inhibitory);
    return ratio;
}

TEST_F(NeuronAssignmentTest, test_density_too_few_neurons) {
    for (auto i = 0; i < iterations; i++) {
        const auto golden_number_ranks = get_adjusted_random_number_ranks(mt);
        const auto golden_fraction_excitatory_neurons = get_random_percentage(mt);
        const auto golden_um_per_neuron = get_random_percentage(mt) * 100;

        for (auto rank = 0; rank < golden_number_ranks; rank++) {
            const auto part = std::make_shared<Partition>(golden_number_ranks, rank);
            ASSERT_THROW(SubdomainFromNeuronDensity sfnd(rank, golden_fraction_excitatory_neurons, golden_um_per_neuron, part), RelearnException);
        }
    }
}

TEST_F(NeuronAssignmentTest, test_density_constructor_single_subdomain) {
    for (auto i = 0; i < iterations; i++) {
        const auto golden_number_neurons = get_random_number_neurons(mt);
        const auto golden_fraction_excitatory_neurons = get_random_percentage(mt);
        const auto golden_um_per_neuron = get_random_percentage(mt) * 100;

        const auto part = std::make_shared<Partition>(1, 0);
        SubdomainFromNeuronDensity sfnd{ golden_number_neurons, golden_fraction_excitatory_neurons, golden_um_per_neuron, part };

        const auto number_neurons = sfnd.get_requested_number_neurons();
        const auto fraction_excitatory_neurons = sfnd.get_requested_ratio_excitatory_neurons();

        ASSERT_EQ(golden_number_neurons, number_neurons);

        ASSERT_NEAR(golden_fraction_excitatory_neurons, fraction_excitatory_neurons, 1.0 / golden_number_neurons);

        const auto& [sim_box_min, sim_box_max] = part->get_simulation_box_size();
        const auto box_length = (sim_box_max - sim_box_min).get_maximum();

        const auto golden_box_length = calculate_box_length(golden_number_neurons, golden_um_per_neuron);
        ASSERT_NEAR(box_length, golden_box_length, 1.0 / golden_number_neurons);

        ASSERT_EQ(0, sfnd.get_number_placed_neurons());
        ASSERT_EQ(0, sfnd.get_ratio_placed_excitatory_neurons());
    }
}

TEST_F(NeuronAssignmentTest, test_density_constructor_multiple_subdomains) {
    for (auto i = 0; i < iterations; i++) {
        const auto golden_number_ranks = get_adjusted_random_number_ranks(mt);
        const auto number_subdomains = round_to_next_exponent(golden_number_ranks, 8);
        const auto golden_number_neurons = get_random_number_neurons(mt) + number_subdomains;
        const auto golden_fraction_excitatory_neurons = get_random_percentage(mt);
        const auto golden_um_per_neuron = get_random_percentage(mt) * 100;

        for (auto rank = 0; rank < golden_number_ranks; rank++) {
            const auto part = std::make_shared<Partition>(golden_number_ranks, rank);
            SubdomainFromNeuronDensity sfnd{ golden_number_neurons, golden_fraction_excitatory_neurons, golden_um_per_neuron, part };

            const auto number_neurons = sfnd.get_requested_number_neurons();
            const auto fraction_excitatory_neurons = sfnd.get_requested_ratio_excitatory_neurons();

            ASSERT_EQ(golden_number_neurons, number_neurons);

            ASSERT_NEAR(golden_fraction_excitatory_neurons, fraction_excitatory_neurons, 1.0 / golden_number_neurons);

            const auto& [sim_box_min, sim_box_max] = part->get_simulation_box_size();
            const auto box_length = (sim_box_max - sim_box_min).get_maximum();

            const auto golden_box_length = calculate_box_length(golden_number_neurons, golden_um_per_neuron);
            ASSERT_NEAR(box_length, golden_box_length, 1.0 / golden_number_neurons);

            ASSERT_EQ(0, sfnd.get_number_placed_neurons());
            ASSERT_EQ(0, sfnd.get_ratio_placed_excitatory_neurons());
        }
    }
}

TEST_F(NeuronAssignmentTest, test_density_initialize_single_subdomain) {
    for (auto i = 0; i < iterations; i++) {
        const auto golden_number_neurons = get_random_number_neurons(mt);
        const auto golden_fraction_excitatory_neurons = get_random_percentage(mt);
        const auto golden_um_per_neuron = get_random_percentage(mt) * 100;

        const auto lower_bound_ex = static_cast<size_t>(floor(golden_number_neurons * golden_fraction_excitatory_neurons));
        const auto upper_bound_ex = static_cast<size_t>(ceil(golden_number_neurons * golden_fraction_excitatory_neurons));

        const auto part = std::make_shared<Partition>(1, 0);
        SubdomainFromNeuronDensity sfnd{ golden_number_neurons, golden_fraction_excitatory_neurons, golden_um_per_neuron, part };

        const auto& [sim_box_min, sim_box_max] = part->get_simulation_box_size();
        const auto box_length = (sim_box_max - sim_box_min).get_maximum();

        const auto golden_box_length = calculate_box_length(golden_number_neurons, golden_um_per_neuron);
        ASSERT_NEAR(box_length, golden_box_length, 1.0 / golden_number_neurons);

        sfnd.initialize();

        const auto requested_number_neurons = sfnd.get_requested_number_neurons();
        const auto placed_number_neurons = sfnd.get_number_placed_neurons();

        const auto requested_fraction_excitatory_neurons = sfnd.get_requested_ratio_excitatory_neurons();
        const auto placed_fraction_excitatory_neurons = sfnd.get_ratio_placed_excitatory_neurons();

        ASSERT_EQ(golden_number_neurons, requested_number_neurons);

        ASSERT_NEAR(golden_fraction_excitatory_neurons, requested_fraction_excitatory_neurons, 1.0 / golden_number_neurons);

        ASSERT_EQ(requested_number_neurons, placed_number_neurons);
        ASSERT_NEAR(requested_fraction_excitatory_neurons, placed_fraction_excitatory_neurons, 1.0 / golden_number_neurons);

        ASSERT_LE(requested_fraction_excitatory_neurons, placed_fraction_excitatory_neurons);
    }
}

TEST_F(NeuronAssignmentTest, test_density_initialize_multiple_subdomains) {
    for (auto i = 0; i < iterations; i++) {
        const auto golden_number_ranks = get_adjusted_random_number_ranks(mt);
        const auto number_subdomains = round_to_next_exponent(golden_number_ranks, 8);
        const auto golden_number_neurons = get_random_number_neurons(mt) + number_subdomains;
        const auto golden_fraction_excitatory_neurons = get_random_percentage(mt);
        const auto golden_um_per_neuron = get_random_percentage(mt) * 100;

        auto accumulated_placed_neurons = 0;
        auto accumulated_ratio_excitatory_neurons = 0.0;

        for (auto rank = 0; rank < golden_number_ranks; rank++) {
            const auto part = std::make_shared<Partition>(golden_number_ranks, rank);
            SubdomainFromNeuronDensity sfnd{ golden_number_neurons, golden_fraction_excitatory_neurons, golden_um_per_neuron, part };

            sfnd.initialize();

            const auto placed_number_neurons = sfnd.get_number_placed_neurons();
            const auto placed_ratio_excitatory_neurons = sfnd.get_ratio_placed_excitatory_neurons();

            accumulated_placed_neurons += placed_number_neurons;
            accumulated_ratio_excitatory_neurons += placed_number_neurons * placed_ratio_excitatory_neurons;
        }

        const auto actual_ratio_excitatory_neurons = accumulated_ratio_excitatory_neurons / accumulated_placed_neurons;

        const auto difference_neurons = (accumulated_placed_neurons > golden_number_neurons) ? accumulated_placed_neurons - golden_number_neurons : golden_number_neurons - accumulated_placed_neurons;

        ASSERT_LE(difference_neurons, number_subdomains);

        ASSERT_NEAR(golden_fraction_excitatory_neurons, actual_ratio_excitatory_neurons, golden_number_ranks / golden_um_per_neuron);
    }
}

TEST_F(NeuronAssignmentTest, test_density_neuron_attributes_size_single_subdomain) {
    for (auto i = 0; i < iterations; i++) {
        const auto golden_number_neurons = get_random_number_neurons(mt);
        const auto golden_fraction_excitatory_neurons = get_random_percentage(mt);
        const auto golden_um_per_neuron = get_random_percentage(mt) * 100;

        const auto part = std::make_shared<Partition>(1, 0);
        SubdomainFromNeuronDensity sfnd{ golden_number_neurons, golden_fraction_excitatory_neurons, golden_um_per_neuron, part };

        sfnd.initialize();

        const auto placed_number_neurons = sfnd.get_number_placed_neurons();
        const auto placed_fraction_excitatory_neurons = sfnd.get_ratio_placed_excitatory_neurons();

        const auto& positions = sfnd.get_neuron_positions_in_subdomain(0, 1);
        const auto& types = sfnd.get_neuron_types_in_subdomain(0, 1);
        const auto& area_names = sfnd.get_neuron_area_names_in_subdomain(0, 1);
        const auto placed_number_neurons_in_subdomain = sfnd.get_number_neurons_in_subdomain(0, 1);

        ASSERT_EQ(placed_number_neurons, placed_number_neurons_in_subdomain);
        ASSERT_EQ(placed_number_neurons, positions.size());
        ASSERT_EQ(placed_number_neurons, types.size());
        ASSERT_EQ(placed_number_neurons, area_names.size());

        const auto& all_positions = sfnd.get_neuron_positions_in_subdomains(0, 0, 1);
        const auto& all_types = sfnd.get_neuron_types_in_subdomains(0, 0, 1);
        const auto& all_area_names = sfnd.get_neuron_area_names_in_subdomains(0, 0, 1);
        const auto all_placed_neurons_in_subdomains = sfnd.get_number_neurons_in_subdomains(0, 0, 1);

        ASSERT_EQ(placed_number_neurons, all_placed_neurons_in_subdomains);
        ASSERT_EQ(placed_number_neurons, all_positions.size());
        ASSERT_EQ(placed_number_neurons, all_types.size());
        ASSERT_EQ(placed_number_neurons, all_area_names.size());
    }
}

TEST_F(NeuronAssignmentTest, test_density_neuron_attributes_size_multiple_subdomains) {
    for (auto i = 0; i < iterations; i++) {
        const auto golden_number_ranks = get_adjusted_random_number_ranks(mt);
        const auto number_subdomains = round_to_next_exponent(golden_number_ranks, 8);
        const auto number_subdomains_per_rank = number_subdomains / golden_number_ranks;
        const auto golden_number_neurons = get_random_number_neurons(mt) + number_subdomains;
        const auto golden_fraction_excitatory_neurons = get_random_percentage(mt);
        const auto golden_um_per_neuron = get_random_percentage(mt) * 100;

        auto accumulated_placed_neurons = 0;

        for (auto rank = 0; rank < golden_number_ranks; rank++) {
            const auto part = std::make_shared<Partition>(golden_number_ranks, rank);
            SubdomainFromNeuronDensity sfnd{ golden_number_neurons, golden_fraction_excitatory_neurons, golden_um_per_neuron, part };

            sfnd.initialize();

            const auto placed_number_neurons = sfnd.get_number_placed_neurons();
            accumulated_placed_neurons += placed_number_neurons;

            const auto& all_positions = sfnd.get_neuron_positions_in_subdomains(number_subdomains_per_rank * rank, number_subdomains_per_rank * rank + number_subdomains_per_rank - 1, number_subdomains);
            const auto& all_types = sfnd.get_neuron_types_in_subdomains(number_subdomains_per_rank * rank, number_subdomains_per_rank * rank + number_subdomains_per_rank - 1, number_subdomains);
            const auto& all_area_names = sfnd.get_neuron_area_names_in_subdomains(number_subdomains_per_rank * rank, number_subdomains_per_rank * rank + number_subdomains_per_rank - 1, number_subdomains);
            const auto all_placed_neurons_in_subdomains = sfnd.get_number_neurons_in_subdomains(number_subdomains_per_rank * rank, number_subdomains_per_rank * rank + number_subdomains_per_rank - 1, number_subdomains);

            ASSERT_EQ(placed_number_neurons, all_placed_neurons_in_subdomains);
            ASSERT_EQ(placed_number_neurons, all_positions.size());
            ASSERT_EQ(placed_number_neurons, all_types.size());
            ASSERT_EQ(placed_number_neurons, all_area_names.size());

            auto counter = 0;

            for (auto subdomain_id = 0; subdomain_id < number_subdomains_per_rank; subdomain_id++) {
                const auto& positions = sfnd.get_neuron_positions_in_subdomain(number_subdomains_per_rank * rank + subdomain_id, number_subdomains);
                const auto& types = sfnd.get_neuron_types_in_subdomain(number_subdomains_per_rank * rank + subdomain_id, number_subdomains);
                const auto& area_names = sfnd.get_neuron_area_names_in_subdomain(number_subdomains_per_rank * rank + subdomain_id, number_subdomains);
                const auto placed_neurons_in_subdomain = sfnd.get_number_neurons_in_subdomain(number_subdomains_per_rank * rank + subdomain_id, number_subdomains);

                ASSERT_EQ(placed_neurons_in_subdomain, positions.size());
                ASSERT_EQ(placed_neurons_in_subdomain, types.size());
                ASSERT_EQ(placed_neurons_in_subdomain, area_names.size());

                counter += placed_neurons_in_subdomain;
            }

            ASSERT_EQ(counter, placed_number_neurons);
        }

        const auto difference_neurons = (accumulated_placed_neurons > golden_number_neurons) ? accumulated_placed_neurons - golden_number_neurons : golden_number_neurons - accumulated_placed_neurons;

        ASSERT_LE(difference_neurons, number_subdomains);
    }
}

TEST_F(NeuronAssignmentTest, test_density_neuron_attributes_semantic_single_subdomain) {
    for (auto i = 0; i < iterations; i++) {
        const auto golden_number_neurons = get_random_number_neurons(mt);
        const auto golden_fraction_excitatory_neurons = get_random_percentage(mt);
        const auto golden_um_per_neuron = get_random_percentage(mt) * 100;

        const auto part = std::make_shared<Partition>(1, 0);
        SubdomainFromNeuronDensity sfnd{ golden_number_neurons, golden_fraction_excitatory_neurons, golden_um_per_neuron, part };

        sfnd.initialize();

        const auto placed_number_neurons_in_subdomain = sfnd.get_number_placed_neurons();
        const auto placed_ratio_excitatory_neurons = sfnd.get_ratio_placed_excitatory_neurons();

        const auto& positions = sfnd.get_neuron_positions_in_subdomain(0, 1);
        const auto& types = sfnd.get_neuron_types_in_subdomain(0, 1);
        const auto& area_names = sfnd.get_neuron_area_names_in_subdomain(0, 1);

        const auto calculated_ratio_excitatory_neurons = calculate_excitatory_fraction(types);
        ASSERT_NEAR(placed_ratio_excitatory_neurons, calculated_ratio_excitatory_neurons, 1 / golden_number_neurons);

        const auto& [sim_box_min, sim_box_max] = part->get_simulation_box_size();
        const auto neurons_per_dimension = pow(golden_number_neurons, 1. / 3);
        const auto number_boxes = static_cast<size_t>(ceil(neurons_per_dimension));

        std::vector<bool> box_full(number_boxes * number_boxes * number_boxes, false);

        for (const auto& position : positions) {
            ASSERT_TRUE(position.check_in_box(sim_box_min, sim_box_max));
            Vec3s cast_position = (Vec3s)(position / golden_um_per_neuron);

            const auto x = cast_position.get_x();
            const auto y = cast_position.get_y();
            const auto z = cast_position.get_z();

            ASSERT_LE(x, number_boxes);
            ASSERT_LE(y, number_boxes);
            ASSERT_LE(z, number_boxes);

            const auto flag = box_full[z * number_boxes * number_boxes + y * number_boxes + x];

            ASSERT_FALSE(flag);
            box_full[z * number_boxes * number_boxes + y * number_boxes + x] = true;
        }
    }
}

TEST_F(NeuronAssignmentTest, test_density_neuron_attributes_semantic_multiple_subdomains) {
    for (auto i = 0; i < iterations; i++) {
        const auto golden_number_ranks = get_adjusted_random_number_ranks(mt);
        const auto number_subdomains = round_to_next_exponent(golden_number_ranks, 8);
        const auto number_subdomains_per_rank = number_subdomains / golden_number_ranks;
        const auto golden_number_neurons = get_random_number_neurons(mt) + number_subdomains;
        const auto golden_fraction_excitatory_neurons = get_random_percentage(mt);
        const auto golden_um_per_neuron = get_random_percentage(mt) * 100;

        auto accumulated_placed_neurons = 0;

        for (auto rank = 0; rank < golden_number_ranks; rank++) {
            const auto part = std::make_shared<Partition>(golden_number_ranks, rank);
            SubdomainFromNeuronDensity sfnd{ golden_number_neurons, golden_fraction_excitatory_neurons, golden_um_per_neuron, part };

            sfnd.initialize();

            const auto placed_number_neurons = sfnd.get_number_placed_neurons();
            const auto placed_ratio_excitatory_neurons = sfnd.get_ratio_placed_excitatory_neurons();

            accumulated_placed_neurons += placed_number_neurons;

            const auto& all_positions = sfnd.get_neuron_positions_in_subdomains(number_subdomains_per_rank * rank, number_subdomains_per_rank * rank + number_subdomains_per_rank - 1, number_subdomains);
            const auto& all_types = sfnd.get_neuron_types_in_subdomains(number_subdomains_per_rank * rank, number_subdomains_per_rank * rank + number_subdomains_per_rank - 1, number_subdomains);
            const auto all_placed_neurons_in_subdomains = sfnd.get_number_neurons_in_subdomains(number_subdomains_per_rank * rank, number_subdomains_per_rank * rank + number_subdomains_per_rank - 1, number_subdomains);

            const auto calculated_ratio_excitatory_neurons = calculate_excitatory_fraction(all_types);
            ASSERT_NEAR(placed_ratio_excitatory_neurons, calculated_ratio_excitatory_neurons, number_subdomains_per_rank / golden_number_neurons);

            for (auto subdomain_id = 0; subdomain_id < number_subdomains_per_rank; subdomain_id++) {
                const auto& positions = sfnd.get_neuron_positions_in_subdomain(number_subdomains_per_rank * rank + subdomain_id, number_subdomains);
                const auto& types = sfnd.get_neuron_types_in_subdomain(number_subdomains_per_rank * rank + subdomain_id, number_subdomains);
                const auto placed_neurons_in_subdomain = sfnd.get_number_neurons_in_subdomain(number_subdomains_per_rank * rank + subdomain_id, number_subdomains);

                const auto calculated_ratio_excitatory_neurons_subdomain = calculate_excitatory_fraction(types);
                ASSERT_NEAR(placed_ratio_excitatory_neurons, calculated_ratio_excitatory_neurons_subdomain, static_cast<double>(number_subdomains) / golden_number_neurons);

                const auto& [subdomain_min, subdomain_max] = part->get_subdomain_boundaries(subdomain_id);

                const auto& subdomain_length = subdomain_max - subdomain_min;
                const auto& boxes_in_subdomain = subdomain_length / golden_um_per_neuron;

                const auto number_boxes = static_cast<size_t>(ceil(boxes_in_subdomain.calculate_p_norm(1.0)));

                std::vector<bool> box_full(number_boxes * number_boxes * number_boxes, false);

                for (const auto& position : positions) {
                    ASSERT_TRUE(position.check_in_box(subdomain_min, subdomain_max));
                    Vec3s cast_position = (Vec3s)((position - subdomain_min) / golden_um_per_neuron);

                    const auto x = cast_position.get_x();
                    const auto y = cast_position.get_y();
                    const auto z = cast_position.get_z();

                    ASSERT_LE(x, number_boxes);
                    ASSERT_LE(y, number_boxes);
                    ASSERT_LE(z, number_boxes);

                    const auto flag = box_full[z * number_boxes * number_boxes + y * number_boxes + x];

                    ASSERT_FALSE(flag);
                    box_full[z * number_boxes * number_boxes + y * number_boxes + x] = true;
                }
            }
        }
    }
}

TEST_F(NeuronAssignmentTest, test_lazily_fill_multiple) {
    std::uniform_int_distribution<size_t> uid(1, 10000);
    std::uniform_int_distribution<size_t> uid_fills(1, 10);
    std::uniform_real_distribution<double> urd(0.0, 1.0);

    mt.seed(rand());

    for (auto i = 0; i < iterations; i++) {
        auto number_neurons = uid(mt);
        auto frac_ex = urd(mt);
        auto um_per_neuron = urd(mt) * 100;

        auto lower_bound_ex = static_cast<size_t>(floor(number_neurons * frac_ex));
        auto upper_bound_ex = static_cast<size_t>(ceil(number_neurons * frac_ex));

        auto part = std::make_shared<Partition>(1, 0);
        SubdomainFromNeuronDensity sfnd{ number_neurons, frac_ex, um_per_neuron, part };

        const auto& [sim_box_min, sim_box_max] = part->get_simulation_box_size();
        auto box_length = (sim_box_max - sim_box_min).get_maximum();

        ASSERT_GE(box_length, 0);

        auto num_fills = uid_fills(mt);

        for (auto j = 0; j < num_fills; j++) {
            if (j >= 1) {
                ASSERT_THROW(sfnd.initialize(), RelearnException);
            } else {
                sfnd.initialize();
            }
        }

        auto num_neurons_ = sfnd.get_requested_number_neurons();
        auto frac_ex_ = sfnd.get_requested_ratio_excitatory_neurons();

        ASSERT_EQ(number_neurons, num_neurons_);

        ASSERT_NEAR(frac_ex, frac_ex_, 1.0 / number_neurons);

        ASSERT_EQ(sfnd.get_requested_number_neurons(), sfnd.get_number_placed_neurons());
        ASSERT_NEAR(sfnd.get_requested_ratio_excitatory_neurons(), sfnd.get_ratio_placed_excitatory_neurons(), 1.0 / number_neurons);

        ASSERT_LE(sfnd.get_requested_ratio_excitatory_neurons(), sfnd.get_ratio_placed_excitatory_neurons());
    }
}

TEST_F(NeuronAssignmentTest, test_lazily_fill_positions) {
    std::uniform_int_distribution<size_t> uid(1, 10000);
    std::uniform_real_distribution<double> urd(0.0, 1.0);

    mt.seed(rand());

    for (auto i = 0; i < iterations; i++) {
        auto number_neurons = uid(mt);
        auto frac_ex = urd(mt);
        auto um_per_neuron = urd(mt) * 100;

        auto expected_neurons_per_dimension = static_cast<size_t>(ceil(pow(number_neurons, 1.0 / 3.0)));
        auto size = expected_neurons_per_dimension * expected_neurons_per_dimension * expected_neurons_per_dimension;

        std::vector<bool> flags(size, false);

        auto part = std::make_shared<Partition>(1, 0);
        SubdomainFromNeuronDensity sfnd{ number_neurons, frac_ex, um_per_neuron, part };

        const auto& [sim_box_min, sim_box_max] = part->get_simulation_box_size();
        auto box_length = (sim_box_max - sim_box_min).get_maximum();

        sfnd.initialize();

        std::vector<Vec3d> pos = sfnd.get_neuron_positions_in_subdomain(0, 1);

        ASSERT_EQ(pos.size(), number_neurons);

        std::vector<Vec3<size_t>> pos_fixed;
        for (auto& p : pos) {
            pos_fixed.emplace_back((Vec3<size_t>)(p * (1 / um_per_neuron)));
        }

        for (auto& p : pos_fixed) {
            auto x = p.get_x();
            auto y = p.get_y();
            auto z = p.get_z();

            ASSERT_LE(x, expected_neurons_per_dimension);
            ASSERT_LE(y, expected_neurons_per_dimension);
            ASSERT_LE(z, expected_neurons_per_dimension);

            auto idx = x * expected_neurons_per_dimension * expected_neurons_per_dimension + y * expected_neurons_per_dimension + z;
            ASSERT_FALSE(flags[idx]);
            flags[idx] = true;
        }

        std::vector<SignalType> types = sfnd.get_neuron_types_in_subdomain(0, 1);

        size_t neurons_ex = 0;
        size_t neurons_in = 0;

        for (auto& type : types) {
            if (type == SignalType::EXCITATORY) {
                neurons_ex++;
            } else if (type == SignalType::INHIBITORY) {
                neurons_in++;
            }
        }

        auto frac_ex_ = (static_cast<double>(neurons_ex) / static_cast<double>(neurons_ex + neurons_in));
        ASSERT_NEAR(frac_ex, frac_ex_, 1.0 / static_cast<double>(number_neurons));
    }
}

TEST_F(NeuronAssignmentTest, test_lazily_fill_positions_multiple_subdomains) {
    std::uniform_int_distribution<size_t> uid(1, 10000);
    std::uniform_real_distribution<double> urd(0.0, 1.0);

    mt.seed(rand());

    std::uniform_int_distribution<size_t> uid_subdomains(1, 5);

    for (auto i = 0; i < iterations; i++) {
        auto number_neurons = uid(mt);
        auto frac_ex = urd(mt);
        auto um_per_neuron = urd(mt) * 100;

        auto subdomains_x = uid_subdomains(mt);
        auto subdomains_y = uid_subdomains(mt);
        auto subdomains_z = uid_subdomains(mt);

        auto total_subdomains = subdomains_x * subdomains_y * subdomains_z;

        auto expected_neurons_per_dimension = static_cast<size_t>(ceil(pow(number_neurons, 1.0 / 3.0)));
        auto size = expected_neurons_per_dimension * expected_neurons_per_dimension * expected_neurons_per_dimension;

        std::vector<bool> flags(size, false);

        auto part = std::make_shared<Partition>(1, 0);
        SubdomainFromNeuronDensity sfnd{ number_neurons, frac_ex, um_per_neuron, part };

        const auto& [sim_box_min, sim_box_max] = part->get_simulation_box_size();
        auto box_length = (sim_box_max - sim_box_min).get_maximum();

        auto subdomain_length_x = box_length / subdomains_x;
        auto subdomain_length_y = box_length / subdomains_y;
        auto subdomain_length_z = box_length / subdomains_z;

        sfnd.initialize();

        std::vector<Vec3d> pos;
        for (auto x_it = 0; x_it < subdomains_x; x_it++) {
            for (auto y_it = 0; y_it < subdomains_y; y_it++) {
                for (auto z_it = 0; z_it < subdomains_z; z_it++) {
                    Vec3d subdomain_min{ x_it * subdomain_length_x, y_it * subdomain_length_y, z_it * subdomain_length_z };
                    Vec3d subdomain_max{ (x_it + 1) * subdomain_length_x, (y_it + 1) * subdomain_length_y, (z_it + 1) * subdomain_length_z };

                    auto current_idx = z_it + y_it * subdomains_z + x_it * subdomains_z * subdomains_y;

                    if (x_it == 0 && y_it == 0 && z_it == 0) {
                        auto vec = sfnd.get_neuron_positions_in_subdomain(current_idx, total_subdomains);
                        pos.insert(pos.end(), vec.begin(), vec.end());
                    } else {
                        ASSERT_THROW(auto vec = sfnd.get_neuron_positions_in_subdomain(current_idx, total_subdomains), RelearnException);
                    }
                }
            }
        }

        ASSERT_EQ(pos.size(), number_neurons);

        std::vector<Vec3<size_t>> pos_fixed;
        for (auto& p : pos) {
            pos_fixed.emplace_back((Vec3<size_t>)(p * (1 / um_per_neuron)));
        }

        for (auto& p : pos_fixed) {
            auto x = p.get_x();
            auto y = p.get_y();
            auto z = p.get_z();

            ASSERT_LE(x, expected_neurons_per_dimension);
            ASSERT_LE(y, expected_neurons_per_dimension);
            ASSERT_LE(z, expected_neurons_per_dimension);

            auto idx = x * expected_neurons_per_dimension * expected_neurons_per_dimension + y * expected_neurons_per_dimension + z;
            ASSERT_FALSE(flags[idx]);
            flags[idx] = true;
        }

        std::vector<SignalType> types = sfnd.get_neuron_types_in_subdomain(0, 1);

        size_t neurons_ex = 0;
        size_t neurons_in = 0;

        for (auto& type : types) {
            if (type == SignalType::EXCITATORY) {
                neurons_ex++;
            } else if (type == SignalType::INHIBITORY) {
                neurons_in++;
            }
        }

        auto frac_ex_ = (static_cast<double>(neurons_ex) / static_cast<double>(neurons_ex + neurons_in));
        ASSERT_NEAR(frac_ex, frac_ex_, static_cast<double>(total_subdomains) / static_cast<double>(number_neurons));
    }
}

TEST_F(NeuronAssignmentTest, test_saving) {
    std::uniform_int_distribution<size_t> uid(1, 1000);
    std::uniform_real_distribution<double> urd(0.0, 1.0);

    mt.seed(rand());

    for (auto i = 0; i < iterations; i++) {
        std::vector<Vec3d> positions;
        std::vector<std::string> area_names;
        std::vector<SignalType> types;

        generate_neuron_positions(positions, area_names, types, mt);

        auto number_neurons = positions.size();

        std::ifstream file("neurons.tmp", std::ios::binary | std::ios::in);

        std::vector<std::string> lines;

        std::string str;
        while (std::getline(file, str)) {
            if (str[0] == '#') {
                continue;
            }

            lines.emplace_back(str);
        }

        file.close();

        ASSERT_EQ(number_neurons, lines.size());

        std::vector<bool> is_there(number_neurons);

        for (auto j = 0; j < number_neurons; j++) {
            const Vec3d& desired_position = positions[j];
            const std::string& desired_area_name = area_names[j];
            const SignalType& desired_signal_type = types[j];

            const std::string& current_line = lines[j];

            std::stringstream sstream(current_line);

            size_t id;
            double x;
            double y;
            double z;
            std::string area;
            std::string type_string;

            sstream
                >> id
                >> x
                >> y
                >> z
                >> area
                >> type_string;

            ASSERT_TRUE(0 < id);
            ASSERT_TRUE(id <= number_neurons);

            ASSERT_FALSE(is_there[id - 1]);
            is_there[id - 1] = true;

            ASSERT_NEAR(x, desired_position.get_x(), eps);
            ASSERT_NEAR(y, desired_position.get_y(), eps);
            ASSERT_NEAR(z, desired_position.get_z(), eps);

            SignalType type;
            if (type_string == "ex") {
                type = SignalType::EXCITATORY;
            } else if (type_string == "in") {
                type = SignalType::INHIBITORY;
            } else {
                ASSERT_TRUE(false);
            }

            ASSERT_TRUE(area == desired_area_name);
            ASSERT_TRUE(type == desired_signal_type);
        }
    }
}

TEST_F(NeuronAssignmentTest, test_reloading) {
    std::uniform_int_distribution<size_t> uid(1, 1000);
    std::uniform_real_distribution<double> urd(0.0, 1.0);

    mt.seed(rand());

    auto part = std::make_shared<Partition>(1, 0);

    for (auto i = 0; i < iterations; i++) {
        std::vector<Vec3d> positions;
        std::vector<std::string> area_names;
        std::vector<SignalType> types;

        generate_neuron_positions(positions, area_names, types, mt);

        auto number_neurons = positions.size();

        SubdomainFromFile sff{ "neurons.tmp", {}, part };
        part->set_total_number_neurons(sff.get_total_number_neurons_in_file());

        const auto& [sim_box_min, sim_box_max] = part->get_simulation_box_size();
        auto box_length = (sim_box_max - sim_box_min).get_maximum();

        sff.initialize();

        std::vector<Vec3d> loaded_positions = sff.get_neuron_positions_in_subdomain(0, 1);
        std::vector<std::string> loaded_area_names = sff.get_neuron_area_names_in_subdomain(0, 1);
        std::vector<SignalType> loaded_types = sff.get_neuron_types_in_subdomain(0, 1);

        for (auto j = 0; j < number_neurons; j++) {
            const auto& curr_pos = positions[j];
            const auto& curr_loaded_pos = loaded_positions[j];

            ASSERT_NEAR(curr_pos.get_x(), curr_loaded_pos.get_x(), eps);
            ASSERT_NEAR(curr_pos.get_y(), curr_loaded_pos.get_y(), eps);
            ASSERT_NEAR(curr_pos.get_z(), curr_loaded_pos.get_z(), eps);

            const auto& curr_name = area_names[j];
            const auto& curr_loaded_name = loaded_area_names[j];

            ASSERT_EQ(curr_name, curr_loaded_name);

            const auto& curr_type = types[j];
            const auto& curr_loaded_type = loaded_types[j];

            ASSERT_EQ(curr_type, curr_loaded_type);
        }
    }
}

TEST_F(NeuronAssignmentTest, test_neuron_placement_store_and_load) {
    const std::string file{ "./test_output_positions_id.txt" };

    constexpr auto subdomain_id = 0;
    constexpr auto number_neurons = 10;
    constexpr auto frac_neurons_exc = 0.5;

    auto part = std::make_shared<Partition>(1, 0);
    // create from density
    SubdomainFromNeuronDensity sdnd{ number_neurons, frac_neurons_exc, 26, part };
    // fill_subdomain

    const auto& [sim_box_min, sim_box_max] = part->get_simulation_box_size();
    auto box_length = (sim_box_max - sim_box_min).get_maximum();
    sdnd.initialize();
    // save to file
    sdnd.write_neurons_to_file(file);

    // load from file
    SubdomainFromFile sdff{ file, {}, part };
    part->set_total_number_neurons(sdff.get_total_number_neurons_in_file());
    // fill_subdomain from file
    sdff.initialize();

    // check neuron placement numbers
    ASSERT_EQ(sdff.get_requested_number_neurons(), sdnd.get_requested_number_neurons());
    ASSERT_EQ(sdff.get_number_placed_neurons(), sdnd.get_number_placed_neurons());
    ASSERT_EQ(sdff.get_requested_ratio_excitatory_neurons(), sdnd.get_requested_ratio_excitatory_neurons());
    ASSERT_EQ(sdff.get_ratio_placed_excitatory_neurons(), sdnd.get_ratio_placed_excitatory_neurons());

    // compare the written files of sdnd and sdff
    std::ifstream saved1{ file };
    sdff.write_neurons_to_file("./test_output_positions_id2.txt");
    std::ifstream saved2{ "./test_output_positions_id2.txt" };

    for (std::string line1{}, line2{}; std::getline(saved1, line1) && std::getline(saved2, line2);) {
        ASSERT_EQ(line1, line2);
    }
}
