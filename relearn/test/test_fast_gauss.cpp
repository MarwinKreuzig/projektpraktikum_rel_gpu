#include "../googletest/include/gtest/gtest.h"

#include "RelearnTest.hpp"

#include "../source/algorithm/FastMultipoleMethods.h"
#include "../source/neurons/models/SynapticElements.h"
#include "../source/structure/Cell.h"
#include "../source/structure/Partition.h"
#include "../source/structure/Octree.h"

#include "../source/util/RelearnException.h"
#include "../source/util/Vec3.h"

#include <algorithm>
#include <map>
#include <numeric>
#include <random>
#include <stack>
#include <tuple>

using AdditionalCellAttributes = FastMultipoleMethodsCell;

const static double sigma = 750;

std::tuple<Vec3d, Vec3d> get_random_simulation_box_size_FMM(std::mt19937& mt) {
    std::uniform_real_distribution<double> urd(-10000.0, +10000.0);

    const auto rand_x_1 = urd(mt);
    const auto rand_x_2 = urd(mt);

    const auto rand_y_1 = urd(mt);
    const auto rand_y_2 = urd(mt);

    const auto rand_z_1 = urd(mt);
    const auto rand_z_2 = urd(mt);

    return {
        { std::min(rand_x_1, rand_x_2), std::min(rand_y_1, rand_y_2), std::min(rand_z_1, rand_z_2) },
        { std::max(rand_x_1, rand_x_2), std::max(rand_y_1, rand_y_2), std::max(rand_z_1, rand_z_2) }
    };
}

std::vector<std::tuple<Vec3d, size_t>> generate_random_neurons_FMM(const Vec3d& min, const Vec3d& max, size_t count, size_t max_id, std::mt19937& mt) {
    std::uniform_real_distribution<double> urd_x(min.get_x(), max.get_x());
    std::uniform_real_distribution<double> urd_y(min.get_y(), max.get_y());
    std::uniform_real_distribution<double> urd_z(min.get_z(), max.get_z());

    std::vector<size_t> ids(max_id);
    std::iota(ids.begin(), ids.end(), 0);
    std::shuffle(ids.begin(), ids.end(), mt);

    std::vector<std::tuple<Vec3d, size_t>> return_value(count);

    for (auto i = 0; i < count; i++) {
        const auto rand_x = urd_x(mt);
        const auto rand_y = urd_y(mt);
        const auto rand_z = urd_z(mt);

        return_value[i] = { { rand_x, rand_y, rand_z }, ids[i] };
    }

    return return_value;
}

std::vector<std::tuple<Vec3d, size_t>> extract_neurons_FMM(OctreeNode<AdditionalCellAttributes>* root) {
    std::vector<std::tuple<Vec3d, size_t>> return_value;

    std::stack<OctreeNode<AdditionalCellAttributes>*> octree_nodes{};
    octree_nodes.push(root);

    while (!octree_nodes.empty()) {
        OctreeNode<AdditionalCellAttributes>* current_node = octree_nodes.top();
        octree_nodes.pop();

        if (current_node->is_parent()) {
            const auto childs = current_node->get_children();
            for (auto i = 0; i < 8; i++) {
                const auto child = childs[i];
                if (child != nullptr) {
                    octree_nodes.push(child);
                }
            }
        } else {
            const auto& cell = current_node->get_cell();
            const auto neuron_id = cell.get_neuron_id();
            const auto& opt_position = cell.get_dendrites_position();

            EXPECT_TRUE(opt_position.has_value());

            const auto position = opt_position.value();

            if (neuron_id < Constants::uninitialized) {
                return_value.emplace_back(position, neuron_id);
            }
        }
    }

    return return_value;
}

SynapticElements create_synaptic_elements_FMM(size_t size, std::mt19937& mt, double max_free, SignalType st) {
    SynapticElements se(ElementType::DENDRITE, 0.0);

    se.init(size);

    std::uniform_real_distribution<double> urd(0, max_free);

    for (auto i = 0; i < size; i++) {
        se.set_signal_type(i, st);
        se.update_count(i, urd(mt));
    }

    return se;
}

TEST(TestFastGauss, test_static_functions) {
    // function_derivative test
    double result[] = {
        0.74, 0, -0.74,
        0.74, -2, 0.74,
        -1.47, 0, 1.47,
        -7.36, 12, -7.36,
        -2.94, 0, 2.94,
        67.69, -120, 67.69,
        170.7, 0, -170.7,
        -606.27, 1680, -606.27,
        -3943.67, 0, 3943.67,
        3025.44, -30240, 3025.44,
        84924.23, 0, -84924.23,
        103288.77, 665280, 103288.77,
        -1831604.05, 0, 1831604.05,
        -6348716.24, -17297280, -6348716.24,
        38587480.85, 0, -38587480.85,
        267636449.02, 518918400, 267636449.02
    };

    for (int i = 0; i < 16; i++) {
        const auto res0 = result[(i * 3) + 0];
        const auto res1 = result[(i * 3) + 1];
        const auto res2 = result[(i * 3) + 2];

        const auto val_new0 = FastMultipoleMethods::function_derivative(-1, i + 1);
        const auto val_new1 = FastMultipoleMethods::function_derivative(0, i + 1);
        const auto val_new2 = FastMultipoleMethods::function_derivative(1, i + 1);

        EXPECT_NEAR(val_new0, res0, 0.01) << i;
        EXPECT_NEAR(val_new1, res1, 0.01) << i;
        EXPECT_NEAR(val_new2, res2, 0.01) << i;
    }

    // kernel test
    Vec3d a = { 0, 0, 0 };
    Vec3d b = { 0, 1, 0 };
    EXPECT_NEAR(FastMultipoleMethods::kernel(a, b, sigma), 0.999956, 0.0001);
    Vec3d c = { 0, 0, -1 };
    EXPECT_EQ(FastMultipoleMethods::kernel(a, a, sigma), 1);
    Vec3d e = { 6, 4.5, -3.4 };
    Vec3d f = { 0, -8.3, 2 };
    EXPECT_NEAR(FastMultipoleMethods::kernel(e, f, sigma), 0.9898, 0.01);
}

TEST(TestFastGauss, test_multiIndex) {
    EXPECT_EQ(FastMultipoleMethods::Multiindex::get_number_of_indices(), Constants::p3);

    const auto& indices = FastMultipoleMethods::Multiindex::get_indices();

    const std::array<unsigned int, 3> temp = indices[1];
    EXPECT_EQ(temp.at(0), 0);
    EXPECT_EQ(temp.at(1), 0);
    EXPECT_EQ(temp.at(2), 1);

    const std::array<unsigned int, 3> temp1 = indices[Constants::p3-1];
    EXPECT_EQ(temp1.at(0), Constants::p - 1);
    EXPECT_EQ(temp1.at(1), Constants::p - 1);
    EXPECT_EQ(temp1.at(2), Constants::p - 1);
}

TEST(TestFastGauss, test_static_multiindex_functions) {
    const std::array<unsigned int, 3> test_index1 = { 0, 0, 0 };
    const std::array<unsigned int, 3> test_index2 = { 1, 2, 3 };
    const std::array<unsigned int, 3> test_index3 = { 3, 3, 3 };

    // factorial
    EXPECT_EQ(FastMultipoleMethods::fac_multiindex(test_index1), 1);
    EXPECT_EQ(FastMultipoleMethods::fac_multiindex(test_index2), 12);
    EXPECT_EQ(FastMultipoleMethods::fac_multiindex(test_index3), 216);

    // abs
    EXPECT_EQ(FastMultipoleMethods::abs_multiindex(test_index1), 0);
    EXPECT_EQ(FastMultipoleMethods::abs_multiindex(test_index2), 6);
    EXPECT_EQ(FastMultipoleMethods::abs_multiindex(test_index3), 9);

    //pow
    Vec3d test_vector1 = Vec3d(0, 0, 0);
    Vec3d test_vector2 = Vec3d(3.12, 5.7, -3.14);
    Vec3d test_vector3 = Vec3d(-6.98, -4.77, 2.94);

    EXPECT_EQ(FastMultipoleMethods::pow_multiindex(test_vector1, test_index1), 1);
    EXPECT_EQ(FastMultipoleMethods::pow_multiindex(test_vector1, test_index2), 0);
    EXPECT_EQ(FastMultipoleMethods::pow_multiindex(test_vector1, test_index3), 0);

    EXPECT_EQ(FastMultipoleMethods::pow_multiindex(test_vector2, test_index1), 1);
    EXPECT_NEAR(FastMultipoleMethods::pow_multiindex(test_vector2, test_index2), -3138.29, 0.01);
    EXPECT_NEAR(FastMultipoleMethods::pow_multiindex(test_vector2, test_index3), -174131.48, 0.01);

    EXPECT_EQ(FastMultipoleMethods::pow_multiindex(test_vector3, test_index1), 1);
    EXPECT_NEAR(FastMultipoleMethods::pow_multiindex(test_vector3, test_index2), -4035.84, 0.01);
    EXPECT_NEAR(FastMultipoleMethods::pow_multiindex(test_vector3, test_index3), 937914.81, 0.01);
}

TEST_F(OctreeTest, testOctreeUpdateLocalTreesNumberDendritesFMM) {
    make_mpi_mem_available();

    const auto my_rank = MPIWrapper::get_my_rank();

    std::uniform_int_distribution<size_t> uid_lvl(0, 6);
    std::uniform_int_distribution<size_t> uid(0, 10000);
    std::uniform_real_distribution<double> urd_sigma(1, 10000.0);
    std::uniform_real_distribution<double> urd_theta(0.0, 1.0);

    std::uniform_real_distribution<double> uid_max_vacant(1.0, 100.0);

    for (auto i = 0; i < iterations; i++) {
        Vec3d min{};
        Vec3d max{};

        std::tie(min, max) = get_random_simulation_box_size_FMM(mt);

        auto octree_ptr = std::make_shared<OctreeImplementation<FastMultipoleMethods>>(min, max, 0);
        auto& octree = *octree_ptr;

        const size_t num_neurons = uid(mt);

        const std::vector<std::tuple<Vec3d, size_t>> neurons_to_place = generate_random_neurons_FMM(min, max, num_neurons, num_neurons, mt);

        for (const auto& [position, id] : neurons_to_place) {
            octree.insert(position, id, my_rank);
        }

        octree.initializes_leaf_nodes(num_neurons);

        const auto max_vacant_exc = uid_max_vacant(mt);
        auto dends_exc = create_synaptic_elements_FMM(num_neurons, mt, max_vacant_exc, SignalType::EXCITATORY);

        const auto max_vacant_inh = uid_max_vacant(mt);
        auto dends_inh = create_synaptic_elements_FMM(num_neurons, mt, max_vacant_inh, SignalType::INHIBITORY);

        FastMultipoleMethods fmm{ octree_ptr };

        std::vector<char> disable_flags(num_neurons, 1);

        auto unique_exc = std::make_unique<SynapticElements>(std::move(dends_exc));
        auto unique_inh = std::make_unique<SynapticElements>(std::move(dends_inh));

        fmm.update_leaf_nodes(disable_flags, unique_exc, unique_exc, unique_inh);
        octree.update_local_trees();

        std::stack<OctreeNode<AdditionalCellAttributes>*> stack{};
        stack.emplace(octree.get_root());

        while (!stack.empty()) {
            const auto* current = stack.top();
            stack.pop();

            size_t sum_dends_exc = 0;
            size_t sum_dends_inh = 0;

            if (current->is_parent()) {
                for (auto* child : current->get_children()) {
                    if (child == nullptr) {
                        continue;
                    }

                    sum_dends_exc += child->get_cell().get_number_excitatory_dendrites();
                    sum_dends_inh += child->get_cell().get_number_inhibitory_dendrites();

                    stack.emplace(child);
                }
            } else {
                sum_dends_exc = static_cast<size_t>(unique_exc->get_count(current->get_cell_neuron_id()));
                sum_dends_inh = static_cast<size_t>(unique_inh->get_count(current->get_cell_neuron_id()));
            }

            ASSERT_EQ(current->get_cell().get_number_excitatory_dendrites(), sum_dends_exc);
            ASSERT_EQ(current->get_cell().get_number_inhibitory_dendrites(), sum_dends_inh);
        }

        make_mpi_mem_available();
    }
}

TEST_F(OctreeTest, testOctreeUpdateLocalTreesPositionDendritesFMM) {
    make_mpi_mem_available();

    const auto my_rank = MPIWrapper::get_my_rank();

    std::uniform_int_distribution<size_t> uid_lvl(0, 6);
    std::uniform_int_distribution<size_t> uid(0, 10000);
    std::uniform_real_distribution<double> urd_sigma(1, 10000.0);
    std::uniform_real_distribution<double> urd_theta(0.0, 1.0);

    for (auto i = 0; i < iterations; i++) {
        Vec3d min{};
        Vec3d max{};

        std::tie(min, max) = get_random_simulation_box_size_FMM(mt);

        auto octree_ptr = std::make_shared<OctreeImplementation<FastMultipoleMethods>>(min, max, 0);
        auto& octree = *octree_ptr;

        const size_t num_neurons = uid(mt);

        const std::vector<std::tuple<Vec3d, size_t>> neurons_to_place = generate_random_neurons_FMM(min, max, num_neurons, num_neurons, mt);

        for (const auto& [position, id] : neurons_to_place) {
            octree.insert(position, id, my_rank);
        }

        octree.initializes_leaf_nodes(num_neurons);

        auto dends_exc = create_synaptic_elements_FMM(num_neurons, mt, 1, SignalType::EXCITATORY);
        auto dends_inh = create_synaptic_elements_FMM(num_neurons, mt, 1, SignalType::INHIBITORY);

        auto unique_exc = std::make_unique<SynapticElements>(std::move(dends_exc));
        auto unique_inh = std::make_unique<SynapticElements>(std::move(dends_inh));

        FastMultipoleMethods fmm{ octree_ptr };

        std::vector<char> disable_flags(num_neurons, 1);

        fmm.update_leaf_nodes(disable_flags, unique_exc, unique_exc, unique_inh);
        octree.update_local_trees();

        std::stack<std::tuple<OctreeNode<AdditionalCellAttributes>*, bool, bool>> stack{};
        const auto flag_exc = octree.get_root()->get_cell().get_number_excitatory_dendrites() != 0;
        const auto flag_inh = octree.get_root()->get_cell().get_number_inhibitory_dendrites() != 0;
        stack.emplace(octree.get_root(), flag_exc, flag_inh);

        while (!stack.empty()) {
            std::tuple<OctreeNode<AdditionalCellAttributes>*, bool, bool> tup = stack.top();
            stack.pop();

            auto* current = std::get<0>(tup);
            auto has_exc = std::get<1>(tup);
            auto has_inh = std::get<2>(tup);

            Vec3d pos_dends_exc{ 0.0 };
            Vec3d pos_dends_inh{ 0.0 };

            bool changed_exc = false;
            bool changed_inh = false;

            if (current->is_parent()) {
                double num_dends_exc = 0.0;
                double num_dends_inh = 0.0;

                for (auto* child : current->get_children()) {
                    if (child == nullptr) {
                        continue;
                    }

                    const auto& cell = child->get_cell();

                    const auto& opt_exc = cell.get_excitatory_dendrites_position();
                    const auto& opt_inh = cell.get_inhibitory_dendrites_position();

                    if (!has_exc) {
                        ASSERT_EQ(cell.get_number_excitatory_dendrites(), 0);
                    }

                    if (!has_inh) {
                        ASSERT_EQ(cell.get_number_inhibitory_dendrites(), 0);
                    }

                    if (opt_exc.has_value() && cell.get_number_excitatory_dendrites() != 0) {
                        changed_exc = true;
                        pos_dends_exc += (opt_exc.value() * cell.get_number_excitatory_dendrites());
                        num_dends_exc += cell.get_number_excitatory_dendrites();
                    }

                    if (opt_inh.has_value() && cell.get_number_inhibitory_dendrites() != 0) {
                        changed_inh = true;
                        pos_dends_inh += (opt_inh.value() * cell.get_number_inhibitory_dendrites());
                        num_dends_inh += cell.get_number_inhibitory_dendrites();
                    }

                    stack.emplace(child, cell.get_number_excitatory_dendrites() != 0, cell.get_number_inhibitory_dendrites() != 0);
                }

                pos_dends_exc /= num_dends_exc;
                pos_dends_inh /= num_dends_inh;

            } else {
                const auto& cell = current->get_cell();

                const auto& opt_exc = cell.get_excitatory_dendrites_position();
                const auto& opt_inh = cell.get_inhibitory_dendrites_position();

                if (!has_exc) {
                    ASSERT_EQ(cell.get_number_excitatory_dendrites(), 0);
                }

                if (!has_inh) {
                    ASSERT_EQ(cell.get_number_inhibitory_dendrites(), 0);
                }

                if (opt_exc.has_value() && cell.get_number_excitatory_dendrites() != 0) {
                    changed_exc = true;
                    pos_dends_exc += (opt_exc.value() * cell.get_number_excitatory_dendrites());
                }

                if (opt_inh.has_value() && cell.get_number_inhibitory_dendrites() != 0) {
                    changed_inh = true;
                    pos_dends_inh += (opt_inh.value() * cell.get_number_inhibitory_dendrites());
                }
            }

            ASSERT_EQ(has_exc, changed_exc);
            ASSERT_EQ(has_inh, changed_inh);

            if (has_exc) {
                const auto& diff = current->get_cell().get_excitatory_dendrites_position().value() - pos_dends_exc;
                ASSERT_NEAR(diff.get_x(), 0.0, eps);
                ASSERT_NEAR(diff.get_y(), 0.0, eps);
                ASSERT_NEAR(diff.get_z(), 0.0, eps);
            }

            if (has_inh) {
                const auto& diff = current->get_cell().get_inhibitory_dendrites_position().value() - pos_dends_inh;
                ASSERT_NEAR(diff.get_x(), 0.0, eps);
                ASSERT_NEAR(diff.get_y(), 0.0, eps);
                ASSERT_NEAR(diff.get_z(), 0.0, eps);
            }
        }

        make_mpi_mem_available();
    }
}
