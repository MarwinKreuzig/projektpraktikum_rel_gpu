#include "../googletest/include/gtest/gtest.h"

#include "commons.h"

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
#include <vector>

std::tuple<Vec3d, Vec3d> get_random_simulation_box_size() {
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

std::vector<std::tuple<Vec3d, size_t>> generate_random_neurons(const Vec3d& min, const Vec3d& max, size_t count, size_t max_id) {
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

std::vector<std::tuple<Vec3d, size_t>> extract_neurons(OctreeNode* root) {
    std::vector<std::tuple<Vec3d, size_t>> return_value;

    std::stack<OctreeNode*> octree_nodes{};
    octree_nodes.push(root);

    while (!octree_nodes.empty()) {
        OctreeNode* current_node = octree_nodes.top();
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
            const auto& opt_position = cell.get_neuron_position();

            EXPECT_TRUE(opt_position.has_value());

            const auto position = opt_position.value();

            if (neuron_id < Constants::uninitialized) {
                return_value.emplace_back(position, neuron_id);
            }
        }
    }

    return return_value;
}

std::vector<std::tuple<Vec3d, size_t>> extract_neurons(const Octree& octree) {
    std::vector<std::tuple<Vec3d, size_t>> return_value;

    const auto root = octree.get_root();
    if (root == nullptr) {
        return return_value;
    }

    return extract_neurons(root);
}

std::vector<std::tuple<Vec3d, size_t>> extract_unused_neurons(OctreeNode* root) {
    std::vector<std::tuple<Vec3d, size_t>> return_value;

    std::stack<OctreeNode*> octree_nodes{};
    octree_nodes.push(root);

    while (!octree_nodes.empty()) {
        OctreeNode* current_node = octree_nodes.top();
        octree_nodes.pop();

        if (current_node->get_cell().get_neuron_id() == Constants::uninitialized) {
            return_value.emplace_back(current_node->get_cell().get_neuron_position().value(), current_node->get_level());
        }

        if (current_node->is_parent()) {
            const auto childs = current_node->get_children();
            for (auto i = 0; i < 8; i++) {
                const auto child = childs[i];
                if (child != nullptr) {
                    octree_nodes.push(child);
                }
            }
        }
    }

    return return_value;
}

std::vector<OctreeNode*> extract_branch_nodes(OctreeNode* root) {
    std::vector<OctreeNode*> return_value;

    std::stack<OctreeNode*> octree_nodes{};
    octree_nodes.push(root);

    while (!octree_nodes.empty()) {
        OctreeNode* current_node = octree_nodes.top();
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
            return_value.emplace_back(current_node);
        }
    }

    return return_value;
}

TEST(TestCell, testCellSize) {
    setup();

    for (auto i = 0; i < iterations; i++) {
        Cell<FastMultipoleMethodsCell> cell{};

        const auto& box_sizes_1 = get_random_simulation_box_size();
        const auto& min_1 = std::get<0>(box_sizes_1);
        const auto& max_1 = std::get<1>(box_sizes_1);

        cell.set_size(min_1, max_1);

        const auto& res_1 = cell.get_size();

        ASSERT_EQ(min_1, std::get<0>(res_1));
        ASSERT_EQ(max_1, std::get<1>(res_1));

        const auto& box_sizes_2 = get_random_simulation_box_size();
        const auto& min_2 = std::get<0>(box_sizes_2);
        const auto& max_2 = std::get<1>(box_sizes_2);

        cell.set_size(min_2, max_2);

        const auto& res_2 = cell.get_size();

        ASSERT_EQ(min_2, std::get<0>(res_2));
        ASSERT_EQ(max_2, std::get<1>(res_2));

        ASSERT_EQ(cell.get_maximal_dimension_difference(), (max_2 - min_2).get_maximum());
    }
}

TEST(TestCell, testCellPosition) {
    setup();

    for (auto i = 0; i < iterations; i++) {
        Cell<FastMultipoleMethodsCell> cell{};

        const auto& box_sizes = get_random_simulation_box_size();
        const auto& min = std::get<0>(box_sizes);
        const auto& max = std::get<1>(box_sizes);

        cell.set_size(min, max);

        std::uniform_real_distribution urd_x(min.get_x(), max.get_x());
        std::uniform_real_distribution urd_y(min.get_y(), max.get_y());
        std::uniform_real_distribution urd_z(min.get_z(), max.get_z());

        const Vec3d pos_ex_1{ urd_x(mt), urd_y(mt), urd_z(mt) };
        cell.set_excitatory_dendrites_position(pos_ex_1);

        ASSERT_TRUE(cell.get_excitatory_dendrites_position().has_value());
        ASSERT_EQ(pos_ex_1, cell.get_excitatory_dendrites_position().value());

        ASSERT_TRUE(cell.get_neuron_position_for(SignalType::EXCITATORY, ElementType::DENDRITE).has_value());
        ASSERT_EQ(pos_ex_1, cell.get_neuron_position_for(SignalType::EXCITATORY, ElementType::DENDRITE).value());

        cell.set_excitatory_dendrites_position({});
        ASSERT_FALSE(cell.get_excitatory_dendrites_position().has_value());
        ASSERT_FALSE(cell.get_neuron_position_for(SignalType::EXCITATORY, ElementType::DENDRITE).has_value());

        const Vec3d pos_ex_2{ urd_x(mt), urd_y(mt), urd_z(mt) };
        cell.set_excitatory_dendrites_position(pos_ex_2);

        ASSERT_TRUE(cell.get_excitatory_dendrites_position().has_value());
        ASSERT_EQ(pos_ex_2, cell.get_excitatory_dendrites_position().value());

        ASSERT_TRUE(cell.get_neuron_position_for(SignalType::EXCITATORY, ElementType::DENDRITE).has_value());
        ASSERT_EQ(pos_ex_2, cell.get_neuron_position_for(SignalType::EXCITATORY, ElementType::DENDRITE).value());

        cell.set_excitatory_dendrites_position({});
        ASSERT_FALSE(cell.get_excitatory_dendrites_position().has_value());
        ASSERT_FALSE(cell.get_neuron_position_for(SignalType::EXCITATORY, ElementType::DENDRITE).has_value());

        const Vec3d pos_in_1{ urd_x(mt), urd_y(mt), urd_z(mt) };
        cell.set_inhibitory_dendrites_position(pos_in_1);

        ASSERT_TRUE(cell.get_inhibitory_dendrites_position().has_value());
        ASSERT_EQ(pos_in_1, cell.get_inhibitory_dendrites_position().value());

        ASSERT_TRUE(cell.get_neuron_position_for(SignalType::INHIBITORY, ElementType::DENDRITE).has_value());
        ASSERT_EQ(pos_in_1, cell.get_neuron_position_for(SignalType::INHIBITORY, ElementType::DENDRITE).value());

        cell.set_inhibitory_dendrites_position({});
        ASSERT_FALSE(cell.get_excitatory_dendrites_position().has_value());
        ASSERT_FALSE(cell.get_neuron_position_for(SignalType::EXCITATORY, ElementType::DENDRITE).has_value());

        const Vec3d pos_in_2{ urd_x(mt), urd_y(mt), urd_z(mt) };
        cell.set_inhibitory_dendrites_position(pos_in_2);

        ASSERT_TRUE(cell.get_inhibitory_dendrites_position().has_value());
        ASSERT_EQ(pos_in_2, cell.get_inhibitory_dendrites_position().value());

        ASSERT_TRUE(cell.get_neuron_position_for(SignalType::INHIBITORY, ElementType::DENDRITE).has_value());
        ASSERT_EQ(pos_in_2, cell.get_neuron_position_for(SignalType::INHIBITORY, ElementType::DENDRITE).value());

        cell.set_inhibitory_dendrites_position({});
        ASSERT_FALSE(cell.get_excitatory_dendrites_position().has_value());
        ASSERT_FALSE(cell.get_neuron_position_for(SignalType::EXCITATORY, ElementType::DENDRITE).has_value());
    }
}

TEST(TestCell, testCellPositionException) {
    setup();

    for (auto i = 0; i < iterations; i++) {
        Cell<FastMultipoleMethodsCell> cell{};

        const auto& box_sizes = get_random_simulation_box_size();
        const auto& min = std::get<0>(box_sizes);
        const auto& max = std::get<1>(box_sizes);

        cell.set_size(min, max);

        std::uniform_real_distribution urd_x(min.get_x(), max.get_x());
        std::uniform_real_distribution urd_y(min.get_y(), max.get_y());
        std::uniform_real_distribution urd_z(min.get_z(), max.get_z());

        const Vec3d pos_ex_1{ urd_x(mt), urd_y(mt), urd_z(mt) };
        cell.set_excitatory_dendrites_position(pos_ex_1);

        const Vec3d pos_ex_1_invalid_x_max = max + Vec3d{ 1, 0, 0 };
        const Vec3d pos_ex_1_invalid_y_max = max + Vec3d{ 0, 1, 0 };
        const Vec3d pos_ex_1_invalid_z_max = max + Vec3d{ 0, 0, 1 };

        ASSERT_THROW(cell.set_excitatory_dendrites_position(pos_ex_1_invalid_x_max), RelearnException);
        ASSERT_THROW(cell.set_excitatory_dendrites_position(pos_ex_1_invalid_y_max), RelearnException);
        ASSERT_THROW(cell.set_excitatory_dendrites_position(pos_ex_1_invalid_z_max), RelearnException);

        const Vec3d pos_ex_1_invalid_x_min = min - Vec3d{ 1, 0, 0 };
        const Vec3d pos_ex_1_invalid_y_min = min - Vec3d{ 0, 1, 0 };
        const Vec3d pos_ex_1_invalid_z_min = min - Vec3d{ 0, 0, 1 };

        ASSERT_THROW(cell.set_excitatory_dendrites_position(pos_ex_1_invalid_x_min), RelearnException);
        ASSERT_THROW(cell.set_excitatory_dendrites_position(pos_ex_1_invalid_y_min), RelearnException);
        ASSERT_THROW(cell.set_excitatory_dendrites_position(pos_ex_1_invalid_z_min), RelearnException);

        ASSERT_TRUE(cell.get_excitatory_dendrites_position().has_value());
        ASSERT_EQ(pos_ex_1, cell.get_excitatory_dendrites_position().value());

        ASSERT_TRUE(cell.get_neuron_position_for(SignalType::EXCITATORY, ElementType::DENDRITE).has_value());
        ASSERT_EQ(pos_ex_1, cell.get_neuron_position_for(SignalType::EXCITATORY, ElementType::DENDRITE).value());

        const Vec3d pos_ex_2{ urd_x(mt), urd_y(mt), urd_z(mt) };
        cell.set_excitatory_dendrites_position(pos_ex_2);

        const Vec3d pos_ex_2_invalid_x_max = max + Vec3d{ 1, 0, 0 };
        const Vec3d pos_ex_2_invalid_y_max = max + Vec3d{ 0, 1, 0 };
        const Vec3d pos_ex_2_invalid_z_max = max + Vec3d{ 0, 0, 1 };

        ASSERT_THROW(cell.set_excitatory_dendrites_position(pos_ex_2_invalid_x_max), RelearnException);
        ASSERT_THROW(cell.set_excitatory_dendrites_position(pos_ex_2_invalid_y_max), RelearnException);
        ASSERT_THROW(cell.set_excitatory_dendrites_position(pos_ex_2_invalid_z_max), RelearnException);

        const Vec3d pos_ex_2_invalid_x_min = min - Vec3d{ 1, 0, 0 };
        const Vec3d pos_ex_2_invalid_y_min = min - Vec3d{ 0, 1, 0 };
        const Vec3d pos_ex_2_invalid_z_min = min - Vec3d{ 0, 0, 1 };

        ASSERT_THROW(cell.set_excitatory_dendrites_position(pos_ex_2_invalid_x_min), RelearnException);
        ASSERT_THROW(cell.set_excitatory_dendrites_position(pos_ex_2_invalid_y_min), RelearnException);
        ASSERT_THROW(cell.set_excitatory_dendrites_position(pos_ex_2_invalid_z_min), RelearnException);

        ASSERT_TRUE(cell.get_excitatory_dendrites_position().has_value());
        ASSERT_EQ(pos_ex_2, cell.get_excitatory_dendrites_position().value());

        ASSERT_TRUE(cell.get_neuron_position_for(SignalType::EXCITATORY, ElementType::DENDRITE).has_value());
        ASSERT_EQ(pos_ex_2, cell.get_neuron_position_for(SignalType::EXCITATORY, ElementType::DENDRITE).value());

        const Vec3d pos_in_1{ urd_x(mt), urd_y(mt), urd_z(mt) };
        cell.set_inhibitory_dendrites_position(pos_in_1);

        const Vec3d pos_in_1_invalid_x_max = max + Vec3d{ 1, 0, 0 };
        const Vec3d pos_in_1_invalid_y_max = max + Vec3d{ 0, 1, 0 };
        const Vec3d pos_in_1_invalid_z_max = max + Vec3d{ 0, 0, 1 };

        ASSERT_THROW(cell.set_inhibitory_dendrites_position(pos_in_1_invalid_x_max), RelearnException);
        ASSERT_THROW(cell.set_inhibitory_dendrites_position(pos_in_1_invalid_y_max), RelearnException);
        ASSERT_THROW(cell.set_inhibitory_dendrites_position(pos_in_1_invalid_z_max), RelearnException);

        const Vec3d pos_in_1_invalid_x_min = min - Vec3d{ 1, 0, 0 };
        const Vec3d pos_in_1_invalid_y_min = min - Vec3d{ 0, 1, 0 };
        const Vec3d pos_in_1_invalid_z_min = min - Vec3d{ 0, 0, 1 };

        ASSERT_THROW(cell.set_inhibitory_dendrites_position(pos_in_1_invalid_x_min), RelearnException);
        ASSERT_THROW(cell.set_inhibitory_dendrites_position(pos_in_1_invalid_y_min), RelearnException);
        ASSERT_THROW(cell.set_inhibitory_dendrites_position(pos_in_1_invalid_z_min), RelearnException);

        ASSERT_TRUE(cell.get_inhibitory_dendrites_position().has_value());
        ASSERT_EQ(pos_in_1, cell.get_inhibitory_dendrites_position().value());

        ASSERT_TRUE(cell.get_neuron_position_for(SignalType::INHIBITORY, ElementType::DENDRITE).has_value());
        ASSERT_EQ(pos_in_1, cell.get_neuron_position_for(SignalType::INHIBITORY, ElementType::DENDRITE).value());

        const Vec3d pos_in_2{ urd_x(mt), urd_y(mt), urd_z(mt) };
        cell.set_inhibitory_dendrites_position(pos_in_2);

        const Vec3d pos_in_2_invalid_x_max = max + Vec3d{ 1, 0, 0 };
        const Vec3d pos_in_2_invalid_y_max = max + Vec3d{ 0, 1, 0 };
        const Vec3d pos_in_2_invalid_z_max = max + Vec3d{ 0, 0, 1 };

        ASSERT_THROW(cell.set_inhibitory_dendrites_position(pos_in_2_invalid_x_max), RelearnException);
        ASSERT_THROW(cell.set_inhibitory_dendrites_position(pos_in_2_invalid_y_max), RelearnException);
        ASSERT_THROW(cell.set_inhibitory_dendrites_position(pos_in_2_invalid_z_max), RelearnException);

        const Vec3d pos_in_2_invalid_x_min = min - Vec3d{ 1, 0, 0 };
        const Vec3d pos_in_2_invalid_y_min = min - Vec3d{ 0, 1, 0 };
        const Vec3d pos_in_2_invalid_z_min = min - Vec3d{ 0, 0, 1 };

        ASSERT_THROW(cell.set_inhibitory_dendrites_position(pos_in_2_invalid_x_min), RelearnException);
        ASSERT_THROW(cell.set_inhibitory_dendrites_position(pos_in_2_invalid_y_min), RelearnException);
        ASSERT_THROW(cell.set_inhibitory_dendrites_position(pos_in_2_invalid_z_min), RelearnException);

        ASSERT_TRUE(cell.get_inhibitory_dendrites_position().has_value());
        ASSERT_EQ(pos_in_2, cell.get_inhibitory_dendrites_position().value());

        ASSERT_TRUE(cell.get_neuron_position_for(SignalType::INHIBITORY, ElementType::DENDRITE).has_value());
        ASSERT_EQ(pos_in_2, cell.get_neuron_position_for(SignalType::INHIBITORY, ElementType::DENDRITE).value());
    }
}

TEST(TestCell, testCellPositionCombined) {
    setup();

    for (auto i = 0; i < iterations; i++) {
        Cell<FastMultipoleMethodsCell> cell{};

        const auto& box_sizes = get_random_simulation_box_size();
        const auto& min = std::get<0>(box_sizes);
        const auto& max = std::get<1>(box_sizes);

        cell.set_size(min, max);

        std::uniform_real_distribution urd_x(min.get_x(), max.get_x());
        std::uniform_real_distribution urd_y(min.get_y(), max.get_y());
        std::uniform_real_distribution urd_z(min.get_z(), max.get_z());

        const Vec3d pos_1{ urd_x(mt), urd_y(mt), urd_z(mt) };
        const Vec3d pos_2{ urd_x(mt), urd_y(mt), urd_z(mt) };
        const Vec3d pos_3{ urd_x(mt), urd_y(mt), urd_z(mt) };
        const Vec3d pos_4{ urd_x(mt), urd_y(mt), urd_z(mt) };

        cell.set_neuron_position({});

        ASSERT_FALSE(cell.get_neuron_position().has_value());

        cell.set_excitatory_dendrites_position(pos_1);
        cell.set_inhibitory_dendrites_position(pos_1);

        ASSERT_TRUE(cell.get_neuron_position().has_value());
        ASSERT_EQ(cell.get_neuron_position().value(), pos_1);

        cell.set_excitatory_dendrites_position({});
        cell.set_inhibitory_dendrites_position({});

        ASSERT_FALSE(cell.get_neuron_position().has_value());

        cell.set_excitatory_dendrites_position(pos_2);

        ASSERT_THROW(cell.get_neuron_position(), RelearnException);

        cell.set_inhibitory_dendrites_position(pos_3);

        if (pos_2 == pos_3) {
            ASSERT_TRUE(cell.get_neuron_position().has_value());
            ASSERT_EQ(cell.get_neuron_position().value(), pos_2);
        } else {
            ASSERT_THROW(cell.get_neuron_position(), RelearnException);
        }

        cell.set_neuron_position({});

        ASSERT_FALSE(cell.get_neuron_position().has_value());

        cell.set_excitatory_dendrites_position(pos_4);
        cell.set_inhibitory_dendrites_position(pos_4);

        ASSERT_TRUE(cell.get_neuron_position().has_value());
        ASSERT_EQ(cell.get_neuron_position().value(), pos_4);
    }
}

TEST(TestCell, testCellSetNumDendrites) {
    setup();

    std::uniform_int_distribution<unsigned int> uid(0, 1000);

    for (auto i = 0; i < iterations; i++) {
        Cell<FastMultipoleMethodsCell> cell{};

        const auto num_dends_ex_1 = uid(mt);
        const auto num_dends_in_1 = uid(mt);

        cell.set_number_excitatory_dendrites(num_dends_ex_1);
        cell.set_number_inhibitory_dendrites(num_dends_in_1);

        ASSERT_EQ(num_dends_ex_1, cell.get_number_excitatory_dendrites());
        ASSERT_EQ(num_dends_ex_1, cell.get_number_dendrites_for(SignalType::EXCITATORY));
        ASSERT_EQ(num_dends_in_1, cell.get_number_inhibitory_dendrites());
        ASSERT_EQ(num_dends_in_1, cell.get_number_dendrites_for(SignalType::INHIBITORY));

        const auto num_dends_ex_2 = uid(mt);
        const auto num_dends_in_2 = uid(mt);

        cell.set_number_excitatory_dendrites(num_dends_ex_2);
        cell.set_number_inhibitory_dendrites(num_dends_in_2);

        ASSERT_EQ(num_dends_ex_2, cell.get_number_excitatory_dendrites());
        ASSERT_EQ(num_dends_ex_2, cell.get_number_dendrites_for(SignalType::EXCITATORY));
        ASSERT_EQ(num_dends_in_2, cell.get_number_inhibitory_dendrites());
        ASSERT_EQ(num_dends_in_2, cell.get_number_dendrites_for(SignalType::INHIBITORY));
    }
}

TEST(TestCell, testCellSetNeuronId) {
    setup();

    std::uniform_int_distribution<size_t> uid(0, 1000);

    for (auto i = 0; i < iterations; i++) {
        Cell<FastMultipoleMethodsCell> cell{};

        const auto neuron_id_1 = uid(mt);
        cell.set_neuron_id(neuron_id_1);
        ASSERT_EQ(neuron_id_1, cell.get_neuron_id());

        const auto neuron_id_2 = uid(mt);
        cell.set_neuron_id(neuron_id_2);
        ASSERT_EQ(neuron_id_2, cell.get_neuron_id());
    }
}

TEST(TestCell, testCellOctants) {
    setup();

    for (auto i = 0; i < iterations; i++) {
        Cell<FastMultipoleMethodsCell> cell{};

        const auto& box_sizes = get_random_simulation_box_size();
        const auto& min = std::get<0>(box_sizes);
        const auto& max = std::get<1>(box_sizes);

        cell.set_size(min, max);

        const auto midpoint = (min + max) / 2;

        std::uniform_real_distribution urd_x(min.get_x(), max.get_x());
        std::uniform_real_distribution urd_y(min.get_y(), max.get_y());
        std::uniform_real_distribution urd_z(min.get_z(), max.get_z());

        for (auto id = 0; id < 1000; id++) {
            const Vec3d position{
                urd_x(mt), urd_y(mt), urd_z(mt)
            };

            const auto larger_x = position.get_x() >= midpoint.get_x() ? 1 : 0;
            const auto larger_y = position.get_y() >= midpoint.get_y() ? 2 : 0;
            const auto larger_z = position.get_z() >= midpoint.get_z() ? 4 : 0;

            const auto expected_octant_idx = larger_x + larger_y + larger_z;

            const auto received_idx = cell.get_octant_for_position(position);

            ASSERT_EQ(expected_octant_idx, received_idx);
        }
    }
}

TEST(TestCell, testCellOctantsException) {
    setup();

    for (auto i = 0; i < iterations; i++) {
        Cell<FastMultipoleMethodsCell> cell{};

        const auto& box_sizes = get_random_simulation_box_size();
        const auto& min = std::get<0>(box_sizes);
        const auto& max = std::get<1>(box_sizes);

        cell.set_size(min, max);

        const Vec3d pos_invalid_x_max = max + Vec3d{ 1, 0, 0 };
        const Vec3d pos_invalid_y_max = max + Vec3d{ 0, 1, 0 };
        const Vec3d pos_invalid_z_max = max + Vec3d{ 0, 0, 1 };

        const Vec3d pos_invalid_x_min = min - Vec3d{ 1, 0, 0 };
        const Vec3d pos_invalid_y_min = min - Vec3d{ 0, 1, 0 };
        const Vec3d pos_invalid_z_min = min - Vec3d{ 0, 0, 1 };

        ASSERT_THROW(cell.get_octant_for_position(pos_invalid_x_max), RelearnException);
        ASSERT_THROW(cell.get_octant_for_position(pos_invalid_y_max), RelearnException);
        ASSERT_THROW(cell.get_octant_for_position(pos_invalid_z_max), RelearnException);
        ASSERT_THROW(cell.get_octant_for_position(pos_invalid_x_min), RelearnException);
        ASSERT_THROW(cell.get_octant_for_position(pos_invalid_y_min), RelearnException);
        ASSERT_THROW(cell.get_octant_for_position(pos_invalid_z_min), RelearnException);
    }
}

TEST(TestCell, testCellOctantsSize) {
    setup();

    for (auto i = 0; i < iterations; i++) {
        Cell<FastMultipoleMethodsCell> cell{};

        const auto& box_sizes = get_random_simulation_box_size();
        const auto& min = std::get<0>(box_sizes);
        const auto& max = std::get<1>(box_sizes);

        cell.set_size(min, max);

        const auto midpoint = (min + max) / 2;

        for (auto id = 0; id < 8; id++) {
            const auto larger_x = ((id & 1) == 0) ? 0 : 1;
            const auto larger_y = ((id & 2) == 0) ? 0 : 1;
            const auto larger_z = ((id & 4) == 0) ? 0 : 1;

            auto subcell_min = min;
            auto subcell_max = midpoint;

            if (larger_x == 1) {
                subcell_min += Vec3d{ midpoint.get_x() - min.get_x(), 0, 0 };
                subcell_max += Vec3d{ midpoint.get_x() - min.get_x(), 0, 0 };
            }

            if (larger_y == 1) {
                subcell_min += Vec3d{ 0, midpoint.get_y() - min.get_y(), 0 };
                subcell_max += Vec3d{ 0, midpoint.get_y() - min.get_y(), 0 };
            }

            if (larger_z == 1) {
                subcell_min += Vec3d{ 0, 0, midpoint.get_z() - min.get_z() };
                subcell_max += Vec3d{ 0, 0, midpoint.get_z() - min.get_z() };
            }

            const auto& subcell_received_dims = cell.get_size_for_octant(id);
            const auto& subcell_received_min = std::get<0>(subcell_received_dims);
            const auto& subcell_received_max = std::get<1>(subcell_received_dims);

            const auto diff_subcell_min = subcell_min - subcell_received_min;
            const auto diff_subcell_max = subcell_max - subcell_received_max;

            ASSERT_NEAR(diff_subcell_min.calculate_p_norm(2), 0.0, eps);
            ASSERT_NEAR(diff_subcell_max.calculate_p_norm(2), 0.0, eps);
        }
    }
}

TEST(TestOctreeNode, testOctreeNodeReset) {
    setup();

    OctreeNode node{};

    ASSERT_FALSE(node.is_parent());
    ASSERT_TRUE(node.get_level() == Constants::uninitialized);
    ASSERT_TRUE(node.get_rank() == -1);
    ASSERT_TRUE(node.get_children().size() == Constants::number_oct);

    const auto& children = node.get_children();

    for (auto i = 0; i < Constants::number_oct; i++) {
        ASSERT_TRUE(node.get_child(i) == nullptr);
        ASSERT_TRUE(children[i] == nullptr);
    }

    for (auto it = 0; it < iterations; it++) {
        node.set_parent();

        std::uniform_int_distribution<size_t> uid_level(0, 1000);
        std::uniform_int_distribution<int> uid_rank(0, 1000);

        node.set_level(uid_level(mt));
        node.set_rank(uid_rank(mt));

        std::vector<OctreeNode> other_nodes(Constants::number_oct);
        for (auto i = 0; i < Constants::number_oct; i++) {
            node.set_child(&(other_nodes[i]), i);
        }

        node.reset();

        ASSERT_FALSE(node.is_parent());
        ASSERT_TRUE(node.get_level() == Constants::uninitialized);
        ASSERT_TRUE(node.get_rank() == -1);
        ASSERT_TRUE(node.get_children().size() == Constants::number_oct);

        const auto& children = node.get_children();

        for (auto i = 0; i < Constants::number_oct; i++) {
            ASSERT_TRUE(node.get_child(i) == nullptr);
            ASSERT_TRUE(children[i] == nullptr);
        }
    }
}

TEST(TestOctreeNode, testOctreeNodeSetterGetter) {
    setup();

    OctreeNode node{};

    for (auto it = 0; it < iterations; it++) {
        node.set_parent();

        std::uniform_int_distribution<size_t> uid_level(0, 1000);
        std::uniform_int_distribution<int> uid_rank(0, 1000);

        const auto lvl = uid_level(mt);
        const auto rank = uid_rank(mt);

        node.set_level(lvl);
        node.set_rank(rank);

        std::vector<OctreeNode> other_nodes(Constants::number_oct);
        for (auto i = 0; i < Constants::number_oct; i++) {
            node.set_child(&(other_nodes[i]), i);
        }

        ASSERT_THROW(node.set_rank(-rank), RelearnException);
        ASSERT_THROW(node.set_level(lvl + Constants::uninitialized), RelearnException);

        ASSERT_TRUE(node.is_parent());
        ASSERT_TRUE(node.get_level() == lvl);
        ASSERT_TRUE(node.get_rank() == rank);
        ASSERT_TRUE(node.get_children().size() == Constants::number_oct);

        const auto& children = node.get_children();

        for (auto i = 0; i < Constants::number_oct; i++) {
            ASSERT_TRUE(node.get_child(i) == &(other_nodes[i]));
            ASSERT_TRUE(children[i] == &(other_nodes[i]));
        }

        const auto lb = -uid_rank(mt);
        const auto ub = uid_rank(mt);

        for (auto i = lb; i < ub; i++) {
            if (i >= 0 && i < Constants::number_oct) {
                continue;
            }

            ASSERT_THROW(node.set_child(nullptr, i), RelearnException);
            ASSERT_THROW(node.set_child(&node, i), RelearnException);
            ASSERT_THROW(node.get_child(i), RelearnException);
        }
    }
}

TEST(TestOctreeNode, testOctreeNodeLocal) {
    setup();

    OctreeNode node{};

    for (auto it = 0; it < iterations; it++) {

        const auto my_rank = MPIWrapper::get_my_rank();

        std::uniform_int_distribution<int> uid_rank(0, 1000);

        for (auto i = 0; i < 1000; i++) {
            const auto rank = uid_rank(mt);

            node.set_rank(rank);

            if (rank == my_rank) {
                ASSERT_TRUE(node.is_local());
            } else {
                ASSERT_FALSE(node.is_local());
            }
        }
    }
}

TEST(TestOctreeNode, testOctreeNodeSetterCell) {
    setup();

    OctreeNode node{};

    const Cell<FastMultipoleMethodsCell>& cell = node.get_cell();

    std::uniform_int_distribution<size_t> uid_id(0, 1000);
    std::uniform_int_distribution<unsigned int> uid_dends(0, 1000);

    for (auto it = 0; it < iterations; it++) {
        const auto& box_sizes = get_random_simulation_box_size();

        const auto id = uid_id(mt);
        const auto dends_ex = uid_dends(mt);
        const auto dends_in = uid_dends(mt);

        const auto& min = std::get<0>(box_sizes);
        const auto& max = std::get<1>(box_sizes);

        std::uniform_real_distribution urd_x(min.get_x(), max.get_x());
        std::uniform_real_distribution urd_y(min.get_y(), max.get_y());
        std::uniform_real_distribution urd_z(min.get_z(), max.get_z());

        const Vec3d pos_ex{ urd_x(mt), urd_y(mt), urd_z(mt) };
        const Vec3d pos_in{ urd_x(mt), urd_y(mt), urd_z(mt) };

        node.set_cell_neuron_id(id);
        node.set_cell_number_dendrites(dends_ex, dends_in);
        node.set_cell_size(min, max);
        node.set_cell_excitatory_dendrites_position(pos_ex);
        node.set_cell_inhibitory_dendrites_position(pos_in);

        ASSERT_TRUE(node.get_cell().get_neuron_id() == id);
        ASSERT_TRUE(cell.get_neuron_id() == id);

        ASSERT_TRUE(node.get_cell().get_number_excitatory_dendrites() == dends_ex);
        ASSERT_TRUE(cell.get_number_excitatory_dendrites() == dends_ex);

        ASSERT_TRUE(node.get_cell().get_number_inhibitory_dendrites() == dends_in);
        ASSERT_TRUE(cell.get_number_inhibitory_dendrites() == dends_in);

        ASSERT_TRUE(node.get_cell().get_size() == box_sizes);
        ASSERT_TRUE(cell.get_size() == box_sizes);

        ASSERT_TRUE(node.get_cell().get_excitatory_dendrites_position().has_value());
        ASSERT_TRUE(cell.get_excitatory_dendrites_position().has_value());

        ASSERT_TRUE(node.get_cell().get_excitatory_dendrites_position().value() == pos_ex);
        ASSERT_TRUE(cell.get_excitatory_dendrites_position().value() == pos_ex);

        ASSERT_TRUE(node.get_cell().get_inhibitory_dendrites_position().has_value());
        ASSERT_TRUE(cell.get_inhibitory_dendrites_position().has_value());

        ASSERT_TRUE(node.get_cell().get_inhibitory_dendrites_position().value() == pos_in);
        ASSERT_TRUE(cell.get_inhibitory_dendrites_position().value() == pos_in);

        node.set_cell_excitatory_dendrites_position({});
        node.set_cell_inhibitory_dendrites_position({});

        ASSERT_FALSE(node.get_cell().get_excitatory_dendrites_position().has_value());
        ASSERT_FALSE(cell.get_excitatory_dendrites_position().has_value());

        ASSERT_FALSE(node.get_cell().get_inhibitory_dendrites_position().has_value());
        ASSERT_FALSE(cell.get_inhibitory_dendrites_position().has_value());
    }
}

TEST(TestOctreeNode, testOctreeNodeInsert) {
    setup();

    const auto my_rank = MPIWrapper::get_my_rank();

    std::uniform_int_distribution<size_t> uid_lvl(0, 6);
    std::uniform_int_distribution<size_t> uid(0, 10000);
    std::uniform_real_distribution<double> urd_sigma(1, 10000.0);
    std::uniform_real_distribution<double> urd_theta(0.0, 1.0);

    for (auto it = 0; it < iterations; it++) {
        Vec3d min{};
        Vec3d max{};

        std::tie(min, max) = get_random_simulation_box_size();
        size_t level = uid_lvl(mt);

        std::uniform_real_distribution<double> urd_x(min.get_x(), max.get_x());
        std::uniform_real_distribution<double> urd_y(min.get_y(), max.get_y());
        std::uniform_real_distribution<double> urd_z(min.get_z(), max.get_z());

        Vec3d own_position{ urd_x(mt), urd_y(mt), urd_z(mt) };

        OctreeNode node{};
        node.set_level(level);
        node.set_rank(my_rank);
        node.set_cell_size(min, max);

        node.set_cell_neuron_position(own_position);

        size_t num_neurons = uid(mt);
        size_t num_additional_ids = uid(mt);

        std::vector<std::tuple<Vec3d, size_t>> neurons_to_place = generate_random_neurons(min, max, num_neurons, num_neurons + num_additional_ids);

        for (const auto& [pos, id] : neurons_to_place) {
            node.insert(pos, id, my_rank);
        }

        std::vector<std::tuple<Vec3d, size_t>> placed_neurons = extract_neurons(&node);

        std::sort(neurons_to_place.begin(), neurons_to_place.end(), [](std::tuple<Vec3d, size_t> a, std::tuple<Vec3d, size_t> b) { return std::get<1>(a) > std::get<1>(b); });
        std::sort(placed_neurons.begin(), placed_neurons.end(), [](std::tuple<Vec3d, size_t> a, std::tuple<Vec3d, size_t> b) { return std::get<1>(a) > std::get<1>(b); });

        ASSERT_EQ(neurons_to_place.size(), placed_neurons.size());

        for (auto i = 0; i < neurons_to_place.size(); i++) {
            const auto& expected_neuron = neurons_to_place[i];
            const auto& found_neuron = placed_neurons[i];

            ASSERT_EQ(expected_neuron, found_neuron);
        }
    }
}

TEST(TestOctree, testOctreeConstructor) {
    setup();

    std::uniform_int_distribution<size_t> uid(0, 6);
    std::uniform_real_distribution<double> urd_sigma(1, 10000.0);
    std::uniform_real_distribution<double> urd_theta(0.0, 1.0);

    for (auto i = 0; i < iterations; i++) {
        Vec3d min{};
        Vec3d max{};

        std::tie(min, max) = get_random_simulation_box_size();

        size_t level_of_branch_nodes = uid(mt);

        Octree octree(min, max, level_of_branch_nodes);

        ASSERT_EQ(octree.get_acceptance_criterion(), Octree::default_theta);
        ASSERT_EQ(octree.get_level_of_branch_nodes(), level_of_branch_nodes);
        ASSERT_EQ(octree.get_probabilty_parameter(), Octree::default_sigma);
        ASSERT_EQ(octree.get_xyz_max(), max);
        ASSERT_EQ(octree.get_xyz_min(), min);

        if (Octree::default_theta == 0.0) {
            ASSERT_TRUE(octree.is_naive_method_used());
        } else {
            ASSERT_FALSE(octree.is_naive_method_used());
        }

        const auto& virtual_neurons = extract_unused_neurons(octree.get_root());

        std::map<size_t, size_t> level_to_count{};

        for (const auto& [pos, id] : virtual_neurons) {
            level_to_count[id]++;
        }

        ASSERT_EQ(level_to_count.size(), level_of_branch_nodes + 1);

        for (auto level = 0; level <= level_of_branch_nodes; level++) {
            auto expected_elements = 1;

            for (auto it = 0; it < level; it++) {
                expected_elements *= 8;
            }

            ASSERT_EQ(level_to_count[level], expected_elements);
        }
    }

    for (auto i = 0; i < iterations; i++) {
        Vec3d min{};
        Vec3d max{};

        std::tie(min, max) = get_random_simulation_box_size();

        size_t level_of_branch_nodes = uid(mt);
        double theta = urd_theta(mt);
        double sigma = urd_sigma(mt);

        Octree octree(min, max, level_of_branch_nodes, theta, sigma);

        ASSERT_EQ(octree.get_acceptance_criterion(), theta);
        ASSERT_EQ(octree.get_level_of_branch_nodes(), level_of_branch_nodes);
        ASSERT_EQ(octree.get_probabilty_parameter(), sigma);
        ASSERT_EQ(octree.get_xyz_max(), max);
        ASSERT_EQ(octree.get_xyz_min(), min);

        if (theta == 0.0) {
            ASSERT_TRUE(octree.is_naive_method_used());
        } else {
            ASSERT_FALSE(octree.is_naive_method_used());
        }

        const auto& virtual_neurons = extract_unused_neurons(octree.get_root());

        std::map<size_t, size_t> level_to_count{};

        for (const auto& [pos, id] : virtual_neurons) {
            level_to_count[id]++;
        }

        ASSERT_EQ(level_to_count.size(), level_of_branch_nodes + 1);

        for (auto level = 0; level <= level_of_branch_nodes; level++) {
            size_t expected_elements = 1;

            for (auto it = 0; it < level; it++) {
                expected_elements *= 8;
            }

            if (level == level_of_branch_nodes) {
                ASSERT_EQ(octree.get_num_local_trees(), expected_elements);
            }

            ASSERT_EQ(level_to_count[level], expected_elements);
        }
    }
}

TEST(TestOctree, testOctreeConstructorExceptions) {
    setup();

    std::uniform_int_distribution<size_t> uid(0, 6);
    std::uniform_real_distribution<double> urd_sigma(1, 10000.0);
    std::uniform_real_distribution<double> urd_theta(0.0, 1.0);

    for (auto i = 0; i < iterations; i++) {
        Vec3d min_xyz{};
        Vec3d max_xyz{};

        std::tie(min_xyz, max_xyz) = get_random_simulation_box_size();

        size_t level_of_branch_nodes = uid(mt);

        ASSERT_THROW(Octree octree(max_xyz, min_xyz, level_of_branch_nodes), RelearnException);
    }

    for (auto i = 0; i < iterations; i++) {
        Vec3d min{};
        Vec3d max{};

        std::tie(min, max) = get_random_simulation_box_size();

        size_t level_of_branch_nodes = uid(mt);
        double theta = urd_theta(mt);
        double sigma = urd_sigma(mt);

        ASSERT_THROW(Octree octree(max, min, level_of_branch_nodes, theta, sigma), RelearnException);
    }

    for (auto i = 0; i < iterations; i++) {
        Vec3d min{};
        Vec3d max{};

        std::tie(min, max) = get_random_simulation_box_size();

        size_t level_of_branch_nodes = uid(mt);
        double theta = urd_theta(mt);
        double sigma = urd_sigma(mt) * -1;

        ASSERT_THROW(Octree octree(min, max, level_of_branch_nodes, theta, sigma), RelearnException);
    }

    for (auto i = 0; i < iterations; i++) {
        Vec3d min{};
        Vec3d max{};

        std::tie(min, max) = get_random_simulation_box_size();

        size_t level_of_branch_nodes = uid(mt);
        double theta = urd_theta(mt) - 1;
        double sigma = urd_sigma(mt);

        ASSERT_THROW(Octree octree(min, max, level_of_branch_nodes, theta, sigma), RelearnException);
    }
}

TEST(TestOctree, testOctreeSetterGetter) {
    setup();

    std::uniform_int_distribution<size_t> uid(0, 6);
    std::uniform_real_distribution<double> urd_sigma(1, 10000.0);
    std::uniform_real_distribution<double> urd_theta(0.0, 1.0);

    for (auto i = 0; i < iterations; i++) {
        Vec3d min{};
        Vec3d max{};

        std::tie(min, max) = get_random_simulation_box_size();

        size_t level_of_branch_nodes = uid(mt);
        double theta = urd_theta(mt);
        double sigma = urd_sigma(mt);

        Octree octree(min, max, level_of_branch_nodes, theta, sigma);

        std::tie(min, max) = get_random_simulation_box_size();

        level_of_branch_nodes = uid(mt);
        theta = urd_theta(mt);
        sigma = urd_sigma(mt);

        octree.set_size(min, max);
        octree.set_acceptance_criterion(theta);
        octree.set_probability_parameter(sigma);
        octree.set_level_of_branch_nodes(level_of_branch_nodes);

        ASSERT_EQ(octree.get_acceptance_criterion(), theta);
        ASSERT_EQ(octree.get_level_of_branch_nodes(), level_of_branch_nodes);
        ASSERT_EQ(octree.get_probabilty_parameter(), sigma);
        ASSERT_EQ(octree.get_xyz_max(), max);
        ASSERT_EQ(octree.get_xyz_min(), min);

        if (theta == 0.0) {
            ASSERT_TRUE(octree.is_naive_method_used());
        } else {
            ASSERT_FALSE(octree.is_naive_method_used());
        }

        octree.set_acceptance_criterion(0.0);
        ASSERT_EQ(octree.get_acceptance_criterion(), 0.0);
        ASSERT_TRUE(octree.is_naive_method_used());
    }
}

TEST(TestOctree, testOctreeSetterGetterExceptions) {
    setup();

    std::uniform_int_distribution<size_t> uid(0, 6);
    std::uniform_real_distribution<double> urd_sigma(1, 10000.0);
    std::uniform_real_distribution<double> urd_theta(0.0, 1.0);

    for (auto i = 0; i < iterations; i++) {
        Vec3d min{};
        Vec3d max{};

        std::tie(min, max) = get_random_simulation_box_size();

        size_t level_of_branch_nodes = uid(mt);
        double theta = urd_theta(mt);
        double sigma = urd_sigma(mt);

        Octree octree(min, max, level_of_branch_nodes, theta, sigma);

        std::tie(min, max) = get_random_simulation_box_size();

        ASSERT_THROW(octree.set_size(max, min), RelearnException);
    }

    for (auto i = 0; i < iterations; i++) {
        Vec3d min{};
        Vec3d max{};

        std::tie(min, max) = get_random_simulation_box_size();

        size_t level_of_branch_nodes = uid(mt);
        double theta = urd_theta(mt);
        double sigma = urd_sigma(mt);

        Octree octree(min, max, level_of_branch_nodes, theta, sigma);

        std::tie(min, max) = get_random_simulation_box_size();

        theta = urd_theta(mt) - 1;

        ASSERT_THROW(octree.set_acceptance_criterion(theta), RelearnException);
    }

    for (auto i = 0; i < iterations; i++) {
        Vec3d min{};
        Vec3d max{};

        std::tie(min, max) = get_random_simulation_box_size();

        size_t level_of_branch_nodes = uid(mt);
        double theta = urd_theta(mt);
        double sigma = urd_sigma(mt);

        Octree octree(min, max, level_of_branch_nodes, theta, sigma);

        std::tie(min, max) = get_random_simulation_box_size();

        sigma = urd_sigma(mt) * -1;

        ASSERT_THROW(octree.set_probability_parameter(sigma), RelearnException);
    }
}

TEST(TestOctree, testOctreeInsertNeurons) {
    setup();

    const auto my_rank = MPIWrapper::get_my_rank();

    std::uniform_int_distribution<size_t> uid_lvl(0, 6);
    std::uniform_int_distribution<size_t> uid(0, 10000);
    std::uniform_real_distribution<double> urd_sigma(1, 10000.0);
    std::uniform_real_distribution<double> urd_theta(0.0, 1.0);

    for (auto i = 0; i < iterations; i++) {
        Vec3d min{};
        Vec3d max{};

        std::tie(min, max) = get_random_simulation_box_size();

        size_t level_of_branch_nodes = uid_lvl(mt);
        double theta = urd_theta(mt);
        double sigma = urd_sigma(mt);

        Octree octree(min, max, level_of_branch_nodes, theta, sigma);

        size_t num_neurons = uid(mt);
        size_t num_additional_ids = uid(mt);

        std::vector<std::tuple<Vec3d, size_t>> neurons_to_place = generate_random_neurons(min, max, num_neurons, num_neurons + num_additional_ids);

        for (const auto& [position, id] : neurons_to_place) {
            octree.insert(position, id, my_rank);
        }

        std::vector<std::tuple<Vec3d, size_t>> placed_neurons = extract_neurons(octree);

        std::sort(neurons_to_place.begin(), neurons_to_place.end(), [](std::tuple<Vec3d, size_t> a, std::tuple<Vec3d, size_t> b) { return std::get<1>(a) > std::get<1>(b); });
        std::sort(placed_neurons.begin(), placed_neurons.end(), [](std::tuple<Vec3d, size_t> a, std::tuple<Vec3d, size_t> b) { return std::get<1>(a) > std::get<1>(b); });

        ASSERT_EQ(neurons_to_place.size(), placed_neurons.size());

        for (auto i = 0; i < neurons_to_place.size(); i++) {
            const auto& expected_neuron = neurons_to_place[i];
            const auto& found_neuron = placed_neurons[i];

            ASSERT_EQ(expected_neuron, found_neuron);
        }
    }
}

TEST(TestOctree, testOctreeInsertNeuronsExceptions) {
    setup();

    std::uniform_int_distribution<size_t> uid_lvl(0, 6);
    std::uniform_int_distribution<int> uid_rank(0, 1000);
    std::uniform_int_distribution<size_t> uid(0, 10000);
    std::uniform_real_distribution<double> urd_sigma(1, 10000.0);
    std::uniform_real_distribution<double> urd_theta(0.0, 1.0);

    for (auto i = 0; i < iterations; i++) {
        Vec3d min{};
        Vec3d max{};

        std::tie(min, max) = get_random_simulation_box_size();

        size_t level_of_branch_nodes = uid_lvl(mt);
        double theta = urd_theta(mt);
        double sigma = urd_sigma(mt);

        Octree octree(min, max, level_of_branch_nodes, theta, sigma);

        size_t num_neurons = uid(mt);
        size_t num_additional_ids = uid(mt);

        std::vector<std::tuple<Vec3d, size_t>> neurons_to_place = generate_random_neurons(min, max, num_neurons, num_neurons + num_additional_ids);

        for (const auto& [position, id] : neurons_to_place) {
            const auto rank = uid_rank(mt);

            const Vec3d pos_invalid_x_max = max + Vec3d{ 1, 0, 0 };
            const Vec3d pos_invalid_y_max = max + Vec3d{ 0, 1, 0 };
            const Vec3d pos_invalid_z_max = max + Vec3d{ 0, 0, 1 };

            const Vec3d pos_invalid_x_min = min - Vec3d{ 1, 0, 0 };
            const Vec3d pos_invalid_y_min = min - Vec3d{ 0, 1, 0 };
            const Vec3d pos_invalid_z_min = min - Vec3d{ 0, 0, 1 };

            ASSERT_THROW(octree.insert(position, id, -rank - 1), RelearnException);
            ASSERT_THROW(octree.insert(position, id + Constants::uninitialized, rank), RelearnException);

            ASSERT_THROW(octree.insert(pos_invalid_x_max, id, rank), RelearnException);
            ASSERT_THROW(octree.insert(pos_invalid_y_max, id, rank), RelearnException);
            ASSERT_THROW(octree.insert(pos_invalid_z_max, id, rank), RelearnException);

            ASSERT_THROW(octree.insert(pos_invalid_x_min, id, rank), RelearnException);
            ASSERT_THROW(octree.insert(pos_invalid_y_min, id, rank), RelearnException);
            ASSERT_THROW(octree.insert(pos_invalid_z_min, id, rank), RelearnException);
        }
    }
}

TEST(TestOctree, testOctreeStructure) {
    setup();

    const auto my_rank = MPIWrapper::get_my_rank();

    std::uniform_int_distribution<size_t> uid_lvl(0, 6);
    std::uniform_int_distribution<size_t> uid(0, 10000);
    std::uniform_real_distribution<double> urd_sigma(1, 10000.0);
    std::uniform_real_distribution<double> urd_theta(0.0, 1.0);

    for (auto i = 0; i < iterations; i++) {
        Vec3d min{};
        Vec3d max{};

        std::tie(min, max) = get_random_simulation_box_size();

        size_t level_of_branch_nodes = uid_lvl(mt);
        double theta = urd_theta(mt);
        double sigma = urd_sigma(mt);

        Octree octree(min, max, level_of_branch_nodes, theta, sigma);

        size_t num_neurons = uid(mt);
        size_t num_additional_ids = uid(mt);

        std::vector<std::tuple<Vec3d, size_t>> neurons_to_place = generate_random_neurons(min, max, num_neurons, num_neurons + num_additional_ids);

        for (const auto& [position, id] : neurons_to_place) {
            octree.insert(position, id, my_rank);
        }

        const auto root = octree.get_root();

        std::stack<std::pair<OctreeNode*, size_t>> octree_nodes{};
        octree_nodes.emplace(root, root->get_level());

        while (!octree_nodes.empty()) {
            const auto& elem = octree_nodes.top();

            OctreeNode* current_node = elem.first;
            auto level = elem.second;

            octree_nodes.pop();

            ASSERT_EQ(current_node->get_rank(), my_rank);

            if (current_node->is_parent()) {
                const auto childs = current_node->get_children();
                for (auto i = 0; i < 8; i++) {
                    const auto child = childs[i];
                    if (child != nullptr) {
                        ASSERT_TRUE(level + 1 == child->get_level());
                        octree_nodes.emplace(child, child->get_level());

                        const auto& subcell_size = child->get_cell().get_size();
                        const auto& expected_subcell_size = current_node->get_cell().get_size_for_octant(i);

                        ASSERT_EQ(expected_subcell_size, subcell_size);
                    }
                }

                ASSERT_EQ(current_node->get_cell().get_neuron_id(), Constants::uninitialized);

            } else {
                const auto& cell = current_node->get_cell();
                const auto& opt_position = cell.get_neuron_position();

                ASSERT_TRUE(opt_position.has_value());

                const auto& position = opt_position.value();

                const auto& cell_size = cell.get_size();
                const auto& cell_min = std::get<0>(cell_size);
                const auto& cell_max = std::get<1>(cell_size);

                ASSERT_LE(cell_min.get_x(), position.get_x());
                ASSERT_LE(cell_min.get_y(), position.get_y());
                ASSERT_LE(cell_min.get_z(), position.get_z());

                ASSERT_LE(position.get_x(), cell_max.get_x());
                ASSERT_LE(position.get_y(), cell_max.get_y());
                ASSERT_LE(position.get_z(), cell_max.get_z());
            }
        }
    }
}

TEST(TestOctree, testOctreeLocalTrees) {
    setup();

    std::uniform_int_distribution<size_t> uid(0, 6);
    std::uniform_real_distribution<double> urd_sigma(1, 10000.0);
    std::uniform_real_distribution<double> urd_theta(0.0, 1.0);

    for (auto i = 0; i < iterations; i++) {
        Vec3d min{};
        Vec3d max{};

        std::tie(min, max) = get_random_simulation_box_size();

        size_t level_of_branch_nodes = uid(mt);
        double theta = urd_theta(mt);
        double sigma = urd_sigma(mt);

        Octree octree(min, max, level_of_branch_nodes, theta, sigma);

        ASSERT_EQ(octree.get_acceptance_criterion(), theta);
        ASSERT_EQ(octree.get_level_of_branch_nodes(), level_of_branch_nodes);
        ASSERT_EQ(octree.get_probabilty_parameter(), sigma);
        ASSERT_EQ(octree.get_xyz_max(), max);
        ASSERT_EQ(octree.get_xyz_min(), min);

        if (theta == 0.0) {
            ASSERT_TRUE(octree.is_naive_method_used());
        } else {
            ASSERT_FALSE(octree.is_naive_method_used());
        }

        SpaceFillingCurve<Morton> sfc(level_of_branch_nodes);
        const auto num_cells_per_dimension = 1 << level_of_branch_nodes;

        const auto& cell_length = (max - min) / num_cells_per_dimension;

        const auto& branch_nodes_extracted = extract_branch_nodes(octree.get_root());

        for (auto* branch_node : branch_nodes_extracted) {
            const auto branch_node_position = branch_node->get_cell().get_neuron_position().value();
            const auto branch_node_offset = branch_node_position - min;

            const auto x_pos = branch_node_offset.get_x() / cell_length.get_x();
            const auto y_pos = branch_node_offset.get_y() / cell_length.get_y();
            const auto z_pos = branch_node_offset.get_z() / cell_length.get_z();

            Vec3s pos3d{ static_cast<size_t>(std::floor(x_pos)), static_cast<size_t>(std::floor(y_pos)), static_cast<size_t>(std::floor(z_pos)) };
            const auto pos1d = sfc.map_3d_to_1d(pos3d);

            const auto local_tree = octree.get_local_root(pos1d);
            ASSERT_EQ(local_tree, branch_node);
        }
    }
}

TEST(TestOctree, testOctreeInsertLocalTree) {
    setup();

    const auto my_rank = MPIWrapper::get_my_rank();

    std::uniform_int_distribution<size_t> uid_lvl(0, 6);
    std::uniform_int_distribution<size_t> uid(0, 10000);
    std::uniform_real_distribution<double> urd_sigma(1, 10000.0);
    std::uniform_real_distribution<double> urd_theta(0.0, 1.0);

    for (auto i = 0; i < iterations; i++) {
        Vec3d min{};
        Vec3d max{};

        std::tie(min, max) = get_random_simulation_box_size();

        size_t level_of_branch_nodes = uid_lvl(mt);
        double theta = urd_theta(mt);
        double sigma = urd_sigma(mt);

        Octree octree(min, max, level_of_branch_nodes, theta, sigma);

        size_t num_neurons = uid(mt);
        size_t num_additional_ids = uid(mt);

        std::vector<std::tuple<Vec3d, size_t>> neurons_to_place = generate_random_neurons(min, max, num_neurons, num_neurons + num_additional_ids);

        for (const auto& [position, id] : neurons_to_place) {
            octree.insert(position, id, my_rank);
        }

        const auto num_local_trees = octree.get_num_local_trees();

        std::vector<OctreeNode*> nodes_to_refer_to(1000);
        for (auto i = 0; i < 1000; i++) {
            nodes_to_refer_to[i] = new OctreeNode;
        }

        std::vector<OctreeNode*> nodes_to_save_local_trees(num_local_trees);
        std::vector<OctreeNode*> nodes_to_save_new_local_trees(num_local_trees);
        for (auto i = 0; i < num_local_trees; i++) {
            nodes_to_save_local_trees[i] = new OctreeNode;
            nodes_to_save_new_local_trees[i] = new OctreeNode;
        }

        std::uniform_int_distribution uid_nodes(0, 2000);

        for (auto i = 0; i < num_local_trees; i++) {
            auto* local_tree = octree.get_local_root(i);

            Vec3d cell_min, cell_max;
            std::tie(cell_min, cell_max) = local_tree->get_cell().get_size();

            std::uniform_real_distribution urd_x(cell_min.get_x(), cell_max.get_x());
            std::uniform_real_distribution urd_y(cell_min.get_y(), cell_max.get_y());
            std::uniform_real_distribution urd_z(cell_min.get_z(), cell_max.get_z());

            Vec3d position{ urd_x(mt), urd_y(mt), urd_z(mt) };

            OctreeNode node{};
            node.set_cell_size(cell_min, cell_max);
            node.set_cell_neuron_position(position);
            node.set_rank(my_rank);
            node.set_level(level_of_branch_nodes);

            nodes_to_save_new_local_trees[i]->set_cell_size(cell_min, cell_max);
            nodes_to_save_new_local_trees[i]->set_cell_neuron_position(position);
            nodes_to_save_new_local_trees[i]->set_rank(my_rank);
            nodes_to_save_new_local_trees[i]->set_level(level_of_branch_nodes);

            for (auto j = 0; j < 8; j++) {
                const auto id_nodes = uid_nodes(mt);
                if (id_nodes < 1000) {
                    node.set_child(nodes_to_refer_to[id_nodes], j);
                    nodes_to_save_new_local_trees[i]->set_child(nodes_to_refer_to[id_nodes], j);
                } else {
                    node.set_child(nullptr, j);
                    nodes_to_save_new_local_trees[i]->set_child(nullptr, j);
                }
            }

            *nodes_to_save_local_trees[i] = *(octree.get_local_root(i));
            octree.insert_local_tree(&node, i);
        }

        for (auto i = 0; i < num_local_trees; i++) {
            auto* local_tree = octree.get_local_root(i);
            auto* local_tree_saved = nodes_to_save_new_local_trees[i];

            ASSERT_EQ(local_tree->get_children(), local_tree_saved->get_children());
            ASSERT_EQ(local_tree->get_level(), local_tree_saved->get_level());
            ASSERT_EQ(local_tree->get_rank(), local_tree_saved->get_rank());

            ASSERT_EQ(local_tree->get_cell().get_neuron_id(), local_tree_saved->get_cell().get_neuron_id());
            ASSERT_EQ(local_tree->get_cell().get_size(), local_tree_saved->get_cell().get_size());
            ASSERT_EQ(local_tree->get_cell().get_neuron_position(), local_tree_saved->get_cell().get_neuron_position());
        }

        for (auto i = 0; i < num_local_trees; i++) {
            auto* local_node = nodes_to_save_local_trees[i];

            octree.insert_local_tree(local_node, i);

            auto* local_tree = octree.get_local_root(i);

            ASSERT_EQ(local_tree->get_children(), local_node->get_children());
            ASSERT_EQ(local_tree->get_level(), local_node->get_level());
            ASSERT_EQ(local_tree->get_rank(), local_node->get_rank());

            ASSERT_EQ(local_tree->get_cell().get_neuron_id(), local_node->get_cell().get_neuron_id());
            ASSERT_EQ(local_tree->get_cell().get_size(), local_node->get_cell().get_size());
            ASSERT_EQ(local_tree->get_cell().get_neuron_position(), local_node->get_cell().get_neuron_position());
        }

        for (auto i = 0; i < 1000; i++) {
            delete nodes_to_refer_to[i];
        }

        for (auto i = 0; i < num_local_trees; i++) {
            delete nodes_to_save_local_trees[i];
            delete nodes_to_save_new_local_trees[i];
        }
    }
}
