#include "../googletest/include/gtest/gtest.h"

#include "commons.h"

#include "../source/structure/Cell.h"
#include "../source/structure/Partition.h"
#include "../source/structure/Octree.h"

#include "../source/util/RelearnException.h"
#include "../source/util/Vec3.h"

#include <algorithm>
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

std::vector<std::tuple<Vec3d, size_t>> extract_neurons(const Octree& octree) {
    std::vector<std::tuple<Vec3d, size_t>> return_value;

    const auto root = octree.get_root();
    if (root == nullptr) {
        return return_value;
    }

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

            return_value.emplace_back(position, neuron_id);
        }
    }

    return return_value;
}

TEST(TestOctree, testOctreeConstructor) {
    setup();

    std::uniform_int_distribution<size_t> uid(0, 10);
    std::uniform_real_distribution<double> urd_sigma(1, 10000.0);
    std::uniform_real_distribution<double> urd_theta(0.0, 1.0);

    for (auto i = 0; i < iterations; i++) {
        Vec3d min{};
        Vec3d max{};

        std::tie(min, max) = get_random_simulation_box_size();

        size_t level_of_branch_nodes = uid(mt);

        Octree octree(min, max, level_of_branch_nodes);

        EXPECT_EQ(octree.get_acceptance_criterion(), Octree::default_theta);
        EXPECT_EQ(octree.get_level_of_branch_nodes(), level_of_branch_nodes);
        EXPECT_EQ(octree.get_probabilty_parameter(), Octree::default_sigma);
        EXPECT_EQ(octree.get_xyz_max(), max);
        EXPECT_EQ(octree.get_xyz_min(), min);

        if (Octree::default_theta == 0.0) {
            EXPECT_TRUE(octree.is_naive_method_used());
        } else {
            EXPECT_FALSE(octree.is_naive_method_used());
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

        EXPECT_EQ(octree.get_acceptance_criterion(), theta);
        EXPECT_EQ(octree.get_level_of_branch_nodes(), level_of_branch_nodes);
        EXPECT_EQ(octree.get_probabilty_parameter(), sigma);
        EXPECT_EQ(octree.get_xyz_max(), max);
        EXPECT_EQ(octree.get_xyz_min(), min);

        if (theta == 0.0) {
            EXPECT_TRUE(octree.is_naive_method_used());
        } else {
            EXPECT_FALSE(octree.is_naive_method_used());
        }
    }
}

TEST(TestOctree, testOctreeConstructorExceptions) {
    setup();

    std::uniform_int_distribution<size_t> uid(0, 10);
    std::uniform_real_distribution<double> urd_sigma(1, 10000.0);
    std::uniform_real_distribution<double> urd_theta(0.0, 1.0);

    for (auto i = 0; i < iterations; i++) {
        Vec3d min_xyz{};
        Vec3d max_xyz{};

        std::tie(min_xyz, max_xyz) = get_random_simulation_box_size();

        size_t level_of_branch_nodes = uid(mt);

        EXPECT_THROW(Octree octree(max_xyz, min_xyz, level_of_branch_nodes), RelearnException);
    }

    for (auto i = 0; i < iterations; i++) {
        Vec3d min{};
        Vec3d max{};

        std::tie(min, max) = get_random_simulation_box_size();

        size_t level_of_branch_nodes = uid(mt);
        double theta = urd_theta(mt);
        double sigma = urd_sigma(mt);

        EXPECT_THROW(Octree octree(max, min, level_of_branch_nodes, theta, sigma), RelearnException);
    }

    for (auto i = 0; i < iterations; i++) {
        Vec3d min{};
        Vec3d max{};

        std::tie(min, max) = get_random_simulation_box_size();

        size_t level_of_branch_nodes = uid(mt);
        double theta = urd_theta(mt);
        double sigma = urd_sigma(mt) * -1;

        EXPECT_THROW(Octree octree(min, max, level_of_branch_nodes, theta, sigma), RelearnException);
    }

    for (auto i = 0; i < iterations; i++) {
        Vec3d min{};
        Vec3d max{};

        std::tie(min, max) = get_random_simulation_box_size();

        size_t level_of_branch_nodes = uid(mt);
        double theta = urd_theta(mt) - 1;
        double sigma = urd_sigma(mt);

        EXPECT_THROW(Octree octree(min, max, level_of_branch_nodes, theta, sigma), RelearnException);
    }
}

TEST(TestOctree, testOctreeSetterGetter) {
    setup();

    std::uniform_int_distribution<size_t> uid(0, 10);
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

        EXPECT_EQ(octree.get_acceptance_criterion(), theta);
        EXPECT_EQ(octree.get_level_of_branch_nodes(), level_of_branch_nodes);
        EXPECT_EQ(octree.get_probabilty_parameter(), sigma);
        EXPECT_EQ(octree.get_xyz_max(), max);
        EXPECT_EQ(octree.get_xyz_min(), min);

        if (theta == 0.0) {
            EXPECT_TRUE(octree.is_naive_method_used());
        } else {
            EXPECT_FALSE(octree.is_naive_method_used());
        }

        octree.set_acceptance_criterion(0.0);
        EXPECT_EQ(octree.get_acceptance_criterion(), 0.0);
        EXPECT_TRUE(octree.is_naive_method_used());
    }
}

TEST(TestOctree, testOctreeSetterGetterExceptions) {
    setup();

    std::uniform_int_distribution<size_t> uid(0, 10);
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

        EXPECT_THROW(octree.set_size(max, min), RelearnException);
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

        EXPECT_THROW(octree.set_acceptance_criterion(theta), RelearnException);
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

        EXPECT_THROW(octree.set_probability_parameter(sigma), RelearnException);
    }
}

TEST(TestOctree, testOctreeInsertNeurons) {
    setup();

    const auto my_rank = MPIWrapper::get_my_rank();

    std::uniform_int_distribution<size_t> uid(0, 10000);
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

        size_t num_neurons = uid(mt);
        size_t num_additional_ids = uid(mt);

        std::vector<std::tuple<Vec3d, size_t>> neurons_to_place = generate_random_neurons(min, max, num_neurons, num_neurons + num_additional_ids);

        for (const auto& [position, id] : neurons_to_place) {
            octree.insert(position, id, my_rank);
        }

        std::vector<std::tuple<Vec3d, size_t>> placed_neurons = extract_neurons(octree);

        std::sort(neurons_to_place.begin(), neurons_to_place.end(), [](std::tuple<Vec3d, size_t> a, std::tuple<Vec3d, size_t> b) { return std::get<1>(a) > std::get<1>(b); });
        std::sort(placed_neurons.begin(), placed_neurons.end(), [](std::tuple<Vec3d, size_t> a, std::tuple<Vec3d, size_t> b) { return std::get<1>(a) > std::get<1>(b); });

        EXPECT_EQ(neurons_to_place.size(), placed_neurons.size());

        for (auto i = 0; i < neurons_to_place.size(); i++) {
            const auto& expected_neuron = neurons_to_place[i];
            const auto& found_neuron = placed_neurons[i];

            EXPECT_EQ(expected_neuron, found_neuron);
        }
    }
}

TEST(TestCell, testCellSize) {
    setup();

    for (auto i = 0; i < iterations; i++) {
        Cell cell{};

        const auto& box_sizes_1 = get_random_simulation_box_size();
        const auto& min_1 = std::get<0>(box_sizes_1);
        const auto& max_1 = std::get<1>(box_sizes_1);

        cell.set_size(min_1, max_1);

        const auto& res_1 = cell.get_size();

        EXPECT_EQ(min_1, std::get<0>(res_1));
        EXPECT_EQ(max_1, std::get<1>(res_1));

        const auto& box_sizes_2 = get_random_simulation_box_size();
        const auto& min_2 = std::get<0>(box_sizes_2);
        const auto& max_2 = std::get<1>(box_sizes_2);

        cell.set_size(min_2, max_2);

        const auto& res_2 = cell.get_size();

        EXPECT_EQ(min_2, std::get<0>(res_2));
        EXPECT_EQ(max_2, std::get<1>(res_2));

        EXPECT_EQ(cell.get_maximal_dimension_difference(), (max_2 - min_2).get_maximum());
    }
}

TEST(TestCell, testCellPosition) {
    setup();

    for (auto i = 0; i < iterations; i++) {
        Cell cell{};

        const auto& box_sizes = get_random_simulation_box_size();
        const auto& min = std::get<0>(box_sizes);
        const auto& max = std::get<1>(box_sizes);

        cell.set_size(min, max);

        std::uniform_real_distribution urd_x(min.get_x(), max.get_x());
        std::uniform_real_distribution urd_y(min.get_y(), max.get_y());
        std::uniform_real_distribution urd_z(min.get_z(), max.get_z());

        const Vec3d pos_ex_1{ urd_x(mt), urd_y(mt), urd_z(mt) };
        cell.set_neuron_position_exc(pos_ex_1);

        EXPECT_TRUE(cell.get_neuron_position_exc().has_value());
        EXPECT_EQ(pos_ex_1, cell.get_neuron_position_exc().value());

        EXPECT_TRUE(cell.get_neuron_position_for(SignalType::EXCITATORY).has_value());
        EXPECT_EQ(pos_ex_1, cell.get_neuron_position_for(SignalType::EXCITATORY).value());

        cell.set_neuron_position_exc({});
        EXPECT_FALSE(cell.get_neuron_position_exc().has_value());
        EXPECT_FALSE(cell.get_neuron_position_for(SignalType::EXCITATORY).has_value());

        const Vec3d pos_ex_2{ urd_x(mt), urd_y(mt), urd_z(mt) };
        cell.set_neuron_position_exc(pos_ex_2);

        EXPECT_TRUE(cell.get_neuron_position_exc().has_value());
        EXPECT_EQ(pos_ex_2, cell.get_neuron_position_exc().value());

        EXPECT_TRUE(cell.get_neuron_position_for(SignalType::EXCITATORY).has_value());
        EXPECT_EQ(pos_ex_2, cell.get_neuron_position_for(SignalType::EXCITATORY).value());

        cell.set_neuron_position_exc({});
        EXPECT_FALSE(cell.get_neuron_position_exc().has_value());
        EXPECT_FALSE(cell.get_neuron_position_for(SignalType::EXCITATORY).has_value());

        const Vec3d pos_in_1{ urd_x(mt), urd_y(mt), urd_z(mt) };
        cell.set_neuron_position_inh(pos_in_1);

        EXPECT_TRUE(cell.get_neuron_position_inh().has_value());
        EXPECT_EQ(pos_in_1, cell.get_neuron_position_inh().value());

        EXPECT_TRUE(cell.get_neuron_position_for(SignalType::INHIBITORY).has_value());
        EXPECT_EQ(pos_in_1, cell.get_neuron_position_for(SignalType::INHIBITORY).value());

        cell.set_neuron_position_inh({});
        EXPECT_FALSE(cell.get_neuron_position_exc().has_value());
        EXPECT_FALSE(cell.get_neuron_position_for(SignalType::EXCITATORY).has_value());

        const Vec3d pos_in_2{ urd_x(mt), urd_y(mt), urd_z(mt) };
        cell.set_neuron_position_inh(pos_in_2);

        EXPECT_TRUE(cell.get_neuron_position_inh().has_value());
        EXPECT_EQ(pos_in_2, cell.get_neuron_position_inh().value());

        EXPECT_TRUE(cell.get_neuron_position_for(SignalType::INHIBITORY).has_value());
        EXPECT_EQ(pos_in_2, cell.get_neuron_position_for(SignalType::INHIBITORY).value());

        cell.set_neuron_position_inh({});
        EXPECT_FALSE(cell.get_neuron_position_exc().has_value());
        EXPECT_FALSE(cell.get_neuron_position_for(SignalType::EXCITATORY).has_value());
    }
}

TEST(TestCell, testCellPositionException) {
    setup();

    for (auto i = 0; i < iterations; i++) {
        Cell cell{};

        const auto& box_sizes = get_random_simulation_box_size();
        const auto& min = std::get<0>(box_sizes);
        const auto& max = std::get<1>(box_sizes);

        cell.set_size(min, max);

        std::uniform_real_distribution urd_x(min.get_x(), max.get_x());
        std::uniform_real_distribution urd_y(min.get_y(), max.get_y());
        std::uniform_real_distribution urd_z(min.get_z(), max.get_z());

        const Vec3d pos_ex_1{ urd_x(mt), urd_y(mt), urd_z(mt) };
        cell.set_neuron_position_exc(pos_ex_1);

        const Vec3d pos_ex_1_invalid_x_max = max + Vec3d{ 1, 0, 0 };
        const Vec3d pos_ex_1_invalid_y_max = max + Vec3d{ 0, 1, 0 };
        const Vec3d pos_ex_1_invalid_z_max = max + Vec3d{ 0, 0, 1 };

        EXPECT_THROW(cell.set_neuron_position_exc(pos_ex_1_invalid_x_max), RelearnException);
        EXPECT_THROW(cell.set_neuron_position_exc(pos_ex_1_invalid_y_max), RelearnException);
        EXPECT_THROW(cell.set_neuron_position_exc(pos_ex_1_invalid_z_max), RelearnException);

        const Vec3d pos_ex_1_invalid_x_min = min - Vec3d{ 1, 0, 0 };
        const Vec3d pos_ex_1_invalid_y_min = min - Vec3d{ 0, 1, 0 };
        const Vec3d pos_ex_1_invalid_z_min = min - Vec3d{ 0, 0, 1 };

        EXPECT_THROW(cell.set_neuron_position_exc(pos_ex_1_invalid_x_min), RelearnException);
        EXPECT_THROW(cell.set_neuron_position_exc(pos_ex_1_invalid_y_min), RelearnException);
        EXPECT_THROW(cell.set_neuron_position_exc(pos_ex_1_invalid_z_min), RelearnException);

        EXPECT_TRUE(cell.get_neuron_position_exc().has_value());
        EXPECT_EQ(pos_ex_1, cell.get_neuron_position_exc().value());

        EXPECT_TRUE(cell.get_neuron_position_for(SignalType::EXCITATORY).has_value());
        EXPECT_EQ(pos_ex_1, cell.get_neuron_position_for(SignalType::EXCITATORY).value());

        const Vec3d pos_ex_2{ urd_x(mt), urd_y(mt), urd_z(mt) };
        cell.set_neuron_position_exc(pos_ex_2);

        const Vec3d pos_ex_2_invalid_x_max = max + Vec3d{ 1, 0, 0 };
        const Vec3d pos_ex_2_invalid_y_max = max + Vec3d{ 0, 1, 0 };
        const Vec3d pos_ex_2_invalid_z_max = max + Vec3d{ 0, 0, 1 };

        EXPECT_THROW(cell.set_neuron_position_exc(pos_ex_2_invalid_x_max), RelearnException);
        EXPECT_THROW(cell.set_neuron_position_exc(pos_ex_2_invalid_y_max), RelearnException);
        EXPECT_THROW(cell.set_neuron_position_exc(pos_ex_2_invalid_z_max), RelearnException);

        const Vec3d pos_ex_2_invalid_x_min = min - Vec3d{ 1, 0, 0 };
        const Vec3d pos_ex_2_invalid_y_min = min - Vec3d{ 0, 1, 0 };
        const Vec3d pos_ex_2_invalid_z_min = min - Vec3d{ 0, 0, 1 };

        EXPECT_THROW(cell.set_neuron_position_exc(pos_ex_2_invalid_x_min), RelearnException);
        EXPECT_THROW(cell.set_neuron_position_exc(pos_ex_2_invalid_y_min), RelearnException);
        EXPECT_THROW(cell.set_neuron_position_exc(pos_ex_2_invalid_z_min), RelearnException);

        EXPECT_TRUE(cell.get_neuron_position_exc().has_value());
        EXPECT_EQ(pos_ex_2, cell.get_neuron_position_exc().value());

        EXPECT_TRUE(cell.get_neuron_position_for(SignalType::EXCITATORY).has_value());
        EXPECT_EQ(pos_ex_2, cell.get_neuron_position_for(SignalType::EXCITATORY).value());

        const Vec3d pos_in_1{ urd_x(mt), urd_y(mt), urd_z(mt) };
        cell.set_neuron_position_inh(pos_in_1);

        const Vec3d pos_in_1_invalid_x_max = max + Vec3d{ 1, 0, 0 };
        const Vec3d pos_in_1_invalid_y_max = max + Vec3d{ 0, 1, 0 };
        const Vec3d pos_in_1_invalid_z_max = max + Vec3d{ 0, 0, 1 };

        EXPECT_THROW(cell.set_neuron_position_inh(pos_in_1_invalid_x_max), RelearnException);
        EXPECT_THROW(cell.set_neuron_position_inh(pos_in_1_invalid_y_max), RelearnException);
        EXPECT_THROW(cell.set_neuron_position_inh(pos_in_1_invalid_z_max), RelearnException);

        const Vec3d pos_in_1_invalid_x_min = min - Vec3d{ 1, 0, 0 };
        const Vec3d pos_in_1_invalid_y_min = min - Vec3d{ 0, 1, 0 };
        const Vec3d pos_in_1_invalid_z_min = min - Vec3d{ 0, 0, 1 };

        EXPECT_THROW(cell.set_neuron_position_inh(pos_in_1_invalid_x_min), RelearnException);
        EXPECT_THROW(cell.set_neuron_position_inh(pos_in_1_invalid_y_min), RelearnException);
        EXPECT_THROW(cell.set_neuron_position_inh(pos_in_1_invalid_z_min), RelearnException);

        EXPECT_TRUE(cell.get_neuron_position_inh().has_value());
        EXPECT_EQ(pos_in_1, cell.get_neuron_position_inh().value());

        EXPECT_TRUE(cell.get_neuron_position_for(SignalType::INHIBITORY).has_value());
        EXPECT_EQ(pos_in_1, cell.get_neuron_position_for(SignalType::INHIBITORY).value());

        const Vec3d pos_in_2{ urd_x(mt), urd_y(mt), urd_z(mt) };
        cell.set_neuron_position_inh(pos_in_2);

        const Vec3d pos_in_2_invalid_x_max = max + Vec3d{ 1, 0, 0 };
        const Vec3d pos_in_2_invalid_y_max = max + Vec3d{ 0, 1, 0 };
        const Vec3d pos_in_2_invalid_z_max = max + Vec3d{ 0, 0, 1 };

        EXPECT_THROW(cell.set_neuron_position_inh(pos_in_2_invalid_x_max), RelearnException);
        EXPECT_THROW(cell.set_neuron_position_inh(pos_in_2_invalid_y_max), RelearnException);
        EXPECT_THROW(cell.set_neuron_position_inh(pos_in_2_invalid_z_max), RelearnException);

        const Vec3d pos_in_2_invalid_x_min = min - Vec3d{ 1, 0, 0 };
        const Vec3d pos_in_2_invalid_y_min = min - Vec3d{ 0, 1, 0 };
        const Vec3d pos_in_2_invalid_z_min = min - Vec3d{ 0, 0, 1 };

        EXPECT_THROW(cell.set_neuron_position_inh(pos_in_2_invalid_x_min), RelearnException);
        EXPECT_THROW(cell.set_neuron_position_inh(pos_in_2_invalid_y_min), RelearnException);
        EXPECT_THROW(cell.set_neuron_position_inh(pos_in_2_invalid_z_min), RelearnException);

        EXPECT_TRUE(cell.get_neuron_position_inh().has_value());
        EXPECT_EQ(pos_in_2, cell.get_neuron_position_inh().value());

        EXPECT_TRUE(cell.get_neuron_position_for(SignalType::INHIBITORY).has_value());
        EXPECT_EQ(pos_in_2, cell.get_neuron_position_for(SignalType::INHIBITORY).value());
    }
}

TEST(TestCell, testCellPositionCombined) {
    setup();

    for (auto i = 0; i < iterations; i++) {
        Cell cell{};

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

        EXPECT_FALSE(cell.get_neuron_position().has_value());

        cell.set_neuron_position_exc(pos_1);
        cell.set_neuron_position_inh(pos_1);

        EXPECT_TRUE(cell.get_neuron_position().has_value());
        EXPECT_EQ(cell.get_neuron_position().value(), pos_1);

        cell.set_neuron_position_exc({});
        cell.set_neuron_position_inh({});

        EXPECT_FALSE(cell.get_neuron_position().has_value());

        cell.set_neuron_position_exc(pos_2);

        EXPECT_THROW(cell.get_neuron_position(), RelearnException);

        cell.set_neuron_position_inh(pos_3);

        if (pos_2 == pos_3) {
            EXPECT_TRUE(cell.get_neuron_position().has_value());
            EXPECT_EQ(cell.get_neuron_position().value(), pos_2);
        } else {
            EXPECT_THROW(cell.get_neuron_position(), RelearnException);
        }

        cell.set_neuron_position({});

        EXPECT_FALSE(cell.get_neuron_position().has_value());

        cell.set_neuron_position_exc(pos_4);
        cell.set_neuron_position_inh(pos_4);

        EXPECT_TRUE(cell.get_neuron_position().has_value());
        EXPECT_EQ(cell.get_neuron_position().value(), pos_4);
    }
}

TEST(TestCell, testCellSetNumDendrites) {
    setup();

    std::uniform_int_distribution<unsigned int> uid(0, 1000);

    for (auto i = 0; i < iterations; i++) {
        Cell cell{};

        const auto num_dends_ex_1 = uid(mt);
        const auto num_dends_in_1 = uid(mt);

        cell.set_neuron_num_dendrites_exc(num_dends_ex_1);
        cell.set_neuron_num_dendrites_inh(num_dends_in_1);

        EXPECT_EQ(num_dends_ex_1, cell.get_neuron_num_dendrites_exc());
        EXPECT_EQ(num_dends_ex_1, cell.get_neuron_num_dendrites_for(SignalType::EXCITATORY));
        EXPECT_EQ(num_dends_in_1, cell.get_neuron_num_dendrites_inh());
        EXPECT_EQ(num_dends_in_1, cell.get_neuron_num_dendrites_for(SignalType::INHIBITORY));

        const auto num_dends_ex_2 = uid(mt);
        const auto num_dends_in_2 = uid(mt);

        cell.set_neuron_num_dendrites_exc(num_dends_ex_2);
        cell.set_neuron_num_dendrites_inh(num_dends_in_2);

        EXPECT_EQ(num_dends_ex_2, cell.get_neuron_num_dendrites_exc());
        EXPECT_EQ(num_dends_ex_2, cell.get_neuron_num_dendrites_for(SignalType::EXCITATORY));
        EXPECT_EQ(num_dends_in_2, cell.get_neuron_num_dendrites_inh());
        EXPECT_EQ(num_dends_in_2, cell.get_neuron_num_dendrites_for(SignalType::INHIBITORY));
    }
}

TEST(TestCell, testCellSetNeuronId) {
    setup();

    std::uniform_int_distribution<size_t> uid(0, 1000);

    for (auto i = 0; i < iterations; i++) {
        Cell cell{};

        const auto neuron_id_1 = uid(mt);
        cell.set_neuron_id(neuron_id_1);
        EXPECT_EQ(neuron_id_1, cell.get_neuron_id());

        const auto neuron_id_2 = uid(mt);
        cell.set_neuron_id(neuron_id_2);
        EXPECT_EQ(neuron_id_2, cell.get_neuron_id());
    }
}

TEST(TestCell, testCellOctants) {
    setup();

    for (auto i = 0; i < iterations; i++) {
        Cell cell{};

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

            EXPECT_EQ(expected_octant_idx, received_idx);
        }
    }
}

TEST(TestCell, testCellOctantsException) {
    setup();

    for (auto i = 0; i < iterations; i++) {
        Cell cell{};

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

        EXPECT_THROW(cell.get_octant_for_position(pos_invalid_x_max), RelearnException);
        EXPECT_THROW(cell.get_octant_for_position(pos_invalid_y_max), RelearnException);
        EXPECT_THROW(cell.get_octant_for_position(pos_invalid_z_max), RelearnException);
        EXPECT_THROW(cell.get_octant_for_position(pos_invalid_x_min), RelearnException);
        EXPECT_THROW(cell.get_octant_for_position(pos_invalid_y_min), RelearnException);
        EXPECT_THROW(cell.get_octant_for_position(pos_invalid_z_min), RelearnException);
    }
}

TEST(TestCell, testCellOctantsSize) {
    setup();

    for (auto i = 0; i < iterations; i++) {
        Cell cell{};

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

            EXPECT_NEAR(diff_subcell_min.calculate_p_norm(2), 0.0, eps);
            EXPECT_NEAR(diff_subcell_max.calculate_p_norm(2), 0.0, eps);
        }
    }
}

TEST(TestOctreeNode, testOctreeNodeReset) {
    setup();

    OctreeNode node{};

    EXPECT_FALSE(node.is_parent());
    EXPECT_TRUE(node.get_level() == Constants::uninitialized);
    EXPECT_TRUE(node.get_rank() == -1);
    EXPECT_TRUE(node.get_children().size() == Constants::number_oct);

    const auto& children = node.get_children();

    for (auto i = 0; i < Constants::number_oct; i++) {
        EXPECT_TRUE(node.get_child(i) == nullptr);
        EXPECT_TRUE(children[i] == nullptr);
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

        EXPECT_FALSE(node.is_parent());
        EXPECT_TRUE(node.get_level() == Constants::uninitialized);
        EXPECT_TRUE(node.get_rank() == -1);
        EXPECT_TRUE(node.get_children().size() == Constants::number_oct);

        const auto& children = node.get_children();

        for (auto i = 0; i < Constants::number_oct; i++) {
            EXPECT_TRUE(node.get_child(i) == nullptr);
            EXPECT_TRUE(children[i] == nullptr);
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

        EXPECT_THROW(node.set_rank(-rank), RelearnException);
        EXPECT_THROW(node.set_level(lvl + Constants::uninitialized), RelearnException);

        EXPECT_TRUE(node.is_parent());
        EXPECT_TRUE(node.get_level() == lvl);
        EXPECT_TRUE(node.get_rank() == rank);
        EXPECT_TRUE(node.get_children().size() == Constants::number_oct);

        const auto& children = node.get_children();

        for (auto i = 0; i < Constants::number_oct; i++) {
            EXPECT_TRUE(node.get_child(i) == &(other_nodes[i]));
            EXPECT_TRUE(children[i] == &(other_nodes[i]));
        }

        const auto lb = -uid_rank(mt);
        const auto ub = uid_rank(mt);

        for (auto i = lb; i < ub; i++) {
            if (i >= 0 && i < Constants::number_oct) {
                continue;
            }

            EXPECT_THROW(node.set_child(nullptr, i), RelearnException);
            EXPECT_THROW(node.set_child(&node, i), RelearnException);
            EXPECT_THROW(node.get_child(i), RelearnException);
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
                EXPECT_TRUE(node.is_local());
            } else {
                EXPECT_FALSE(node.is_local());
            }
        }
    }
}

TEST(TestOctreeNode, testOctreeNodeSetterCell) {
    setup();

    OctreeNode node{};

    const Cell& cell = node.get_cell();

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
        node.set_cell_num_dendrites(dends_ex, dends_in);
        node.set_cell_size(min, max);
        node.set_cell_neuron_pos_exc(pos_ex);
        node.set_cell_neuron_pos_inh(pos_in);

        EXPECT_TRUE(node.get_cell().get_neuron_id() == id);
        EXPECT_TRUE(cell.get_neuron_id() == id);

        EXPECT_TRUE(node.get_cell().get_neuron_num_dendrites_exc() == dends_ex);
        EXPECT_TRUE(cell.get_neuron_num_dendrites_exc() == dends_ex);

        EXPECT_TRUE(node.get_cell().get_neuron_num_dendrites_inh() == dends_in);
        EXPECT_TRUE(cell.get_neuron_num_dendrites_inh() == dends_in);

        EXPECT_TRUE(node.get_cell().get_size() == box_sizes);
        EXPECT_TRUE(cell.get_size() == box_sizes);

        EXPECT_TRUE(node.get_cell().get_neuron_position_exc().has_value());
        EXPECT_TRUE(cell.get_neuron_position_exc().has_value());

        EXPECT_TRUE(node.get_cell().get_neuron_position_exc().value() == pos_ex);
        EXPECT_TRUE(cell.get_neuron_position_exc().value() == pos_ex);

        EXPECT_TRUE(node.get_cell().get_neuron_position_inh().has_value());
        EXPECT_TRUE(cell.get_neuron_position_inh().has_value());

        EXPECT_TRUE(node.get_cell().get_neuron_position_inh().value() == pos_in);
        EXPECT_TRUE(cell.get_neuron_position_inh().value() == pos_in);

        node.set_cell_neuron_pos_exc({});
        node.set_cell_neuron_pos_inh({});

        EXPECT_FALSE(node.get_cell().get_neuron_position_exc().has_value());
        EXPECT_FALSE(cell.get_neuron_position_exc().has_value());

        EXPECT_FALSE(node.get_cell().get_neuron_position_inh().has_value());
        EXPECT_FALSE(cell.get_neuron_position_inh().has_value());

    }
}
