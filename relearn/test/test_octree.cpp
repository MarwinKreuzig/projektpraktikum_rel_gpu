#include "../googletest/include/gtest/gtest.h"

#include "commons.h"

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
