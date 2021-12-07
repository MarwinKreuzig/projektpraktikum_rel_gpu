#include "../googletest/include/gtest/gtest.h"

#include "RelearnTest.hpp"

#include "../source/algorithm/BarnesHut.h"
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
#include <vector>

using AdditionalCellAttributes = BarnesHutCell;

std::vector<std::tuple<Vec3d, size_t>> generate_random_neurons(const Vec3d& min, const Vec3d& max, size_t count, size_t max_id, std::mt19937& mt) {
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

std::vector<std::tuple<Vec3d, size_t>> extract_neurons(OctreeNode<AdditionalCellAttributes>* root) {
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

template <typename T>
std::vector<std::tuple<Vec3d, size_t>> extract_neurons(const OctreeImplementation<T>& octree) {
    std::vector<std::tuple<Vec3d, size_t>> return_value;

    const auto root = octree.get_root();
    if (root == nullptr) {
        return return_value;
    }

    return extract_neurons(root);
}

std::vector<std::tuple<Vec3d, size_t>> extract_unused_neurons(OctreeNode<AdditionalCellAttributes>* root) {
    std::vector<std::tuple<Vec3d, size_t>> return_value{};

    std::stack<std::pair<OctreeNode<AdditionalCellAttributes>*, size_t>> octree_nodes{};
    octree_nodes.emplace(root, 0);

    while (!octree_nodes.empty()) {
        const auto [current_node, level] = octree_nodes.top();
        octree_nodes.pop();

        if (current_node->get_cell().get_neuron_id() == Constants::uninitialized) {
            return_value.emplace_back(current_node->get_cell().get_dendrites_position().value(), level);
        }

        if (current_node->is_parent()) {
            const auto childs = current_node->get_children();
            for (auto i = 0; i < 8; i++) {
                const auto child = childs[i];
                if (child != nullptr) {
                    octree_nodes.emplace(child, level + 1);
                }
            }
        }
    }

    return return_value;
}

std::vector<OctreeNode<AdditionalCellAttributes>*> extract_branch_nodes(OctreeNode<AdditionalCellAttributes>* root) {
    std::vector<OctreeNode<AdditionalCellAttributes>*> return_value;

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
            return_value.emplace_back(current_node);
        }
    }

    return return_value;
}

SynapticElements create_synaptic_elements(size_t size, std::mt19937& mt, double max_free, SignalType st) {
    SynapticElements se(ElementType::DENDRITE, 0.0);

    se.init(size);

    std::uniform_real_distribution<double> urd(0, max_free);

    for (auto i = 0; i < size; i++) {
        se.set_signal_type(i, st);
        se.update_count(i, urd(mt));
    }

    return se;
}

TEST_F(OctreeTest, testOctreeNodeReset) {

    OctreeNode<AdditionalCellAttributes> node{};

    ASSERT_FALSE(node.is_parent());
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

        node.set_rank(uid_rank(mt));

        std::vector<OctreeNode<AdditionalCellAttributes>> other_nodes(Constants::number_oct);
        for (auto i = 0; i < Constants::number_oct; i++) {
            node.set_child(&(other_nodes[i]), i);
        }

        node.reset();

        ASSERT_FALSE(node.is_parent());
        ASSERT_TRUE(node.get_rank() == -1);
        ASSERT_TRUE(node.get_children().size() == Constants::number_oct);

        const auto& children = node.get_children();

        for (auto i = 0; i < Constants::number_oct; i++) {
            ASSERT_TRUE(node.get_child(i) == nullptr);
            ASSERT_TRUE(children[i] == nullptr);
        }
    }
}

TEST_F(OctreeTest, testOctreeNodeSetterGetter) {

    OctreeNode<AdditionalCellAttributes> node{};

    for (auto it = 0; it < iterations; it++) {
        node.set_parent();

        std::uniform_int_distribution<size_t> uid_level(0, 1000);
        std::uniform_int_distribution<int> uid_rank(0, 1000);

        const auto lvl = uid_level(mt);
        const auto rank = uid_rank(mt);

        node.set_rank(rank);

        std::vector<OctreeNode<AdditionalCellAttributes>> other_nodes(Constants::number_oct);
        for (auto i = 0; i < Constants::number_oct; i++) {
            node.set_child(&(other_nodes[i]), i);
        }

        if (rank != 0) {
            ASSERT_THROW(node.set_rank(-rank), RelearnException) << rank;
        }

        ASSERT_TRUE(node.is_parent());
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
            ASSERT_THROW(auto tmp = node.get_child(i), RelearnException);
        }
    }
}

TEST_F(OctreeTest, testOctreeNodeLocal) {

    OctreeNode<AdditionalCellAttributes> node{};

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

TEST_F(OctreeTest, testOctreeNodeSetterCell) {

    OctreeNode<AdditionalCellAttributes> node{};

    const Cell<AdditionalCellAttributes>& cell = node.get_cell();

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

TEST_F(OctreeTest, testOctreeNodeInsert) {

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

        OctreeNode<AdditionalCellAttributes> node{};
        node.set_rank(my_rank);
        node.set_cell_size(min, max);

        node.set_cell_neuron_position(own_position);

        size_t number_neurons = uid(mt);
        size_t num_additional_ids = uid(mt);

        std::vector<std::tuple<Vec3d, size_t>> neurons_to_place = generate_random_neurons(min, max, number_neurons, number_neurons + num_additional_ids, mt);

        for (const auto& [pos, id] : neurons_to_place) {
            auto tmp = node.insert(pos, id, my_rank);
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

TEST_F(OctreeTest, testOctreeConstructor) {

    std::uniform_int_distribution<size_t> uid(0, 6);
    std::uniform_real_distribution<double> urd_sigma(1, 10000.0);
    std::uniform_real_distribution<double> urd_theta(0.0, 1.0);

    for (auto i = 0; i < iterations; i++) {
        Vec3d min{};
        Vec3d max{};

        std::tie(min, max) = get_random_simulation_box_size();

        size_t level_of_branch_nodes = uid(mt);

        OctreeImplementation<BarnesHut> octree(min, max, level_of_branch_nodes);

        ASSERT_EQ(octree.get_level_of_branch_nodes(), level_of_branch_nodes);
        ASSERT_EQ(octree.get_xyz_max(), max);
        ASSERT_EQ(octree.get_xyz_min(), min);

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

        make_mpi_mem_available();
    }

    for (auto i = 0; i < iterations; i++) {
        Vec3d min{};
        Vec3d max{};

        std::tie(min, max) = get_random_simulation_box_size();

        size_t level_of_branch_nodes = uid(mt);

        OctreeImplementation<BarnesHut> octree(min, max, level_of_branch_nodes);

        ASSERT_EQ(octree.get_level_of_branch_nodes(), level_of_branch_nodes);
        ASSERT_EQ(octree.get_xyz_max(), max);
        ASSERT_EQ(octree.get_xyz_min(), min);

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

        make_mpi_mem_available();
    }
}

TEST_F(OctreeTest, testOctreeConstructorExceptions) {

    std::uniform_int_distribution<size_t> uid(0, 6);
    std::uniform_real_distribution<double> urd_sigma(1, 10000.0);
    std::uniform_real_distribution<double> urd_theta(0.0, 1.0);

    for (auto i = 0; i < iterations; i++) {
        Vec3d min_xyz{};
        Vec3d max_xyz{};

        std::tie(min_xyz, max_xyz) = get_random_simulation_box_size();

        size_t level_of_branch_nodes = uid(mt);

        ASSERT_THROW(OctreeImplementation<BarnesHut> octree(max_xyz, min_xyz, level_of_branch_nodes), RelearnException);

        make_mpi_mem_available();
    }
}

TEST_F(OctreeTest, testOctreeSetterGetter) {

    std::uniform_int_distribution<size_t> uid(0, 6);
    std::uniform_real_distribution<double> urd_sigma(1, 10000.0);
    std::uniform_real_distribution<double> urd_theta(0.0, 1.0);

    for (auto i = 0; i < iterations; i++) {
        Vec3d min{};
        Vec3d max{};

        std::tie(min, max) = get_random_simulation_box_size();

        size_t level_of_branch_nodes = uid(mt);

        OctreeImplementation<BarnesHut> octree(min, max, level_of_branch_nodes);

        ASSERT_EQ(octree.get_level_of_branch_nodes(), level_of_branch_nodes);
        ASSERT_EQ(octree.get_xyz_max(), max);
        ASSERT_EQ(octree.get_xyz_min(), min);

        make_mpi_mem_available();
    }
}

TEST_F(OctreeTest, testOctreeSetterGetterExceptions) {

    std::uniform_int_distribution<size_t> uid(0, 6);
    std::uniform_real_distribution<double> urd_sigma(1, 10000.0);
    std::uniform_real_distribution<double> urd_theta(0.0, 1.0);

    for (auto i = 0; i < iterations; i++) {
        Vec3d min{};
        Vec3d max{};

        std::tie(min, max) = get_random_simulation_box_size();

        size_t level_of_branch_nodes = uid(mt);

        OctreeImplementation<BarnesHut> octree(min, max, level_of_branch_nodes);

        std::tie(min, max) = get_random_simulation_box_size();

        make_mpi_mem_available();
    }

    for (auto i = 0; i < iterations; i++) {
        Vec3d min{};
        Vec3d max{};

        std::tie(min, max) = get_random_simulation_box_size();

        size_t level_of_branch_nodes = uid(mt);

        OctreeImplementation<BarnesHut> octree(min, max, level_of_branch_nodes);

        std::tie(min, max) = get_random_simulation_box_size();

        make_mpi_mem_available();
    }

    for (auto i = 0; i < iterations; i++) {
        Vec3d min{};
        Vec3d max{};

        std::tie(min, max) = get_random_simulation_box_size();

        size_t level_of_branch_nodes = uid(mt);

        OctreeImplementation<BarnesHut> octree(min, max, level_of_branch_nodes);

        std::tie(min, max) = get_random_simulation_box_size();

        make_mpi_mem_available();
    }
}

TEST_F(OctreeTest, testOctreeInsertNeurons) {

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

        OctreeImplementation<BarnesHut> octree(min, max, level_of_branch_nodes);

        size_t number_neurons = uid(mt);
        size_t num_additional_ids = uid(mt);

        std::vector<std::tuple<Vec3d, size_t>> neurons_to_place = generate_random_neurons(min, max, number_neurons, number_neurons + num_additional_ids, mt);

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

        make_mpi_mem_available();
    }
}

TEST_F(OctreeTest, testOctreeInsertNeuronsExceptions) {

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

        OctreeImplementation<BarnesHut> octree(min, max, level_of_branch_nodes);

        size_t number_neurons = uid(mt);
        size_t num_additional_ids = uid(mt);

        std::vector<std::tuple<Vec3d, size_t>> neurons_to_place = generate_random_neurons(min, max, number_neurons, number_neurons + num_additional_ids, mt);

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

        make_mpi_mem_available();
    }
}

TEST_F(OctreeTest, testOctreeStructure) {

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

        OctreeImplementation<BarnesHut> octree(min, max, level_of_branch_nodes);

        size_t number_neurons = uid(mt);
        size_t num_additional_ids = uid(mt);

        std::vector<std::tuple<Vec3d, size_t>> neurons_to_place = generate_random_neurons(min, max, number_neurons, number_neurons + num_additional_ids, mt);

        for (const auto& [position, id] : neurons_to_place) {
            octree.insert(position, id, my_rank);
        }

        const auto root = octree.get_root();

        std::stack<std::pair<OctreeNode<AdditionalCellAttributes>*, size_t>> octree_nodes{};

        while (!octree_nodes.empty()) {
            const auto& elem = octree_nodes.top();

            OctreeNode<AdditionalCellAttributes>* current_node = elem.first;
            auto level = elem.second;

            octree_nodes.pop();

            ASSERT_EQ(current_node->get_rank(), my_rank);

            if (current_node->is_parent()) {
                const auto childs = current_node->get_children();
                auto one_child_exists = false;

                for (auto i = 0; i < 8; i++) {
                    const auto child = childs[i];
                    if (child != nullptr) {
                        octree_nodes.emplace(child, level + 1);

                        const auto& subcell_size = child->get_cell().get_size();
                        const auto& expected_subcell_size = current_node->get_cell().get_size_for_octant(i);

                        ASSERT_EQ(expected_subcell_size, subcell_size);

                        one_child_exists = true;
                    }
                }

                ASSERT_TRUE(one_child_exists);
                ASSERT_EQ(current_node->get_cell().get_neuron_id(), Constants::uninitialized);

            } else {
                const auto& cell = current_node->get_cell();
                const auto& opt_position = cell.get_dendrites_position();

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

                const auto neuron_id = cell.get_neuron_id();

                if (neuron_id < Constants::uninitialized) {
                    ASSERT_LE(neuron_id, number_neurons + num_additional_ids);
                }
            }
        }

        make_mpi_mem_available();
    }
}

TEST_F(OctreeTest, testOctreeLocalTrees) {

    std::uniform_int_distribution<size_t> uid(0, 6);
    std::uniform_real_distribution<double> urd_sigma(1, 10000.0);
    std::uniform_real_distribution<double> urd_theta(0.0, 1.0);

    for (auto i = 0; i < iterations; i++) {
        Vec3d min{};
        Vec3d max{};

        std::tie(min, max) = get_random_simulation_box_size();

        size_t level_of_branch_nodes = uid(mt);

        OctreeImplementation<BarnesHut> octree(min, max, level_of_branch_nodes);

        ASSERT_EQ(octree.get_level_of_branch_nodes(), level_of_branch_nodes);
        ASSERT_EQ(octree.get_xyz_max(), max);
        ASSERT_EQ(octree.get_xyz_min(), min);

        SpaceFillingCurve<Morton> sfc(static_cast<uint8_t>(level_of_branch_nodes));
        const auto num_cells_per_dimension = 1 << level_of_branch_nodes;

        const auto& cell_length = (max - min) / num_cells_per_dimension;

        const auto& branch_nodes_extracted = extract_branch_nodes(octree.get_root());

        for (auto* branch_node : branch_nodes_extracted) {
            const auto branch_node_position = branch_node->get_cell().get_dendrites_position().value();
            const auto branch_node_offset = branch_node_position - min;

            const auto x_pos = branch_node_offset.get_x() / cell_length.get_x();
            const auto y_pos = branch_node_offset.get_y() / cell_length.get_y();
            const auto z_pos = branch_node_offset.get_z() / cell_length.get_z();

            Vec3s pos3d{ static_cast<size_t>(std::floor(x_pos)), static_cast<size_t>(std::floor(y_pos)), static_cast<size_t>(std::floor(z_pos)) };
            const auto pos1d = sfc.map_3d_to_1d(pos3d);

            const auto local_tree = octree.get_local_root(pos1d);
            ASSERT_EQ(local_tree, branch_node);
        }

        make_mpi_mem_available();
    }
}

TEST_F(OctreeTest, testOctreeInsertLocalTree) {
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

        OctreeImplementation<BarnesHut> octree(min, max, level_of_branch_nodes);

        size_t number_neurons = uid(mt);
        size_t num_additional_ids = uid(mt);

        std::vector<std::tuple<Vec3d, size_t>> neurons_to_place = generate_random_neurons(min, max, number_neurons, number_neurons + num_additional_ids, mt);

        for (const auto& [position, id] : neurons_to_place) {
            octree.insert(position, id, my_rank);
        }

        const auto num_local_trees = octree.get_num_local_trees();

        std::vector<OctreeNode<AdditionalCellAttributes>*> nodes_to_refer_to(1000);
        for (auto i = 0; i < 1000; i++) {
            nodes_to_refer_to[i] = new OctreeNode<AdditionalCellAttributes>;
        }

        std::vector<OctreeNode<AdditionalCellAttributes>*> nodes_to_save_local_trees(num_local_trees);
        std::vector<OctreeNode<AdditionalCellAttributes>*> nodes_to_save_new_local_trees(num_local_trees);
        for (auto i = 0; i < num_local_trees; i++) {
            nodes_to_save_local_trees[i] = new OctreeNode<AdditionalCellAttributes>;
            nodes_to_save_new_local_trees[i] = new OctreeNode<AdditionalCellAttributes>;
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

            OctreeNode<AdditionalCellAttributes> node{};
            node.set_cell_size(cell_min, cell_max);
            node.set_cell_neuron_position(position);
            node.set_rank(my_rank);

            nodes_to_save_new_local_trees[i]->set_cell_size(cell_min, cell_max);
            nodes_to_save_new_local_trees[i]->set_cell_neuron_position(position);
            nodes_to_save_new_local_trees[i]->set_rank(my_rank);

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
            node.set_parent();
            octree.insert_local_tree(&node, i);
        }

        for (auto i = 0; i < num_local_trees; i++) {
            auto* local_tree = octree.get_local_root(i);
            auto* local_tree_saved = nodes_to_save_new_local_trees[i];

            ASSERT_EQ(local_tree->get_children(), local_tree_saved->get_children());
            ASSERT_EQ(local_tree->get_rank(), local_tree_saved->get_rank());

            ASSERT_EQ(local_tree->get_cell().get_neuron_id(), local_tree_saved->get_cell().get_neuron_id());
            ASSERT_EQ(local_tree->get_cell().get_size(), local_tree_saved->get_cell().get_size());
            ASSERT_EQ(local_tree->get_cell().get_dendrites_position(), local_tree_saved->get_cell().get_dendrites_position());
        }

        for (auto i = 0; i < num_local_trees; i++) {
            auto* local_node = nodes_to_save_local_trees[i];
            local_node->set_parent();
            octree.insert_local_tree(local_node, i);

            auto* local_tree = octree.get_local_root(i);

            ASSERT_EQ(local_tree->get_children(), local_node->get_children());
            ASSERT_EQ(local_tree->get_rank(), local_node->get_rank());

            ASSERT_EQ(local_tree->get_cell().get_neuron_id(), local_node->get_cell().get_neuron_id());
            ASSERT_EQ(local_tree->get_cell().get_size(), local_node->get_cell().get_size());
            ASSERT_EQ(local_tree->get_cell().get_dendrites_position(), local_node->get_cell().get_dendrites_position());
        }

        for (auto i = 0; i < 1000; i++) {
            delete nodes_to_refer_to[i];
        }

        for (auto i = 0; i < num_local_trees; i++) {
            delete nodes_to_save_local_trees[i];
            delete nodes_to_save_new_local_trees[i];
        }

        make_mpi_mem_available();
    }
}

TEST_F(OctreeTest, testOctreeUpdateLocalTreesNumberDendrites) {
    const auto my_rank = MPIWrapper::get_my_rank();

    std::uniform_int_distribution<size_t> uid_lvl(0, 6);
    std::uniform_int_distribution<size_t> uid(0, 10000);
    std::uniform_real_distribution<double> urd_sigma(1, 10000.0);
    std::uniform_real_distribution<double> urd_theta(0.0, 1.0);

    std::uniform_real_distribution<double> uid_max_vacant(1.0, 100.0);

    for (auto i = 0; i < iterations; i++) {
        Vec3d min{};
        Vec3d max{};

        std::tie(min, max) = get_random_simulation_box_size();

        auto octree_ptr = std::make_shared<OctreeImplementation<BarnesHut>>(min, max, 0);
        auto& octree = *octree_ptr;

        const size_t number_neurons = uid(mt);

        const std::vector<std::tuple<Vec3d, size_t>> neurons_to_place = generate_random_neurons(min, max, number_neurons, number_neurons, mt);

        for (const auto& [position, id] : neurons_to_place) {
            octree.insert(position, id, my_rank);
        }

        octree.initializes_leaf_nodes(number_neurons);

        const auto max_vacant_exc = uid_max_vacant(mt);
        auto dends_exc = create_synaptic_elements(number_neurons, mt, max_vacant_exc, SignalType::EXCITATORY);

        const auto max_vacant_inh = uid_max_vacant(mt);
        auto dends_inh = create_synaptic_elements(number_neurons, mt, max_vacant_inh, SignalType::INHIBITORY);

        BarnesHut bh{ octree_ptr };

        std::vector<char> disable_flags(number_neurons, 1);

        auto unique_exc = std::make_unique<SynapticElements>(std::move(dends_exc));
        auto unique_inh = std::make_unique<SynapticElements>(std::move(dends_inh));

        bh.update_leaf_nodes(disable_flags, unique_exc, unique_exc, unique_inh);
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

TEST_F(OctreeTest, testOctreeUpdateLocalTreesPositionDendrites) {
    const auto my_rank = MPIWrapper::get_my_rank();

    std::uniform_int_distribution<size_t> uid_lvl(0, 6);
    std::uniform_int_distribution<size_t> uid(0, 10000);
    std::uniform_real_distribution<double> urd_sigma(1, 10000.0);
    std::uniform_real_distribution<double> urd_theta(0.0, 1.0);

    for (auto i = 0; i < iterations; i++) {
        Vec3d min{};
        Vec3d max{};

        std::tie(min, max) = get_random_simulation_box_size();

        auto octree_ptr = std::make_shared<OctreeImplementation<BarnesHut>>(min, max, 0);
        auto& octree = *octree_ptr;

        const size_t number_neurons = uid(mt);

        const std::vector<std::tuple<Vec3d, size_t>> neurons_to_place = generate_random_neurons(min, max, number_neurons, number_neurons, mt);

        for (const auto& [position, id] : neurons_to_place) {
            octree.insert(position, id, my_rank);
        }

        octree.initializes_leaf_nodes(number_neurons);

        auto dends_exc = create_synaptic_elements(number_neurons, mt, 1, SignalType::EXCITATORY);
        auto dends_inh = create_synaptic_elements(number_neurons, mt, 1, SignalType::INHIBITORY);

        auto unique_exc = std::make_unique<SynapticElements>(std::move(dends_exc));
        auto unique_inh = std::make_unique<SynapticElements>(std::move(dends_inh));

        BarnesHut bh{ octree_ptr };

        std::vector<char> disable_flags(number_neurons, 1);

        bh.update_leaf_nodes(disable_flags, unique_exc, unique_exc, unique_inh);
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
