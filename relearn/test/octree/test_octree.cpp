#include "test_octree.h"

#include "algorithm/Algorithms.h"
#include "algorithm/Cells.h"
#include "neurons/models/SynapticElements.h"
#include "structure/Cell.h"
#include "structure/Partition.h"
#include "structure/Octree.h"
#include "util/RelearnException.h"
#include "util/Vec3.h"

#include <algorithm>
#include <map>
#include <numeric>
#include <random>
#include <stack>
#include <tuple>
#include <vector>

using test_types = ::testing::Types<BarnesHutCell, BarnesHutInvertedCell, FastMultipoleMethodsCell, NaiveCell>;
TYPED_TEST_SUITE(OctreeTest, test_types);

TYPED_TEST(OctreeTest, testOctreeNodeReset) {
    using AdditionalCellAttributes = TypeParam;

    OctreeNode<AdditionalCellAttributes> node{};

    ASSERT_FALSE(node.is_parent());
    ASSERT_TRUE(node.get_rank() == -1);
    ASSERT_TRUE(node.get_children().size() == Constants::number_oct);

    const auto& children = node.get_children();

    for (auto i = 0; i < Constants::number_oct; i++) {
        ASSERT_TRUE(node.get_child(i) == nullptr);
        ASSERT_TRUE(children[i] == nullptr);
    }

    node.set_parent();

    const auto number_ranks = this->get_random_number_ranks();
    const auto rank = this->get_random_rank(number_ranks);

    node.set_rank(rank);

    std::array<OctreeNode<AdditionalCellAttributes>, Constants::number_oct> other_nodes{};
    for (auto i = 0; i < Constants::number_oct; i++) {
        node.set_child(&(other_nodes[i]), i);
    }

    node.reset();

    ASSERT_FALSE(node.is_parent());
    ASSERT_TRUE(node.get_rank() == -1);
    ASSERT_TRUE(node.get_children().size() == Constants::number_oct);

    const auto& new_children = node.get_children();

    for (auto i = 0; i < Constants::number_oct; i++) {
        ASSERT_TRUE(node.get_child(i) == nullptr);
        ASSERT_TRUE(new_children[i] == nullptr);
    }
}

TYPED_TEST(OctreeTest, testOctreeNodeSetterGetter) {
    using AdditionalCellAttributes = TypeParam;

    OctreeNode<AdditionalCellAttributes> node{};

    node.set_parent();

    const auto number_ranks = this->get_random_number_ranks();
    const auto rank = this->get_random_rank(number_ranks);

    node.set_rank(rank);

    std::array<OctreeNode<AdditionalCellAttributes>, Constants::number_oct> other_nodes{};
    for (auto i = 0; i < Constants::number_oct; i++) {
        node.set_child(&(other_nodes[i]), i);
    }

    ASSERT_THROW(node.set_rank(-rank - 1), RelearnException) << rank;

    ASSERT_TRUE(node.is_parent());
    ASSERT_TRUE(node.get_rank() == rank);
    ASSERT_TRUE(node.get_children().size() == Constants::number_oct);

    const auto& children = node.get_children();

    for (auto i = 0; i < Constants::number_oct; i++) {
        ASSERT_TRUE(node.get_child(i) == &(other_nodes[i]));
        ASSERT_TRUE(children[i] == &(other_nodes[i]));
    }

    const auto ub = Constants::number_oct * 100 + 100;

    for (auto i = 0; i < ub; i++) {
        if (i < Constants::number_oct) {
            continue;
        }

        ASSERT_THROW(node.set_child(nullptr, i), RelearnException);
        ASSERT_THROW(node.set_child(&node, i), RelearnException);
        ASSERT_THROW(auto tmp = node.get_child(i), RelearnException);
    }

    this->template make_mpi_mem_available<AdditionalCellAttributes>();
}

TYPED_TEST(OctreeTest, testOctreeNodeLocal) {
    using AdditionalCellAttributes = TypeParam;

    OctreeNode<AdditionalCellAttributes> node{};
    const auto my_rank = MPIWrapper::get_my_rank();

    for (auto i = 0; i < 1000; i++) {
        node.set_rank(i);

        if (i == my_rank) {
            ASSERT_TRUE(node.is_local());
        } else {
            ASSERT_FALSE(node.is_local());
        }
    }

    this->template make_mpi_mem_available<AdditionalCellAttributes>();
}

TYPED_TEST(OctreeTest, testOctreeNodeInsert) {
    using AdditionalCellAttributes = TypeParam;

    const auto my_rank = MPIWrapper::get_my_rank();

    const auto& [min, max] = this->get_random_simulation_box_size();
    const auto& own_position = this->get_random_position_in_box(min, max);
    size_t level = this->get_small_refinement_level();

    OctreeNode<AdditionalCellAttributes> node{};
    node.set_level(0);
    node.set_rank(my_rank);
    node.set_cell_size(min, max);
    node.set_cell_neuron_id(NeuronID::virtual_id());
    node.set_cell_neuron_position(own_position);

    size_t number_neurons = this->get_random_number_neurons();
    size_t num_additional_ids = this->get_random_number_neurons();

    std::vector<std::tuple<Vec3d, NeuronID>> neurons_to_place = this->generate_random_neurons(min, max, number_neurons, number_neurons + num_additional_ids);

    for (const auto& [pos, id] : neurons_to_place) {
        auto tmp = node.insert(pos, id);
    }

    std::vector<std::tuple<Vec3d, NeuronID>> placed_neurons = this->template extract_neurons<AdditionalCellAttributes>(&node);

    std::sort(neurons_to_place.begin(), neurons_to_place.end(), [](std::tuple<Vec3d, NeuronID> a, std::tuple<Vec3d, NeuronID> b) { return std::get<1>(a) > std::get<1>(b); });
    std::sort(placed_neurons.begin(), placed_neurons.end(), [](std::tuple<Vec3d, NeuronID> a, std::tuple<Vec3d, NeuronID> b) { return std::get<1>(a) > std::get<1>(b); });

    ASSERT_EQ(neurons_to_place.size(), placed_neurons.size());

    for (auto i = 0; i < neurons_to_place.size(); i++) {
        const auto& expected_neuron = neurons_to_place[i];
        const auto& found_neuron = placed_neurons[i];

        ASSERT_EQ(expected_neuron, found_neuron);
    }

    this->template make_mpi_mem_available<AdditionalCellAttributes>();
}

TYPED_TEST(OctreeTest, testOctreeConstructor) {
    using AdditionalCellAttributes = TypeParam;

    const auto& [min, max] = this->get_random_simulation_box_size();
    const auto level_of_branch_nodes = this->get_small_refinement_level();

    OctreeImplementation<TypeParam> octree(min, max, level_of_branch_nodes);

    ASSERT_EQ(octree.get_level_of_branch_nodes(), level_of_branch_nodes);
    ASSERT_EQ(octree.get_xyz_min(), min);
    ASSERT_EQ(octree.get_xyz_max(), max);

    const auto virtual_neurons = this->extract_virtual_neurons(octree.get_root());

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

    this->template make_mpi_mem_available<AdditionalCellAttributes>();
}

TYPED_TEST(OctreeTest, testOctreeConstructorExceptions) {
    using AdditionalCellAttributes = TypeParam;

    const auto& [min, max] = this->get_random_simulation_box_size();
    const auto level_of_branch_nodes = this->get_small_refinement_level();

    ASSERT_THROW(OctreeImplementation<TypeParam> octree(max, min, level_of_branch_nodes), RelearnException);

    this->template make_mpi_mem_available<AdditionalCellAttributes>();
}

TYPED_TEST(OctreeTest, testOctreeInsertNeurons) {
    using AdditionalCellAttributes = TypeParam;

    const auto& [min, max] = this->get_random_simulation_box_size();
    const auto level_of_branch_nodes = this->get_small_refinement_level();

    OctreeImplementation<TypeParam> octree(min, max, level_of_branch_nodes);

    size_t number_neurons = this->get_random_number_neurons();
    size_t num_additional_ids = this->get_random_number_neurons();

    std::vector<std::tuple<Vec3d, NeuronID>> neurons_to_place = this->generate_random_neurons(min, max, number_neurons, number_neurons + num_additional_ids);

    const auto my_rank = MPIWrapper::get_my_rank();

    for (const auto& [position, id] : neurons_to_place) {
        octree.insert(position, id);
    }

    std::vector<std::tuple<Vec3d, NeuronID>> placed_neurons = this->template extract_neurons_tree<TypeParam>(octree);

    ASSERT_EQ(neurons_to_place.size(), placed_neurons.size());

    std::sort(neurons_to_place.begin(), neurons_to_place.end(), [](std::tuple<Vec3d, NeuronID> a, std::tuple<Vec3d, NeuronID> b) { return std::get<1>(a) > std::get<1>(b); });
    std::sort(placed_neurons.begin(), placed_neurons.end(), [](std::tuple<Vec3d, NeuronID> a, std::tuple<Vec3d, NeuronID> b) { return std::get<1>(a) > std::get<1>(b); });

    for (auto i = 0; i < neurons_to_place.size(); i++) {
        const auto& expected_neuron = neurons_to_place[i];
        const auto& found_neuron = placed_neurons[i];

        ASSERT_EQ(expected_neuron, found_neuron);
    }

    this->template make_mpi_mem_available<AdditionalCellAttributes>();
}

TYPED_TEST(OctreeTest, testOctreeInsertNeuronsExceptions) {
    using AdditionalCellAttributes = TypeParam;

    const auto& [min, max] = this->get_random_simulation_box_size();
    const auto level_of_branch_nodes = this->get_small_refinement_level();

    OctreeImplementation<TypeParam> octree(min, max, level_of_branch_nodes);

    size_t number_neurons = this->get_random_number_neurons();
    size_t num_additional_ids = this->get_random_number_neurons();

    std::vector<std::tuple<Vec3d, NeuronID>> neurons_to_place = this->generate_random_neurons(min, max, number_neurons, number_neurons + num_additional_ids);

    const auto number_ranks = this->get_random_number_ranks();

    for (const auto& [position, id] : neurons_to_place) {
        const auto rank = this->get_random_rank(number_ranks);

        const Vec3d pos_invalid_x_max = max + Vec3d{ 1, 0, 0 };
        const Vec3d pos_invalid_y_max = max + Vec3d{ 0, 1, 0 };
        const Vec3d pos_invalid_z_max = max + Vec3d{ 0, 0, 1 };

        const Vec3d pos_invalid_x_min = min - Vec3d{ 1, 0, 0 };
        const Vec3d pos_invalid_y_min = min - Vec3d{ 0, 1, 0 };
        const Vec3d pos_invalid_z_min = min - Vec3d{ 0, 0, 1 };

        ASSERT_THROW(octree.insert(position, NeuronID::uninitialized_id()), RelearnException);

        ASSERT_THROW(octree.insert(pos_invalid_x_max, id), RelearnException);
        ASSERT_THROW(octree.insert(pos_invalid_y_max, id), RelearnException);
        ASSERT_THROW(octree.insert(pos_invalid_z_max, id), RelearnException);

        ASSERT_THROW(octree.insert(pos_invalid_x_min, id), RelearnException);
        ASSERT_THROW(octree.insert(pos_invalid_y_min, id), RelearnException);
        ASSERT_THROW(octree.insert(pos_invalid_z_min, id), RelearnException);
    }

    this->template make_mpi_mem_available<AdditionalCellAttributes>();
}

TYPED_TEST(OctreeTest, testOctreeStructure) {
    using AdditionalCellAttributes = TypeParam;

    const auto& [min, max] = this->get_random_simulation_box_size();
    const auto level_of_branch_nodes = this->get_small_refinement_level();

    OctreeImplementation<TypeParam> octree(min, max, level_of_branch_nodes);

    size_t number_neurons = this->get_random_number_neurons();
    size_t num_additional_ids = this->get_random_number_neurons();

    std::vector<std::tuple<Vec3d, NeuronID>> neurons_to_place = this->generate_random_neurons(min, max, number_neurons, number_neurons + num_additional_ids);

    const auto my_rank = MPIWrapper::get_my_rank();
    for (const auto& [position, id] : neurons_to_place) {
        octree.insert(position, id);
    }

    const auto root = octree.get_root();

    std::stack<std::pair<OctreeNode<AdditionalCellAttributes>*, size_t>> octree_nodes{};

    while (!octree_nodes.empty()) {
        const auto& elem = octree_nodes.top();

        OctreeNode<AdditionalCellAttributes>* current_node = elem.first;
        auto level = elem.second;

        octree_nodes.pop();
        ASSERT_EQ(level, current_node->get_level());
        ASSERT_EQ(current_node->get_rank(), my_rank);

        if (current_node->is_parent()) {
            const auto& childs = current_node->get_children();
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
            ASSERT_EQ(current_node->get_cell().get_neuron_id(), NeuronID::uninitialized_id());

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

            const auto neuron_id = cell.get_neuron_id();

            if (!neuron_id.is_initialized()) {
                ASSERT_LE(neuron_id, NeuronID{ number_neurons + num_additional_ids });
            }
        }
    }

    this->template make_mpi_mem_available<AdditionalCellAttributes>();
}

TYPED_TEST(OctreeTest, testOctreeMemoryStructure) {
    using AdditionalCellAttributes = TypeParam;

    const auto& [min, max] = this->get_random_simulation_box_size();
    const auto level_of_branch_nodes = this->get_small_refinement_level();

    OctreeImplementation<TypeParam> octree(min, max, level_of_branch_nodes);

    size_t number_neurons = this->get_random_number_neurons();
    size_t num_additional_ids = this->get_random_number_neurons();

    std::vector<std::tuple<Vec3d, NeuronID>> neurons_to_place = this->generate_random_neurons(min, max, number_neurons, number_neurons + num_additional_ids);

    const auto my_rank = MPIWrapper::get_my_rank();
    for (const auto& [position, id] : neurons_to_place) {
        octree.insert(position, id);
    }

    const auto root = octree.get_root();

    std::stack<OctreeNode<AdditionalCellAttributes>*> octree_nodes{};

    while (!octree_nodes.empty()) {
        const auto* current_node = octree_nodes.top();
        octree_nodes.pop();

        if (current_node->is_leaf()) {
            continue;
        }

        const auto& children = current_node->get_children();

        OctreeNode<AdditionalCellAttributes>* child_pointer = nullptr;
        int child_id = -1;

        for (auto i = 0; i < 8; i++) {
            const auto child = children[i];
            if (child == nullptr) {
                continue;
            }

            octree_nodes.emplace(child);

            if (child_pointer == nullptr) {
                child_pointer = child;
                child_id = i;
            }

            auto ptr = child_pointer + i - child_id;
            ASSERT_EQ(ptr, child);
        }
    }

    this->template make_mpi_mem_available<AdditionalCellAttributes>();
}

TYPED_TEST(OctreeTest, testOctreeLocalTrees) {
    using AdditionalCellAttributes = TypeParam;

    const auto& [min, max] = this->get_random_simulation_box_size();
    const auto level_of_branch_nodes = this->get_small_refinement_level();

    OctreeImplementation<TypeParam> octree(min, max, level_of_branch_nodes);

    ASSERT_EQ(octree.get_level_of_branch_nodes(), level_of_branch_nodes);
    ASSERT_EQ(octree.get_xyz_max(), max);
    ASSERT_EQ(octree.get_xyz_min(), min);

    SpaceFillingCurve<Morton> sfc(static_cast<uint8_t>(level_of_branch_nodes));
    const auto num_cells_per_dimension = 1 << level_of_branch_nodes;

    const auto& cell_length = (max - min) / num_cells_per_dimension;

    const auto& branch_nodes_extracted = this->extract_branch_nodes(octree.get_root());

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

    this->template make_mpi_mem_available<AdditionalCellAttributes>();
}

TYPED_TEST(OctreeTest, testOctreeInsertLocalTree) {
    using AdditionalCellAttributes = TypeParam;

    const auto& [min, max] = this->get_random_simulation_box_size();
    const auto level_of_branch_nodes = this->get_small_refinement_level();

    OctreeImplementation<TypeParam> octree(min, max, level_of_branch_nodes);

    size_t number_neurons = this->get_random_number_neurons();
    size_t num_additional_ids = this->get_random_number_neurons();

    std::vector<std::tuple<Vec3d, NeuronID>> neurons_to_place = this->generate_random_neurons(min, max, number_neurons, number_neurons + num_additional_ids);

    const auto my_rank = MPIWrapper::get_my_rank();
    for (const auto& [position, id] : neurons_to_place) {
        octree.insert(position, id);
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

    for (auto i = 0; i < num_local_trees; i++) {
        auto* local_tree = octree.get_local_root(i);

        const auto& [cell_min, cell_max] = local_tree->get_cell().get_size();
        const auto& position = this->get_random_position_in_box(cell_min, cell_max);

        OctreeNode<AdditionalCellAttributes> node{};
        node.set_cell_size(cell_min, cell_max);
        node.set_cell_neuron_position(position);
        node.set_rank(my_rank);

        nodes_to_save_new_local_trees[i]->set_cell_size(cell_min, cell_max);
        nodes_to_save_new_local_trees[i]->set_cell_neuron_position(position);
        nodes_to_save_new_local_trees[i]->set_rank(my_rank);

        for (auto j = 0; j < 8; j++) {
            const auto id_nodes = this->get_random_neuron_id(2000).get_neuron_id();
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
        ASSERT_EQ(local_tree->get_cell().get_neuron_position(), local_tree_saved->get_cell().get_neuron_position());
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
        ASSERT_EQ(local_tree->get_cell().get_neuron_position(), local_node->get_cell().get_neuron_position());
    }

    for (auto i = 0; i < 1000; i++) {
        delete nodes_to_refer_to[i];
    }

    for (auto i = 0; i < num_local_trees; i++) {
        delete nodes_to_save_local_trees[i];
        delete nodes_to_save_new_local_trees[i];
    }

    this->template make_mpi_mem_available<AdditionalCellAttributes>();
}

TYPED_TEST(OctreeTest, testOctreeLevel) {
    using AdditionalCellAttributes = TypeParam;

    const auto my_rank = MPIWrapper::get_my_rank();

    const auto& [min, max] = this->get_random_simulation_box_size();
    const auto& own_position = this->get_random_position_in_box(min, max);
    std::uint16_t level = this->get_small_refinement_level();

    OctreeNode<AdditionalCellAttributes> node{};
    node.set_level(level);
    node.set_rank(my_rank);
    node.set_cell_size(min, max);
    node.set_cell_neuron_id(NeuronID::virtual_id());
    node.set_cell_neuron_position(own_position);

    size_t number_neurons = this->get_random_number_neurons();
    size_t num_additional_ids = this->get_random_number_neurons();

    std::vector<std::tuple<Vec3d, NeuronID>> neurons_to_place = this->generate_random_neurons(min, max, number_neurons, number_neurons + num_additional_ids);

    for (const auto& [pos, id] : neurons_to_place) {
        auto tmp = node.insert(pos, id, my_rank);
    }

    ASSERT_EQ(node.get_level(), level);
    Stack<std::pair<const OctreeNode<AdditionalCellAttributes>*, const OctreeNode<AdditionalCellAttributes>*>> stack;
    for (const auto* child : node.get_children()) {
        if (child != nullptr) {
            stack.emplace_back(&node, child);
        }
    }

    while (!stack.empty()) {
        const auto& [parent, child] = stack.pop_back();
        const auto parent_level = parent->get_level();
        const auto child_level = child->get_level();
        ASSERT_EQ(parent_level + 1, child_level);
        if (child->is_parent()) {
            for (const auto* new_child : child->get_children()) {
                if (new_child != nullptr) {
                    stack.emplace_back(child, new_child);
                }
            }
        }
    }

    this->template make_mpi_mem_available<AdditionalCellAttributes>();
}
