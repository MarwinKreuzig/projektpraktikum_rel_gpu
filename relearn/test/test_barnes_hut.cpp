#include "../googletest/include/gtest/gtest.h"

#include "RelearnTest.hpp"

#include "../source/algorithm/Algorithms.h"
#include "../source/algorithm/Cells.h"
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
//
//
//SynapticElements create_synaptic_elements(size_t size, std::mt19937& mt, double max_free, SignalType st) {
//    SynapticElements se(ElementType::Dendrite, 0.0);
//
//    se.init(size);
//
//    std::uniform_real_distribution<double> urd(0, max_free);
//
//    for (auto i = 0; i < size; i++) {
//        const auto id = NeuronID{ i };
//        se.set_signal_type(id, st);
//        se.update_grown_elements(id, urd(mt));
//    }
//
//    return se;
//}
//
//TEST_F(OctreeTest, testOctreeUpdateLocalTreesNumberDendrites) {
//    const auto my_rank = MPIWrapper::get_my_rank();
//
//    std::uniform_real_distribution<double> urd_sigma(1, 10000.0);
//    std::uniform_real_distribution<double> urd_theta(0.0, 1.0);
//
//    std::uniform_real_distribution<double> uid_max_vacant(1.0, 100.0);
//
//    Vec3d min{};
//    Vec3d max{};
//
//    std::tie(min, max) = get_random_simulation_box_size();
//
//    auto octree_ptr = std::make_shared<OctreeImplementation<BarnesHut>>(min, max, 0);
//    auto& octree = *octree_ptr;
//
//    const size_t number_neurons = get_random_number_neurons();
//
//    const std::vector<std::tuple<Vec3d, NeuronID>> neurons_to_place = generate_random_neurons(min, max, number_neurons, number_neurons);
//
//    for (const auto& [position, id] : neurons_to_place) {
//        octree.insert(position, id, my_rank);
//    }
//
//    octree.initializes_leaf_nodes(number_neurons);
//
//    const auto max_vacant_exc = uid_max_vacant(mt);
//    auto dends_exc = create_synaptic_elements(number_neurons, mt, max_vacant_exc, SignalType::Excitatory);
//
//    const auto max_vacant_inh = uid_max_vacant(mt);
//    auto dends_inh = create_synaptic_elements(number_neurons, mt, max_vacant_inh, SignalType::Inhibitory);
//
//    BarnesHut bh{ octree_ptr };
//
//    std::vector<UpdateStatus> disable_flags(number_neurons, UpdateStatus::Enabled);
//
//    auto unique_exc = std::make_shared<SynapticElements>(std::move(dends_exc));
//    auto unique_inh = std::make_shared<SynapticElements>(std::move(dends_inh));
//    bh.set_synaptic_elements(unique_exc, unique_exc, unique_inh);
//    bh.update_leaf_nodes(disable_flags);
//    octree.update_local_trees();
//
//    std::stack<OctreeNode<AdditionalCellAttributes>*> stack{};
//    stack.emplace(octree.get_root());
//
//    while (!stack.empty()) {
//        const auto* current = stack.top();
//        stack.pop();
//
//        size_t sum_dends_exc = 0;
//        size_t sum_dends_inh = 0;
//
//        if (current->is_parent()) {
//            for (auto* child : current->get_children()) {
//                if (child == nullptr) {
//                    continue;
//                }
//
//                sum_dends_exc += child->get_cell().get_number_excitatory_dendrites();
//                sum_dends_inh += child->get_cell().get_number_inhibitory_dendrites();
//
//                stack.emplace(child);
//            }
//        } else {
//            sum_dends_exc = static_cast<size_t>(unique_exc->get_grown_elements(current->get_cell_neuron_id()));
//            sum_dends_inh = static_cast<size_t>(unique_inh->get_grown_elements(current->get_cell_neuron_id()));
//        }
//
//        ASSERT_EQ(current->get_cell().get_number_excitatory_dendrites(), sum_dends_exc);
//        ASSERT_EQ(current->get_cell().get_number_inhibitory_dendrites(), sum_dends_inh);
//    }
//
//    make_mpi_mem_available<AdditionalCellAttributes>();
//}
//
//TEST_F(OctreeTest, testOctreeUpdateLocalTreesPositionDendrites) {
//    const auto my_rank = MPIWrapper::get_my_rank();
//
//    std::uniform_real_distribution<double> urd_sigma(1, 10000.0);
//    std::uniform_real_distribution<double> urd_theta(0.0, 1.0);
//
//    Vec3d min{};
//    Vec3d max{};
//
//    std::tie(min, max) = get_random_simulation_box_size();
//
//    auto octree_ptr = std::make_shared<OctreeImplementation<BarnesHut>>(min, max, 0);
//    auto& octree = *octree_ptr;
//
//    const size_t number_neurons = get_random_number_neurons();
//
//    const std::vector<std::tuple<Vec3d, NeuronID>> neurons_to_place = generate_random_neurons(min, max, number_neurons, number_neurons, mt);
//
//    for (const auto& [position, id] : neurons_to_place) {
//        octree.insert(position, id, my_rank);
//    }
//
//    octree.initializes_leaf_nodes(number_neurons);
//
//    auto dends_exc = create_synaptic_elements(number_neurons, mt, 1, SignalType::Excitatory);
//    auto dends_inh = create_synaptic_elements(number_neurons, mt, 1, SignalType::Inhibitory);
//
//    auto unique_exc = std::make_shared<SynapticElements>(std::move(dends_exc));
//    auto unique_inh = std::make_shared<SynapticElements>(std::move(dends_inh));
//
//    BarnesHut bh{ octree_ptr };
//
//    std::vector<UpdateStatus> disable_flags(number_neurons, UpdateStatus::Enabled);
//
//    bh.set_synaptic_elements(unique_exc, unique_exc, unique_inh);
//    bh.update_leaf_nodes(disable_flags);
//    octree.update_local_trees();
//
//    std::stack<std::tuple<OctreeNode<AdditionalCellAttributes>*, bool, bool>> stack{};
//    const auto flag_exc = octree.get_root()->get_cell().get_number_excitatory_dendrites() != 0;
//    const auto flag_inh = octree.get_root()->get_cell().get_number_inhibitory_dendrites() != 0;
//    stack.emplace(octree.get_root(), flag_exc, flag_inh);
//
//    while (!stack.empty()) {
//        std::tuple<OctreeNode<AdditionalCellAttributes>*, bool, bool> tup = stack.top();
//        stack.pop();
//
//        auto* current = std::get<0>(tup);
//        auto has_exc = std::get<1>(tup);
//        auto has_inh = std::get<2>(tup);
//
//        Vec3d pos_dends_exc{ 0.0 };
//        Vec3d pos_dends_inh{ 0.0 };
//
//        bool changed_exc = false;
//        bool changed_inh = false;
//
//        if (current->is_parent()) {
//            double num_dends_exc = 0.0;
//            double num_dends_inh = 0.0;
//
//            for (auto* child : current->get_children()) {
//                if (child == nullptr) {
//                    continue;
//                }
//
//                const auto& cell = child->get_cell();
//
//                const auto& opt_exc = cell.get_excitatory_dendrites_position();
//                const auto& opt_inh = cell.get_inhibitory_dendrites_position();
//
//                if (!has_exc) {
//                    ASSERT_EQ(cell.get_number_excitatory_dendrites(), 0);
//                }
//
//                if (!has_inh) {
//                    ASSERT_EQ(cell.get_number_inhibitory_dendrites(), 0);
//                }
//
//                if (opt_exc.has_value() && cell.get_number_excitatory_dendrites() != 0) {
//                    changed_exc = true;
//                    pos_dends_exc += (opt_exc.value() * cell.get_number_excitatory_dendrites());
//                    num_dends_exc += cell.get_number_excitatory_dendrites();
//                }
//
//                if (opt_inh.has_value() && cell.get_number_inhibitory_dendrites() != 0) {
//                    changed_inh = true;
//                    pos_dends_inh += (opt_inh.value() * cell.get_number_inhibitory_dendrites());
//                    num_dends_inh += cell.get_number_inhibitory_dendrites();
//                }
//
//                stack.emplace(child, cell.get_number_excitatory_dendrites() != 0, cell.get_number_inhibitory_dendrites() != 0);
//            }
//
//            pos_dends_exc /= num_dends_exc;
//            pos_dends_inh /= num_dends_inh;
//
//        } else {
//            const auto& cell = current->get_cell();
//
//            const auto& opt_exc = cell.get_excitatory_dendrites_position();
//            const auto& opt_inh = cell.get_inhibitory_dendrites_position();
//
//            if (!has_exc) {
//                ASSERT_EQ(cell.get_number_excitatory_dendrites(), 0);
//            }
//
//            if (!has_inh) {
//                ASSERT_EQ(cell.get_number_inhibitory_dendrites(), 0);
//            }
//
//            if (opt_exc.has_value() && cell.get_number_excitatory_dendrites() != 0) {
//                changed_exc = true;
//                pos_dends_exc += (opt_exc.value() * cell.get_number_excitatory_dendrites());
//            }
//
//            if (opt_inh.has_value() && cell.get_number_inhibitory_dendrites() != 0) {
//                changed_inh = true;
//                pos_dends_inh += (opt_inh.value() * cell.get_number_inhibitory_dendrites());
//            }
//        }
//
//        ASSERT_EQ(has_exc, changed_exc);
//        ASSERT_EQ(has_inh, changed_inh);
//
//        if (has_exc) {
//            const auto& diff = current->get_cell().get_excitatory_dendrites_position().value() - pos_dends_exc;
//            ASSERT_NEAR(diff.get_x(), 0.0, eps);
//            ASSERT_NEAR(diff.get_y(), 0.0, eps);
//            ASSERT_NEAR(diff.get_z(), 0.0, eps);
//        }
//
//        if (has_inh) {
//            const auto& diff = current->get_cell().get_inhibitory_dendrites_position().value() - pos_dends_inh;
//            ASSERT_NEAR(diff.get_x(), 0.0, eps);
//            ASSERT_NEAR(diff.get_y(), 0.0, eps);
//            ASSERT_NEAR(diff.get_z(), 0.0, eps);
//        }
//    }
//
//    make_mpi_mem_available<AdditionalCellAttributes>();
//}


TEST_F(BarnesHutTest, testOctreeNodeSetterCell) {
    OctreeNode<BarnesHutCell> node{};

    const auto& cell = node.get_cell();

    const auto& [min, max] = get_random_simulation_box_size();

    const auto number_neurons = get_random_number_neurons();
    const auto id = get_random_neuron_id(number_neurons);
    const auto dends_ex = static_cast<RelearnTypes::counter_type>(get_random_synaptic_element_count());
    const auto dends_in = static_cast<RelearnTypes::counter_type>(get_random_synaptic_element_count());

    const auto& pos_ex = get_random_position_in_box(min, max);
    const auto& pos_in = get_random_position_in_box(min, max);

    node.set_cell_neuron_id(id);
    node.set_cell_size(min, max);
    node.set_cell_number_dendrites(dends_ex, dends_in);
    node.set_cell_excitatory_dendrites_position(pos_ex);
    node.set_cell_inhibitory_dendrites_position(pos_in);

    ASSERT_TRUE(node.get_cell().get_neuron_id() == id);
    ASSERT_TRUE(cell.get_neuron_id() == id);

    ASSERT_TRUE(node.get_cell().get_number_excitatory_dendrites() == dends_ex);
    ASSERT_TRUE(cell.get_number_excitatory_dendrites() == dends_ex);

    ASSERT_TRUE(node.get_cell().get_number_inhibitory_dendrites() == dends_in);
    ASSERT_TRUE(cell.get_number_inhibitory_dendrites() == dends_in);

    const auto& [min1, max1] = node.get_cell().get_size();
    const auto& [min2, max2] = cell.get_size();

    ASSERT_EQ(min, min1);
    ASSERT_EQ(min, min2);
    ASSERT_EQ(max, max1);
    ASSERT_EQ(max, max2);

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
