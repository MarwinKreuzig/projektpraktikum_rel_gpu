// #include "gtest/gtest.h"
//
// #include "RelearnTest.hpp"
//
// #include "algorithm/Algorithms.h"
// #include "algorithm/Cells.h"
// #include "neurons/models/SynapticElements.h"
// #include "structure/Cell.h"
// #include "structure/Octree.h"
//
// #include "util/RelearnException.h"
// #include "util/Vec3.h"
//
// #include <memory>
// #include <tuple>
// #include <vector>
//
// TEST_F(BarnesHutTest, testGetterSetter) {
//     const auto& [min, max] = get_random_simulation_box_size();
//     auto octree = std::make_shared<OctreeImplementation<BarnesHutCell>>(min, max, 0);
//
//     ASSERT_NO_THROW(BarnesHut algorithm(octree););
//
//     BarnesHut algorithm(octree);
//
//     ASSERT_EQ(algorithm.get_acceptance_criterion(), BarnesHut::default_theta);
//
//     const auto random_acceptance_criterion = get_random_double(0.0, BarnesHut::max_theta);
//
//     ASSERT_NO_THROW(algorithm.set_acceptance_criterion(random_acceptance_criterion));
//     ASSERT_EQ(algorithm.get_acceptance_criterion(), random_acceptance_criterion);
//
//     make_mpi_mem_available<BarnesHutCell>();
// }
//
// TEST_F(BarnesHutTest, testGetterSetterOctreeNode) {
//     OctreeNode<BarnesHutCell> node{};
//
//     const auto& cell = node.get_cell();
//
//     const auto& [min, max] = get_random_simulation_box_size();
//
//     const auto number_neurons = get_random_number_neurons();
//     const auto id = get_random_neuron_id(number_neurons);
//     const auto dends_ex = static_cast<RelearnTypes::counter_type>(get_random_synaptic_element_count());
//     const auto dends_in = static_cast<RelearnTypes::counter_type>(get_random_synaptic_element_count());
//
//     const auto& pos_ex = get_random_position_in_box(min, max);
//     const auto& pos_in = get_random_position_in_box(min, max);
//
//     node.set_cell_neuron_id(id);
//     node.set_cell_size(min, max);
//     node.set_cell_number_dendrites(dends_ex, dends_in);
//     node.set_cell_excitatory_dendrites_position(pos_ex);
//     node.set_cell_inhibitory_dendrites_position(pos_in);
//
//     ASSERT_TRUE(node.get_cell().get_neuron_id() == id);
//     ASSERT_TRUE(cell.get_neuron_id() == id);
//
//     ASSERT_TRUE(node.get_cell().get_number_excitatory_dendrites() == dends_ex);
//     ASSERT_TRUE(cell.get_number_excitatory_dendrites() == dends_ex);
//
//     ASSERT_TRUE(node.get_cell().get_number_inhibitory_dendrites() == dends_in);
//     ASSERT_TRUE(cell.get_number_inhibitory_dendrites() == dends_in);
//
//     const auto& [min1, max1] = node.get_cell().get_size();
//     const auto& [min2, max2] = cell.get_size();
//
//     ASSERT_EQ(min, min1);
//     ASSERT_EQ(min, min2);
//     ASSERT_EQ(max, max1);
//     ASSERT_EQ(max, max2);
//
//     ASSERT_TRUE(node.get_cell().get_excitatory_dendrites_position().has_value());
//     ASSERT_TRUE(cell.get_excitatory_dendrites_position().has_value());
//
//     ASSERT_TRUE(node.get_cell().get_excitatory_dendrites_position().value() == pos_ex);
//     ASSERT_TRUE(cell.get_excitatory_dendrites_position().value() == pos_ex);
//
//     ASSERT_TRUE(node.get_cell().get_inhibitory_dendrites_position().has_value());
//     ASSERT_TRUE(cell.get_inhibitory_dendrites_position().has_value());
//
//     ASSERT_TRUE(node.get_cell().get_inhibitory_dendrites_position().value() == pos_in);
//     ASSERT_TRUE(cell.get_inhibitory_dendrites_position().value() == pos_in);
//
//     node.set_cell_excitatory_dendrites_position({});
//     node.set_cell_inhibitory_dendrites_position({});
//
//     ASSERT_FALSE(node.get_cell().get_excitatory_dendrites_position().has_value());
//     ASSERT_FALSE(cell.get_excitatory_dendrites_position().has_value());
//
//     ASSERT_FALSE(node.get_cell().get_inhibitory_dendrites_position().has_value());
//     ASSERT_FALSE(cell.get_inhibitory_dendrites_position().has_value());
//
//     make_mpi_mem_available<BarnesHutCell>();
// }
//
// TEST_F(BarnesHutTest, testUpdateFunctor) {
//     const auto number_neurons = get_random_number_neurons();
//     const auto& [min, max] = get_random_simulation_box_size();
//
//     const auto& axons = create_axons(number_neurons);
//     const auto& excitatory_dendrites = create_dendrites(number_neurons, SignalType::Excitatory);
//     const auto& inhibitory_dendrites = create_dendrites(number_neurons, SignalType::Excitatory);
//
//     std::vector<std::tuple<Vec3d, NeuronID>> neurons_to_place = generate_random_neurons(min, max, number_neurons, number_neurons);
//
//     auto octree = std::make_shared<OctreeImplementation<BarnesHutCell>>(min, max, 0);
//
//     std::map<NeuronID::value_type, Vec3d> positions{};
//     for (const auto& [position, id] : neurons_to_place) {
//         octree->insert(position, id);
//         positions[id.get_neuron_id()] = position;
//     }
//
//     octree->initializes_leaf_nodes(number_neurons);
//
//     BarnesHut barnes_hut(octree);
//     barnes_hut.set_synaptic_elements(axons, excitatory_dendrites, inhibitory_dendrites);
//
//     const auto update_status = get_update_status(number_neurons);
//
//     ASSERT_NO_THROW(barnes_hut.update_octree(update_status));
//
//     std::stack<OctreeNode<BarnesHutCell>*> stack{};
//     stack.push(octree->get_root());
//
//     while (!stack.empty()) {
//         auto* node = stack.top();
//         stack.pop();
//
//         const auto& cell = node->get_cell();
//
//         if (node->is_leaf()) {
//             const auto id = cell.get_neuron_id();
//             const auto local_id = id.get_neuron_id();
//
//             ASSERT_TRUE(cell.get_excitatory_dendrites_position().has_value());
//             ASSERT_TRUE(cell.get_inhibitory_dendrites_position().has_value());
//
//             const auto& golden_position = positions[local_id];
//
//             ASSERT_EQ(cell.get_excitatory_dendrites_position().value(), golden_position);
//             ASSERT_EQ(cell.get_inhibitory_dendrites_position().value(), golden_position);
//
//             if (update_status[local_id] == UpdateStatus::Disabled) {
//                 ASSERT_EQ(cell.get_number_excitatory_dendrites(), 0);
//                 ASSERT_EQ(cell.get_number_inhibitory_dendrites(), 0);
//             } else {
//                 const auto& golden_excitatory_dendrites = excitatory_dendrites->get_free_elements(id);
//                 const auto& golden_inhibitory_dendrites = inhibitory_dendrites->get_free_elements(id);
//
//                 ASSERT_EQ(cell.get_number_excitatory_dendrites(), golden_excitatory_dendrites);
//                 ASSERT_EQ(cell.get_number_inhibitory_dendrites(), golden_inhibitory_dendrites);
//             }
//         } else {
//             auto total_number_excitatory_dendrites = 0;
//             auto total_number_inhibitory_dendrites = 0;
//
//             Vec3d excitatory_dendrites_position = { 0, 0, 0 };
//             Vec3d inhibitory_dendrites_position = { 0, 0, 0 };
//
//             for (auto* child : node->get_children()) {
//                 if (child == nullptr) {
//                     continue;
//                 }
//
//                 const auto& child_cell = child->get_cell();
//
//                 const auto number_excitatory_dendrites = child_cell.get_number_excitatory_dendrites();
//                 const auto number_inhibitory_dendrites = child_cell.get_number_inhibitory_dendrites();
//
//                 total_number_excitatory_dendrites += number_excitatory_dendrites;
//                 total_number_inhibitory_dendrites += number_inhibitory_dendrites;
//
//                 if (number_excitatory_dendrites != 0) {
//                     const auto& opt = child_cell.get_excitatory_dendrites_position();
//                     ASSERT_TRUE(opt.has_value());
//                     const auto& position = opt.value();
//
//                     excitatory_dendrites_position += (position * number_excitatory_dendrites);
//                 }
//
//                 if (number_inhibitory_dendrites != 0) {
//                     const auto& opt = child_cell.get_inhibitory_dendrites_position();
//                     ASSERT_TRUE(opt.has_value());
//                     const auto& position = opt.value();
//
//                     inhibitory_dendrites_position += (position * number_inhibitory_dendrites);
//                 }
//
//                 stack.push(child);
//             }
//
//             ASSERT_EQ(total_number_excitatory_dendrites, cell.get_number_excitatory_dendrites());
//             ASSERT_EQ(total_number_inhibitory_dendrites, cell.get_number_inhibitory_dendrites());
//
//             if (total_number_excitatory_dendrites == 0) {
//                 ASSERT_FALSE(cell.get_excitatory_dendrites_position().has_value());
//             } else {
//                 const auto& opt = cell.get_excitatory_dendrites_position();
//                 ASSERT_TRUE(opt.has_value());
//                 const auto& position = opt.value();
//
//                 const auto& diff = (excitatory_dendrites_position / total_number_excitatory_dendrites) - position;
//                 const auto& norm = diff.calculate_2_norm();
//
//                 ASSERT_NEAR(norm, 0.0, eps);
//             }
//
//             if (total_number_inhibitory_dendrites == 0) {
//                 ASSERT_FALSE(cell.get_inhibitory_dendrites_position().has_value());
//             } else {
//                 const auto& opt = cell.get_inhibitory_dendrites_position();
//                 ASSERT_TRUE(opt.has_value());
//                 const auto& position = opt.value();
//
//                 const auto& diff = (inhibitory_dendrites_position / total_number_inhibitory_dendrites) - position;
//                 const auto& norm = diff.calculate_2_norm();
//
//                 ASSERT_NEAR(norm, 0.0, eps);
//             }
//         }
//     }
//
//     make_mpi_mem_available<BarnesHutCell>();
// }
