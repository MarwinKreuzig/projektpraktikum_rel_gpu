#include "test_octree_gpu.h"

#include "adapter/mpi/MpiRankAdapter.h"
#include "adapter/neurons/NeuronsAdapter.h"
#include "adapter/octree/OctreeAdapter.h"
#include "adapter/simulation/SimulationAdapter.h"
#include "adapter/mpi/MpiRankAdapter.h"
#include "adapter/neuron_id/NeuronIdAdapter.h"
#include "adapter/neurons/NeuronsAdapter.h"
#include "adapter/octree/OctreeAdapter.h"
#include "adapter/simulation/SimulationAdapter.h"

#include "algorithm/Algorithms.h"
#include "algorithm/Cells.h"
#include "neurons/models/SynapticElements.h"
#include "structure/Cell.h"
#include "structure/Octree.h"
#include "structure/Partition.h"
#include "util/RelearnException.h"
#include "util/Vec3.h"
#include "util/ranges/Functional.hpp"

#include <algorithm>
#include <map>
#include <numeric>
#include <random>
#include <stack>
#include <tuple>
#include <vector>

#include <range/v3/algorithm/sort.hpp>
#include <range/v3/view/map.hpp>

using test_types = ::testing::Types<BarnesHutCell, BarnesHutInvertedCell>;
TYPED_TEST_SUITE(OctreeTestGpu, test_types);

bool compare_vec(const std::vector<gpu::Vec3d>& v1, const std::vector<gpu::Vec3d>& v2) {
    if (v1.size() != v2.size()) {
        return false;
    }

    for (size_t i = 0; i < v1.size(); ++i) {
        if (v1[i] != v2[i]) {
            return false;
        }
    }

    return true;
}

const auto convert_vec_to_gpu = [](const Vec3d cpu_vec) -> gpu::Vec3d {
    return gpu::Vec3d { cpu_vec.get_x(), cpu_vec.get_y(), cpu_vec.get_z() };
};

TYPED_TEST(OctreeTestGpu, OctreeConstructTest) {

using AdditionalCellAttributes = TypeParam;
using box_size_type = RelearnTypes::box_size_type;
using OctreeCPUCopy = gpu::algorithm::OctreeCPUCopy;
using num_neurons_gpu = RelearnGPUTypes::number_neurons_type;
using num_neurons = RelearnTypes::number_neurons_type;
//
//// ================== FIRST TEST-CASE ==============================
//
////    Octree (Neuron ID, is_leaf()); X = virtual
////    (X, 0)                                                X
////    |--(5, 1)                  ___________________________|____________________________
////    |--(X, 0)                  |      |       |       |       |       |       |       |
////        |--(0, 1)              5      X       8       7       X       4       9       6
////        |--(1, 1)                     |                       |
////    |--(8, 1)                        / \                     / \
////    |--(7, 1)                       0   1                   2   3
////    |--(X, 0)
////        |--(2, 1)
////        |--(3, 1)
////    |--(4, 1)
////    |--(9, 1)
////    |--(6, 1)
//
//
OctreeImplementation<TypeParam> octree({0, 0, 0 }, {10, 10, 10 }, 1);

octree.insert(box_size_type(6, 0, 0 ) , NeuronID{false, 0} ); //vorne rechts unten
octree.insert(box_size_type(9, 0, 4 ) , NeuronID{false, 1} ); //vorne rechts unten
octree.insert(box_size_type(0, 0, 6 ) , NeuronID{false, 2} ); //vorne links oben
octree.insert(box_size_type(4, 0, 9 ) , NeuronID{false, 3} ); //vorne links oben
octree.insert(box_size_type(4, 0, 4 ) , NeuronID{false, 5} ); //vorne links unten
octree.insert(box_size_type(9, 0, 9 ) , NeuronID{false, 4} ); //vorne rechts oben
octree.insert(box_size_type(9, 6, 9 ) , NeuronID{false, 6} ); //hinten rechts oben
octree.insert(box_size_type(9, 6, 4 ) , NeuronID{false, 7} ); //hinten rechts unten
octree.insert(box_size_type(4, 6, 4 ) , NeuronID{false, 8} ); //hinten links unten
octree.insert(box_size_type(4, 6, 7 ) , NeuronID{false, 9} ); //hinten links oben


OctreeCPUCopy octree_cpu_copy(num_neurons_gpu(10), num_neurons_gpu(3));


std::vector<OctreeNode<AdditionalCellAttributes>*> nodes = {
        octree.get_root()->get_child(0),
        octree.get_root()->get_child(1)->get_child(0),
        octree.get_root()->get_child(1)->get_child(5),
        octree.get_root()->get_child(2),
        octree.get_root()->get_child(3),
        octree.get_root()->get_child(4)->get_child(0),
        octree.get_root()->get_child(4)->get_child(5),
        octree.get_root()->get_child(5),
        octree.get_root()->get_child(6),
        octree.get_root()->get_child(7),
        octree.get_root()->get_child(1),
        octree.get_root()->get_child(4),
        octree.get_root()};




for (int i = 0; i < 8; ++i) {
octree_cpu_copy.child_indices[i].resize(3);
}
octree_cpu_copy.num_children.resize(3);


octree_cpu_copy.num_free_elements_excitatory.resize(13);
octree_cpu_copy.num_free_elements_excitatory_virtual.resize(13);
octree_cpu_copy.num_free_elements_inhibitory.resize(13);
octree_cpu_copy.num_free_elements_inhibitory_virtual.resize(13);

for (int i = 0; i < 13; ++i) {

ElementType element_type_excitatory;
if (nodes[i]->has_excitatory_dendrite) {
element_type = ElementType::Dendrite;
}
else {
element_type = ElementType::Axon;
}

ElementType element_type_inhibitory;
if (nodes[i]->has_inhibitory_dendrite) {
element_type = ElementType::Dendrite;
}
else {
element_type = ElementType::Axon;
}


octree_cpu_copy.neuron_ids.push_back(nodes[i]->get_cell_neuron_id().get_neuron_id());


if(nodes[i]->get_cell_neuron_id().is_virtual() == 1) {

octree_cpu_copy.minimum_cell_position_virtual.push_back(gpu::Vec3d(
        std::get<0>(nodes[i]->get_size()).get_x(),
        std::get<0>(nodes[i]->get_size()).get_y(),
        std::get<0>(nodes[i]->get_size()).get_z()));
octree_cpu_copy.maximum_cell_position_virtual.push_back(gpu::Vec3d(
        std::get<1>(nodes[i]->get_size()).get_x(),
        std::get<1>(nodes[i]->get_size()).get_y(),
        std::get<1>(nodes[i]->get_size()).get_z()));

octree_cpu_copy.position_excitatory_element_virtual.push_back(
        convert_vec_to_gpu(nodes[i]->get_cell().get_position_for(element_type_excitatory, SignalType::Excitatory).value()));
octree_cpu_copy.position_inhibitory_element_virtual.push_back(
        convert_vec_to_gpu(nodes[i]->get_cell().get_position_for(element_type_inhibitory, SignalType::Inhibitory).value()));

octree_cpu_copy.num_free_elements_excitatory_virtual.push_back(
        nodes[i]->get_cell().get_number_elements_for(element_type_excitatory, SignalType::Excitatory));
octree_cpu_copy.num_free_elements_inhibitory_virtual.push_back(
        nodes[i]->get_cell().get_number_elements_for(element_type_inhibitory, SignalType::Inhibitory));

}
else   {

octree_cpu_copy.minimum_cell_position.push_back(gpu::Vec3d(
        std::get<0>(nodes[i]->get_size()).get_x(),
        std::get<0>(nodes[i]->get_size()).get_y(),
        std::get<0>(nodes[i]->get_size()).get_z()));
octree_cpu_copy.maximum_cell_position.push_back(gpu::Vec3d(
        std::get<1>(nodes[i]->get_size()).get_x(),
        std::get<1>(nodes[i]->get_size()).get_y(),
        std::get<1>(nodes[i]->get_size()).get_z()));

octree_cpu_copy.position_excitatory_element.push_back(
        convert_vec_to_gpu(nodes[i]->get_cell().get_position_for(element_type_excitatory, SignalType::Excitatory).value()));
octree_cpu_copy.position_inhibitory_element.push_back(
        convert_vec_to_gpu(nodes[i]->get_cell().get_position_for(element_type_inhibitory, SignalType::Inhibitory).value()));

octree_cpu_copy.num_free_elements_excitatory.push_back(
        nodes[i]->get_cell().get_number_elements_for(element_type_excitatory, SignalType::Excitatory));
octree_cpu_copy.num_free_elements_inhibitory.push_back(
        nodes[i]->get_cell().get_number_elements_for(element_type_inhibitory, SignalType::Inhibitory));
}
}



octree_cpu_copy.child_indices[0][0] = 1;
octree_cpu_copy.child_indices[1][0] = 2;

octree_cpu_copy.child_indices[0][1] = 5;
octree_cpu_copy.child_indices[1][1] = 6;

octree_cpu_copy.child_indices[0][2] = 0;
octree_cpu_copy.child_indices[1][2] = 10;
octree_cpu_copy.child_indices[2][2] = 3;
octree_cpu_copy.child_indices[3][2] = 4;
octree_cpu_copy.child_indices[4][2] = 11;
octree_cpu_copy.child_indices[5][2] = 7;
octree_cpu_copy.child_indices[6][2] = 8;
octree_cpu_copy.child_indices[7][2] = 9;

octree_cpu_copy.num_children[0] = 2;
octree_cpu_copy.num_children[1] = 2;
octree_cpu_copy.num_children[2] = 8;

auto result = octree.octree_to_octree_cpu_copy(num_neurons(10));
auto expected = octree_cpu_copy;


ASSERT_EQ(result.neuron_ids, expected.neuron_ids);

for (size_t i = 0; i < expected.child_indices.size(); ++i) {
ASSERT_EQ(result.child_indices[i], expected.child_indices[i]);
}

ASSERT_EQ(result.num_children, expected.num_children);

ASSERT_TRUE(compare_vec(result.minimum_cell_position, expected.minimum_cell_position));
ASSERT_TRUE(compare_vec(result.minimum_cell_position_virtual, expected.minimum_cell_position_virtual));

ASSERT_TRUE(compare_vec(result.maximum_cell_position, expected.maximum_cell_position));
ASSERT_TRUE(compare_vec(result.maximum_cell_position_virtual, expected.maximum_cell_position_virtual));

ASSERT_TRUE(compare_vec(result.position_excitatory_element, expected.position_excitatory_element));
ASSERT_TRUE(compare_vec(result.position_excitatory_element_virtual, expected.position_excitatory_element_virtual));

ASSERT_TRUE(compare_vec(result.position_inhibitory_element, expected.position_inhibitory_element));
ASSERT_TRUE(compare_vec(result.position_inhibitory_element_virtual, expected.position_inhibitory_element_virtual));

ASSERT_EQ(result.num_free_elements_excitatory, expected.num_free_elements_excitatory);
ASSERT_EQ(result.num_free_elements_excitatory_virtual, expected.num_free_elements_excitatory_virtual);

ASSERT_EQ(result.num_free_elements_inhibitory, expected.num_free_elements_inhibitory);
ASSERT_EQ(result.num_free_elements_inhibitory_virtual, expected.num_free_elements_inhibitory_virtual);

}

TYPED_TEST(OctreeTestGpu, OctreeConstructAndCopyTest) {
    using AdditionalCellAttributes = TypeParam;

    const auto& [min, max] = SimulationAdapter::get_random_simulation_box_size(this->mt);
    const auto level_of_branch_nodes = SimulationAdapter::get_small_refinement_level(this->mt);

    OctreeImplementation<TypeParam> octree(min, max, level_of_branch_nodes);

    size_t number_neurons = NeuronIdAdapter::get_random_number_neurons(this->mt);
    size_t num_additional_ids = NeuronIdAdapter::get_random_number_neurons(this->mt);

    std::vector<std::pair<Vec3d, NeuronID>> neurons_to_place = NeuronsAdapter::generate_random_neurons(min, max, number_neurons, number_neurons + num_additional_ids, this->mt);

    const auto my_rank = MPIWrapper::get_my_rank();
    for (const auto& [position, id] : neurons_to_place) {
        octree.insert(position, id);
    }

    octree.construct_on_gpu(neurons_to_place.size());

    const std::shared_ptr<gpu::algorithm::OctreeHandle> gpu_handle = octree.get_gpu_handle();
    gpu::algorithm::OctreeCPUCopy octree_cpu_copy(neurons_to_place.size(), gpu_handle->get_number_virtual_neurons());
    gpu_handle->copy_to_cpu(octree_cpu_copy);


    auto* root = octree.get_root();
    std::stack<const OctreeNode<AdditionalCellAttributes>*> octree_nodes_cpu{};
    octree_nodes_cpu.push(root);

    std::stack<uint64_t> octree_nodes_gpu{};

    size_t num_neurons = neurons_to_place.size();

    // assumes root is in the last index
    octree_nodes_gpu.push(num_neurons + gpu_handle->get_number_virtual_neurons() - 1);

    while (!octree_nodes_cpu.empty()) {
        const auto current_node_cpu = octree_nodes_cpu.top();
        octree_nodes_cpu.pop();

        auto current_node_gpu = octree_nodes_gpu.top();
        octree_nodes_gpu.pop();

        ElementType elem_type;
        if (Cell<AdditionalCellAttributes>::has_excitatory_dendrite)
            elem_type = ElementType::Dendrite;
        else
            elem_type = ElementType::Axon;

        if (current_node_cpu->get_cell().get_neuron_id().is_virtual() && current_node_gpu >= num_neurons) {

            gpu::Vec3d pos_ex_elem_virt = octree_cpu_copy.position_excitatory_element_virtual.at(current_node_gpu - num_neurons);
            ASSERT_EQ(Vec3d(pos_ex_elem_virt.x, pos_ex_elem_virt.y, pos_ex_elem_virt.z), current_node_cpu->get_cell().get_position_for(elem_type, SignalType::Excitatory).value());

            gpu::Vec3d pos_in_elem_virt = octree_cpu_copy.position_inhibitory_element_virtual.at(current_node_gpu - num_neurons);
            ASSERT_EQ(Vec3d(pos_in_elem_virt.x, pos_in_elem_virt.y, pos_in_elem_virt.z), current_node_cpu->get_cell().get_position_for(elem_type, SignalType::Inhibitory).value());

            RelearnTypes::counter_type num_ex_elem_virt = octree_cpu_copy.num_free_elements_excitatory_virtual.at(current_node_gpu - num_neurons);
            ASSERT_EQ(num_ex_elem_virt, current_node_cpu->get_cell().get_number_elements_for(elem_type, SignalType::Excitatory));

            RelearnTypes::counter_type num_in_elem_virt = octree_cpu_copy.num_free_elements_inhibitory_virtual.at(current_node_gpu - num_neurons);
            ASSERT_EQ(num_in_elem_virt, current_node_cpu->get_cell().get_number_elements_for(elem_type, SignalType::Inhibitory));
        }
        else if (!current_node_cpu->get_cell().get_neuron_id().is_virtual() && !current_node_gpu >= num_neurons) {

            gpu::Vec3d pos_ex_elem = octree_cpu_copy.position_excitatory_element.at(current_node_gpu);
            ASSERT_EQ(Vec3d(pos_ex_elem.x, pos_ex_elem.y, pos_ex_elem.z), current_node_cpu->get_cell().get_position_for(elem_type, SignalType::Excitatory).value());

            gpu::Vec3d pos_in_elem = octree_cpu_copy.position_inhibitory_element.at(current_node_gpu);
            ASSERT_EQ(Vec3d(pos_in_elem.x, pos_in_elem.y, pos_in_elem.z), current_node_cpu->get_cell().get_position_for(elem_type, SignalType::Inhibitory).value());

            RelearnTypes::counter_type num_ex_elem = octree_cpu_copy.num_free_elements_excitatory.at(current_node_gpu);
            ASSERT_EQ(num_ex_elem, current_node_cpu->get_cell().get_number_elements_for(elem_type, SignalType::Excitatory));

            RelearnTypes::counter_type num_in_elem = octree_cpu_copy.num_free_elements_inhibitory.at(current_node_gpu);
            ASSERT_EQ(num_in_elem, current_node_cpu->get_cell().get_number_elements_for(elem_type, SignalType::Inhibitory));
        }
        else {
            RelearnException::fail("Octree::overwrite_cpu_tree_with_gpu: GPU and CPU Octree structure differs");
        }

        // This assumes that nodes on the gpu are in the same order as on the cpu
        if (current_node_cpu->is_parent() && current_node_gpu >= num_neurons) {
            const auto& childs_cpu = current_node_cpu->get_children();
            int children_processed = 0;
            for (auto i = 0; i < 8; i++) {
                const auto child = childs_cpu[i];
                if (child != nullptr) {
                    octree_nodes_cpu.push(child);
                    octree_nodes_gpu.push(octree_cpu_copy.child_indices[children_processed].at(current_node_gpu - num_neurons));

                    children_processed++;
                }
            }

            if (children_processed != octree_cpu_copy.num_children.at(current_node_gpu - num_neurons)) {
                RelearnException::fail("Octree::overwrite_cpu_tree_with_gpu: GPU and CPU Octree structure differs");
            }
        }
    }

    if (!octree_nodes_gpu.empty())
                RelearnException::fail("Octree::overwrite_cpu_tree_with_gpu: GPU and CPU Octree structure differs");
}
