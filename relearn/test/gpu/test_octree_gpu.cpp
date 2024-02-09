#include "test_octree_gpu.h"

#include "adapter/neurons/NeuronsAdapter.h"
#include "adapter/simulation/SimulationAdapter.h"
#include "adapter/neuron_id/NeuronIdAdapter.h"

#include "algorithm/Algorithms.h"
#include "algorithm/Cells.h"
#include "structure/Cell.h"
#include "structure/Octree.h"
#include "util/RelearnException.h"
#include "util/Vec3.h"

using test_types = ::testing::Types<BarnesHutCell, BarnesHutInvertedCell>;
TYPED_TEST_SUITE(OctreeTestGpu, test_types);

const auto convert_gpu_vec_to_vec(const gpu::Vec3d gpu_vec) {
    return Vec3(gpu_vec.x, gpu_vec.y, gpu_vec.z);
}


const auto convert_vec_to_gpu_vec = [](const Vec3d cpu_vec) -> gpu::Vec3d {
    return gpu::Vec3d { cpu_vec.get_x(), cpu_vec.get_y(), cpu_vec.get_z() };
};

const auto assert_eq_vec = [](const Vec3d vec1, const Vec3d vec2) {
    ASSERT_DOUBLE_EQ(vec1.get_x(), vec2.get_x());
    ASSERT_DOUBLE_EQ(vec1.get_y(), vec2.get_y());
    ASSERT_DOUBLE_EQ(vec1.get_z(), vec2.get_z());
};


TYPED_TEST(OctreeTestGpu, OctreeConstructTest) {

    using AdditionalCellAttributes = TypeParam;
    using box_size_type = RelearnTypes::box_size_type;
    using num_neurons_gpu = RelearnGPUTypes::number_neurons_type;
    using num_neurons = RelearnTypes::number_neurons_type;

    // ================== FIRST TEST-CASE ==============================
    //
    //    Octree (Neuron ID, is_leaf()); X = virtual
    //    (X, 0)                                                X
    //    |--(5, 1)                  ___________________________|____________________________
    //    |--(X, 0)                  |      |       |       |       |       |       |       |
    //        |--(0, 1)              5      X       8       7       X       4       9       6
    //        |--(1, 1)                     |                       |
    //    |--(8, 1)                        / \                     / \
    //    |--(7, 1)                       0   1                   2   3
    //    |--(X, 0)
    //        |--(2, 1)
    //        |--(3, 1)
    //    |--(4, 1)
    //    |--(9, 1)
    //    |--(6, 1)


    num_neurons_gpu num_virtual_neurons = 3;
    num_neurons_gpu num_leafs = 10;

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


    gpu::algorithm::OctreeCPUCopy expected(num_leafs, num_virtual_neurons);

    std::vector<OctreeNode<AdditionalCellAttributes>*> nodes = {
            octree.get_root()->get_child(7),
            octree.get_root()->get_child(6),
            octree.get_root()->get_child(5),
            octree.get_root()->get_child(4)->get_child(5),
            octree.get_root()->get_child(4)->get_child(0),
            octree.get_root()->get_child(3),
            octree.get_root()->get_child(2),
            octree.get_root()->get_child(1)->get_child(5),
            octree.get_root()->get_child(1)->get_child(0),
            octree.get_root()->get_child(0),
            octree.get_root()->get_child(4),
            octree.get_root()->get_child(1),
            octree.get_root()
    };

    element_type = ElementType::Dendrite;

    int current_leaf_node_num = 0;
    int current_virtual_neuron_num = 0;
    for (int i = 0; i < nodes.size(); ++i) {


        int index = 0;

        if (!nodes[i]->get_cell_neuron_id().is_virtual()) {
            expected.neuron_ids[current_leaf_node_num] = nodes[i]->get_cell_neuron_id().get_neuron_id();
            index = current_leaf_node_num++;
        } else {
            index = num_leafs + current_virtual_neuron_num++;
        }

        expected.minimum_cell_position[index] = gpu::Vec3d(
                std::get<0>(nodes[i]->get_size()).get_x(),
                std::get<0>(nodes[i]->get_size()).get_y(),
                std::get<0>(nodes[i]->get_size()).get_z());
        expected.maximum_cell_position[index] = gpu::Vec3d(
                std::get<1>(nodes[i]->get_size()).get_x(),
                std::get<1>(nodes[i]->get_size()).get_y(),
                std::get<1>(nodes[i]->get_size()).get_z());

        expected.position_excitatory_element[index] = convert_vec_to_gpu_vec(nodes[i]->get_cell().get_position_for(element_type, SignalType::Excitatory).value());
        expected.position_inhibitory_element[index] = convert_vec_to_gpu_vec(nodes[i]->get_cell().get_position_for(element_type, SignalType::Inhibitory).value());
        expected.num_free_elements_excitatory[index] = nodes[i]->get_cell().get_number_elements_for(element_type, SignalType::Excitatory);
        expected.num_free_elements_inhibitory[index] = nodes[i]->get_cell().get_number_elements_for(element_type, SignalType::Inhibitory);
    }

    //                                                          X0
    //    |--(5, 1)                  ___________________________|____________________________
    //    |--(X, 0)                  |      |       |       |       |       |       |       |
    //        |--(0, 1)              5      X2      8       7       X1      4       9       6
    //        |--(1, 1)                     |                       |
    //    |--(8, 1)                        / \                     / \
    //    |--(7, 1)                       0   1                   2   3

    expected.child_indices[0 * num_virtual_neurons + 1] = 7;
    expected.child_indices[1 * num_virtual_neurons + 1] = 8;

    expected.child_indices[0 * num_virtual_neurons + 0] = 3;
    expected.child_indices[1 * num_virtual_neurons + 0] = 4;

    expected.child_indices[0 * num_virtual_neurons + 2] = 0;
    expected.child_indices[1 * num_virtual_neurons + 2] = 1;
    expected.child_indices[2 * num_virtual_neurons + 2] = 2;
    expected.child_indices[3 * num_virtual_neurons + 2] = 10;
    expected.child_indices[4 * num_virtual_neurons + 2] = 5;
    expected.child_indices[5 * num_virtual_neurons + 2] = 6;
    expected.child_indices[6 * num_virtual_neurons + 2] = 11;
    expected.child_indices[7 * num_virtual_neurons + 2] = 9;

    expected.num_children[0] = 2;
    expected.num_children[1] = 2;
    expected.num_children[2] = 8;

    auto result = octree.octree_to_octree_cpu_copy(num_leafs);

    ASSERT_EQ(result.neuron_ids, expected.neuron_ids);
    ASSERT_EQ(result.num_children, expected.num_children);

    ASSERT_EQ(result.child_indices, expected.child_indices);

    for(int i = 0; i < expected.minimum_cell_position.size(); i++)  {
        assert_eq_vec(convert_gpu_vec_to_vec(result.minimum_cell_position[i]), convert_gpu_vec_to_vec(expected.minimum_cell_position[i]));
        assert_eq_vec(convert_gpu_vec_to_vec(result.maximum_cell_position[i]), convert_gpu_vec_to_vec(expected.maximum_cell_position[i]));
        assert_eq_vec(convert_gpu_vec_to_vec(result.position_excitatory_element[i]), convert_gpu_vec_to_vec(expected.position_excitatory_element[i]));
        assert_eq_vec(convert_gpu_vec_to_vec(result.position_inhibitory_element[i]), convert_gpu_vec_to_vec(expected.position_inhibitory_element[i]));
    }

    ASSERT_EQ(result.num_free_elements_excitatory, expected.num_free_elements_excitatory);
    ASSERT_EQ(result.num_free_elements_inhibitory, expected.num_free_elements_inhibitory);
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

    octree.initializes_leaf_nodes(neurons_to_place.size());

    octree.construct_on_gpu(neurons_to_place.size());

    const std::shared_ptr<gpu::algorithm::OctreeHandle> gpu_handle = octree.get_gpu_handle();
    gpu::algorithm::OctreeCPUCopy octree_cpu_copy(neurons_to_place.size(), gpu_handle->get_number_virtual_neurons());
    gpu_handle->copy_to_cpu(octree_cpu_copy);

    ASSERT_EQ(octree_cpu_copy.neuron_ids.size(), octree.get_leaf_nodes().size());
    for (int i = 0; i < octree_cpu_copy.neuron_ids.size(); i++) {
        ASSERT_EQ(octree_cpu_copy.neuron_ids[i], octree.get_leaf_nodes()[octree_cpu_copy.neuron_ids[i]]->get_cell_neuron_id().get_neuron_id());
    }

    octree.overwrite_cpu_tree_with_gpu();

    auto* root = octree.get_root();
    std::stack<const OctreeNode<AdditionalCellAttributes> *> octree_nodes_cpu{};
    octree_nodes_cpu.push(root);

    std::stack<uint64_t> octree_nodes_gpu{};
    size_t num_neurons = neurons_to_place.size();

    // assumes root is in the last index
    octree_nodes_gpu.push(num_neurons + gpu_handle->get_number_virtual_neurons() - 1);
        
    size_t correct_counts = 0;
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

        bool cpu_node_is_virtual = current_node_cpu->get_cell().get_neuron_id().is_virtual();
        bool gpu_node_is_virtual = current_node_gpu >= num_neurons;
        if (cpu_node_is_virtual != gpu_node_is_virtual)
            RelearnException::fail("Octree::overwrite_cpu_tree_with_gpu: GPU and CPU Octree structure differs");

        gpu::Vec3d pos_ex_elem = octree_cpu_copy.position_excitatory_element.at(current_node_gpu);
        ASSERT_EQ(Vec3d(pos_ex_elem.x, pos_ex_elem.y, pos_ex_elem.z), current_node_cpu->get_cell().get_position_for(elem_type, SignalType::Excitatory).value()) << " Correct nodes before fail:" << correct_counts;

        gpu::Vec3d pos_in_elem = octree_cpu_copy.position_inhibitory_element.at(current_node_gpu);
        ASSERT_EQ(Vec3d(pos_in_elem.x, pos_in_elem.y, pos_in_elem.z), current_node_cpu->get_cell().get_position_for(elem_type, SignalType::Inhibitory).value());

        RelearnTypes::counter_type num_ex_elem = octree_cpu_copy.num_free_elements_excitatory.at(current_node_gpu);
        ASSERT_EQ(num_ex_elem, current_node_cpu->get_cell().get_number_elements_for(elem_type, SignalType::Excitatory));

        RelearnTypes::counter_type num_in_elem = octree_cpu_copy.num_free_elements_inhibitory.at(current_node_gpu);
        ASSERT_EQ(num_in_elem, current_node_cpu->get_cell().get_number_elements_for(elem_type, SignalType::Inhibitory));

        correct_counts++;
            
        // The order of the children should in theory be correct here
        if (current_node_cpu->is_parent() && current_node_gpu >= num_neurons) {
            const auto &children_cpu = current_node_cpu->get_children();
            int children_processed = 0;
            for (auto i = 0; i < 8; i++) {
                const auto child = children_cpu[7 - i];
                if (child != nullptr) {
                    octree_nodes_cpu.push(child);
                    octree_nodes_gpu.push(octree_cpu_copy.child_indices[children_processed * gpu_handle->get_number_virtual_neurons() + current_node_gpu - num_neurons]);

                    children_processed++;
                }
            }

            if (children_processed != octree_cpu_copy.num_children.at(current_node_gpu - num_neurons)) 
                RelearnException::fail("Octree::overwrite_cpu_tree_with_gpu: GPU and CPU Octree structure differs");
        }
    }

    if (!octree_nodes_gpu.empty())
        RelearnException::fail("Octree::overwrite_cpu_tree_with_gpu: GPU and CPU Octree structure differs");
}