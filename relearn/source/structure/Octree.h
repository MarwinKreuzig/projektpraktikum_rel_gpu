/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#pragma once

#include "../mpi/MPIWrapper.h"
#include "../structure/OctreeNode.h"
#include "../neurons/helper/ProbabilitySubinterval.h"
#include "../neurons/helper/RankNeuronId.h"
#include "../neurons/SignalType.h"
#include "../util/Vec3.h"

#include <map>
#include <optional>
#include <stack>
#include <utility>
#include <vector>

class Neurons;
class Partition;
class SynapticElements;

class Octree {
public:
    friend class Partition;

    using AccessEpochsStarted = std::vector<bool>;

private:
    /**
	 * Type for stack used in postorder tree walk
	 */
    struct StackElement {
    private:
        OctreeNode* ptr;

        // True if node has been on stack already
        // twice and can be visited now
        bool already_visited{ false };

        // Node's depth in the tree
        size_t depth;

    public:
        StackElement(OctreeNode* octree_node, size_t depth_in_tree) noexcept
            : ptr(octree_node)
            , depth(depth_in_tree) {
        }

        [[nodiscard]] OctreeNode* get_octree_node() const noexcept {
            return ptr;
        }

        void set_visited() noexcept {
            already_visited = true;
        }

        [[nodiscard]] bool get_visited() const noexcept {
            return already_visited;
        }

        [[nodiscard]] size_t get_depth_in_tree() const noexcept {
            return depth;
        }
    };

    /**
	 * Visit function used with postorder tree walk
	 *
	 * For inner nodes:
	 *  - Updates number of dendrites (exc, inh) and position of super neuron based on children
	 * For leaves (normal neurons):
	 *  - Updates number of dendrites (exc, inh) based on changes of synaptic elements
	 */
    class FunctorUpdateNode {
    public:
        FunctorUpdateNode(const std::vector<double>& dendrites_excitatory_counts, const std::vector<unsigned int>& dendrites_excitatory_connected_counts,
            const std::vector<double>& dendrites_inhibitory_counts, const std::vector<unsigned int>& dendrites_inhibitory_connected_counts,
            size_t number_neurons) noexcept
            : dendrites_excitatory_counts(dendrites_excitatory_counts)
            , dendrites_excitatory_connected_counts(dendrites_excitatory_connected_counts)
            , dendrites_inhibitory_counts(dendrites_inhibitory_counts)
            , dendrites_inhibitory_connected_counts(dendrites_inhibitory_connected_counts)
            , number_neurons(number_neurons) {
        }

        void operator()(OctreeNode* node) /*noexcept*/ {
            RelearnException::check(node != nullptr, "In FunctorUpdateNode, node is nullptr");

            if (!node->is_parent()) {
                // Get ID of the node's neuron
                const size_t neuron_id = node->get_cell().get_neuron_id();

                // Calculate number of vacant dendrites for my neuron
                RelearnException::check(neuron_id < number_neurons, "Neuron id was too large in the operator");

                const auto number_vacant_dendrites_excitatory = static_cast<unsigned int>(dendrites_excitatory_counts[neuron_id] - dendrites_excitatory_connected_counts[neuron_id]);
                const auto number_vacant_dendrites_inhibitory = static_cast<unsigned int>(dendrites_inhibitory_counts[neuron_id] - dendrites_inhibitory_connected_counts[neuron_id]);

                node->set_cell_num_dendrites(number_vacant_dendrites_excitatory, number_vacant_dendrites_inhibitory);
                return;
            }

            // I'm inner node, i.e., I have a super neuron
            Vec3d my_position_dendrites_excitatory = { 0., 0., 0. };
            Vec3d my_position_dendrites_inhibitory = { 0., 0., 0. };

            // Sum of number of dendrites of all my children
            auto my_number_dendrites_excitatory = 0;
            auto my_number_dendrites_inhibitory = 0;

            // For all my children
            for (const auto& child : node->get_children()) {
                if (child == nullptr) {
                    continue;
                }

                // Sum up number of dendrites
                const auto child_number_dendrites_excitatory = child->get_cell().get_neuron_num_dendrites_exc();
                const auto child_number_dendrites_inhibitory = child->get_cell().get_neuron_num_dendrites_inh();

                my_number_dendrites_excitatory += child_number_dendrites_excitatory;
                my_number_dendrites_inhibitory += child_number_dendrites_inhibitory;

                // Average the position by using the number of dendrites as weights
                std::optional<Vec3d> child_position_dendrites_excitatory = child->get_cell().get_neuron_position_exc();
                std::optional<Vec3d> child_position_dendrites_inhibitory = child->get_cell().get_neuron_position_inh();

                /**
				* We can use position if it's valid or if corresponding num of dendrites is 0 
				*/
                RelearnException::check(child_position_dendrites_excitatory.has_value() || (0 == child_number_dendrites_excitatory), "temp position exc was bad");
                RelearnException::check(child_position_dendrites_inhibitory.has_value() || (0 == child_number_dendrites_inhibitory), "temp position inh was bad");

                if (child_position_dendrites_excitatory.has_value()) {
                    const auto scaled_position = child_position_dendrites_excitatory.value() * static_cast<double>(child_number_dendrites_excitatory);
                    my_position_dendrites_excitatory += scaled_position;
                }

                if (child_position_dendrites_inhibitory.has_value()) {
                    const auto scaled_position = child_position_dendrites_inhibitory.value() * static_cast<double>(child_number_dendrites_inhibitory);
                    my_position_dendrites_inhibitory += scaled_position;
                }
            }

            node->set_cell_num_dendrites(my_number_dendrites_excitatory, my_number_dendrites_inhibitory);

            /**
			* For calculating the new weighted position, make sure that we don't
			* divide by 0. This happens if the my number of dendrites is 0.
			*/
            if (0 == my_number_dendrites_excitatory) {
                node->set_cell_neuron_pos_exc({});
            } else {
                const auto scaled_position = my_position_dendrites_excitatory / my_number_dendrites_excitatory;
                node->set_cell_neuron_pos_exc(std::optional<Vec3d>{ scaled_position });
            }

            if (0 == my_number_dendrites_inhibitory) {
                node->set_cell_neuron_pos_inh({});
            } else {
                const auto scaled_position = my_position_dendrites_inhibitory / my_number_dendrites_inhibitory;
                node->set_cell_neuron_pos_inh(std::optional<Vec3d>{ scaled_position });
            }
        }

    private:
        const std::vector<double>& dendrites_excitatory_counts;
        const std::vector<unsigned int>& dendrites_excitatory_connected_counts;
        const std::vector<double>& dendrites_inhibitory_counts;
        const std::vector<unsigned int>& dendrites_inhibitory_connected_counts;
        size_t number_neurons = 0;
    };

    /**
	 * Visit function used with postorder tree walk
	 * Frees the respective node
	 */
    class FunctorFreeNode {
    public:
        FunctorFreeNode() noexcept = default;

        void operator()(OctreeNode* node) {
            RelearnException::check(node != nullptr, "In FunctorFreeNode, node was nullptr");
            MPIWrapper::delete_octree_node(node); 
        }
    };

    Octree() = default;

public:
    explicit Octree(const Partition& part);
    Octree(const Partition& part, double acceptance_criterion, double sigma);
    ~Octree() /*noexcept(false)*/;

    Octree(const Octree& other) = delete;
    Octree(Octree&& other) = delete;

    Octree& operator=(const Octree& other) = delete;
    Octree& operator=(Octree&& other) = delete;

    // Set simulation box size of the tree
    void set_size(const Vec3d& min, const Vec3d& max) noexcept {
        xyz_min = min;
        xyz_max = max;
    }

    // Set acceptance criterion for cells in the tree
    void set_acceptance_criterion(double acceptance_criterion) noexcept {
        this->acceptance_criterion = acceptance_criterion;
    }

    // Set probability parameter used to determine the probability
    // for a cell of being selected
    void set_probability_parameter(double sigma) noexcept {
        this->sigma = sigma;
    }

    // Set naive method parameter used to determine if all cells
    // should be expanded regardless of whether dendrites are available
    void set_naive_method_parameter(bool naive_method) noexcept {
        this->naive_method = naive_method;
    }

    void set_root_level(size_t root_level) noexcept {
        this->root_level = root_level;
    }

    void set_no_free_in_destructor() noexcept {
        no_free_in_destructor = true;
    }

    void set_level_of_branch_nodes(size_t level) noexcept {
        level_of_branch_nodes = level;
    }

    [[nodiscard]] OctreeNode* get_root() const noexcept {
        return root;
    }

    [[nodiscard]] size_t get_level_of_branch_nodes() const noexcept {
        return level_of_branch_nodes;
    }

    [[nodiscard]] size_t get_num_local_trees() const noexcept {
        return local_trees.size();
    }

    [[nodiscard]] OctreeNode* get_local_root(size_t local_id) noexcept {
        const Octree* local_tree = local_trees[local_id];
        return local_tree->get_root();
    }

    void print();

    void free();

    // Insert neuron into the tree
    [[nodiscard]] OctreeNode* insert(const Vec3d& position, size_t neuron_id, int rank);

    // Insert an octree node with its subtree into the tree
    void insert(OctreeNode* node_to_insert);

    void insert_local_tree(Octree* node_to_insert);

    // The caller must ensure that only inner nodes are visited.
    // "max_level" must be chosen correctly for this
    void update_from_level(size_t max_level);

    void update_local_trees(const SynapticElements& dendrites_exc, const SynapticElements& dendrites_inh, size_t num_neurons);

    [[nodiscard]] std::optional<RankNeuronId> find_target_neuron(size_t src_neuron_id, const Vec3d& axon_pos_xyz, SignalType dendrite_type_needed);

    void empty_remote_nodes_cache();

private:
    /**
	 * Do a postorder tree walk startring at "octree" and run the function "visit" for every node when it is visited
     * Does ignore every node which's level in the octree is greater than "max_level"
	 */
    template <typename Functor>
    void tree_walk_postorder(Octree* octree, Functor visit, size_t max_level = std::numeric_limits<size_t>::max()) {
        RelearnException::check(octree != nullptr, "In tree_walk_postorder, octree was nullptr");

        // Tree is empty
        if (octree->root == nullptr) {
            return;
        }

        std::stack<StackElement> stack{};
        // Push node onto stack
        stack.emplace(octree->root, 0);

        while (!stack.empty()) {
            // Get top-of-stack node
            auto& current_element = stack.top();
            const auto current_depth = current_element.get_depth_in_tree();
            auto* current_octree_node = current_element.get_octree_node();

            // Node should be visited now?
            if (current_element.get_visited()) {
                RelearnException::check(current_octree_node->get_level() <= max_level, "current_element had bad level");

                // Apply action to node
                visit(current_octree_node);

                // Pop node from stack
                stack.pop();
            } else {
                // Mark node to be visited next time
                current_element.set_visited();

                // We're at the border of where we want to update, so don't push children
                if (current_depth >= max_level) {
                    continue;
                }

                const auto& children = current_octree_node->get_children();
                for (auto it = children.crbegin(); it != children.crend(); ++it) {
                    if (*it != nullptr) {
                        stack.emplace(*it, current_depth + 1);
                    }
                }                
            }
        } /* while */
    }

    /**
	 * Print tree in postorder
	 */
    void postorder_print();

    /**
	 * If we use the naive method accept leaf cells only, otherwise
	 * test if cell has dendrites available and is precise enough.
	 * Returns true if accepted, false otherwise
	 */
    [[nodiscard]] std::tuple<bool, bool> acceptance_criterion_test(
        const Vec3d& axon_pos_xyz,
        const OctreeNode* node_with_dendrite,
        SignalType dendrite_type_needed,
        bool naive_method) const /*noexcept*/;

    /**
	 * Returns vector with nodes for creating the probability interval
	 */
    [[nodiscard]] ProbabilitySubintervalVector get_nodes_for_interval(
        const Vec3d& axon_pos_xyz,
        OctreeNode* root,
        SignalType dendrite_type_needed,
        bool naive_method);

    /**
	 * Returns probability interval, i.e., vector with nodes where each node is assigned a probability.
	 * Nodes with probability 0 are removed from the vector.
	 * The probabilities sum up to 1
	 */
    [[nodiscard]] std::vector<double> create_interval(size_t src_neuron_id, const Vec3d& axon_pos_xyz, SignalType dendrite_type_needed, const ProbabilitySubintervalVector& vector) const;

    /**
	 * Returns attractiveness for connecting two given nodes
	 * NOTE: This is not a probability yet as it could be >1
	 */
    [[nodiscard]] double calc_attractiveness_to_connect(size_t src_neuron_id, const Vec3d& axon_pos_xyz, const OctreeNode& node_with_dendrite, SignalType dendrite_type_needed) const /*noexcept*/;

    // Root of the tree
    OctreeNode* root{ nullptr };

    std::vector<Octree*> local_trees{};

    // Level which is assigned to the root of the tree (default = 0)
    size_t root_level{ Constants::uninitialized };

    // 'True' if destructor should not free the tree nodes
    bool no_free_in_destructor{ false };

    // Two points describe simulation box size of the tree
    Vec3d xyz_min{ 0 };
    Vec3d xyz_max{ 0 };

    double acceptance_criterion{ default_theta }; // Acceptance criterion
    double sigma{ default_sigma }; // Probability parameter
    bool naive_method{ false }; // If true, expand every cell regardless of whether dendrites are available or not
    size_t level_of_branch_nodes{ Constants::uninitialized };

    // Cache with nodes owned by other ranks
    using NodesCacheKey = std::pair<int, OctreeNode*>;
    using NodesCacheValue = OctreeNode*;
    using NodesCache = std::map<NodesCacheKey, NodesCacheValue>;
    NodesCache remote_nodes_cache{};

public:
    constexpr static double default_theta{ 0.3 };
    constexpr static double default_sigma{ 750.0 };

    constexpr static double max_theta{ 0.5 };
};
