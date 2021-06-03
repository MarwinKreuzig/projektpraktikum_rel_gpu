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
#include "../neurons/SignalType.h"
#include "../neurons/helper/ProbabilitySubinterval.h"
#include "../neurons/helper/RankNeuronId.h"
#include "../structure/OctreeNode.h"
#include "../util/RelearnException.h"
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
        FunctorUpdateNode(
            const std::vector<double>& dendrites_exc_cnts, 
            const std::vector<unsigned int>& dendrites_exc_connected_cnts,
            const std::vector<double>& dendrites_inh_cnts, 
            const std::vector<unsigned int>& dendrites_inh_connected_cnts,
            const std::vector<double>& axons_exc_cnts, 
            const std::vector<unsigned int>& axons_exc_connected_cnts,
            const std::vector<double>& axons_inh_cnts, 
            const std::vector<unsigned int>& axons_inh_connected_cnts,
            size_t num_neurons) noexcept
            : dendrites_exc_cnts(dendrites_exc_cnts)
            , dendrites_exc_connected_cnts(dendrites_exc_connected_cnts)
            , dendrites_inh_cnts(dendrites_inh_cnts)
            , dendrites_inh_connected_cnts(dendrites_inh_connected_cnts)
            , axons_exc_cnts(axons_exc_cnts)
            , axons_exc_connected_cnts(axons_exc_connected_cnts)
            , axons_inh_cnts(axons_inh_cnts)
            , axons_inh_connected_cnts(axons_inh_connected_cnts)
            , num_neurons(num_neurons) {
        }

        void operator()(OctreeNode* node) /*noexcept*/ {
            RelearnException::check(node != nullptr, "In FunctorUpdateNode, node is nullptr");

            // NOLINTNEXTLINE
            if (!node->is_parent()) {
                // Get ID of the node's neuron
                const size_t neuron_id = node->get_cell().get_neuron_id();

                // Calculate number of vacant dendrites for my neuron
                RelearnException::check(neuron_id < num_neurons, "Neuron id was too large in the operator");

                const auto num_vacant_dendrites_exc = static_cast<unsigned int>(dendrites_exc_cnts[neuron_id] - dendrites_exc_connected_cnts[neuron_id]);
                const auto num_vacant_dendrites_inh = static_cast<unsigned int>(dendrites_inh_cnts[neuron_id] - dendrites_inh_connected_cnts[neuron_id]);
                node->set_cell_num_dendrites(num_vacant_dendrites_exc, num_vacant_dendrites_inh);

                // Calculate number of vacant axons for my neuron
                const auto num_vacant_axons_exc = static_cast<unsigned int>(axons_exc_cnts[neuron_id] - axons_exc_connected_cnts[neuron_id]);
                const auto num_vacant_axons_inh = static_cast<unsigned int>(axons_inh_cnts[neuron_id] - axons_inh_connected_cnts[neuron_id]);
                node->set_cell_num_axons(num_vacant_axons_exc,num_vacant_axons_inh);
                
                return;
            }

            // I'm inner node, i.e., I have a super neuron
            Vec3d xyz_pos_dend_exc = { 0., 0., 0. };
            Vec3d xyz_pos_dend_inh = { 0., 0., 0. };
            Vec3d xyz_pos_ax_exc = { 0., 0., 0. };
            Vec3d xyz_pos_ax_inh = { 0., 0., 0. };

            // Sum of number of dendrites of all my children
            auto num_dendrites_exc = 0;
            auto num_dendrites_inh = 0;
            auto num_axons_exc = 0;
            auto num_axons_inh = 0;

            // For all my children
            for (const auto& child : node->get_children()) {
                if (child == nullptr) {
                    continue;
                }

                // Sum up number of dendrites
                auto temp_num_dendrites_exc = child->get_cell().get_neuron_num_dendrites_exc();
                auto temp_num_dendrites_inh = child->get_cell().get_neuron_num_dendrites_inh();
                auto temp_num_axons_exc = child->get_cell().get_neuron_num_axons_exc();
                auto temp_num_axons_inh = child->get_cell().get_neuron_num_axons_inh();
                num_dendrites_exc += temp_num_dendrites_exc;
                num_dendrites_inh += temp_num_dendrites_inh;
                num_axons_exc += temp_num_axons_exc;
                num_axons_inh += temp_num_axons_inh;

                // Average the position by using the number of dendrites as weights
                std::optional<Vec3d> temp_xyz_pos_dend_exc = child->get_cell().get_neuron_position_dendrites_exc();
                std::optional<Vec3d> temp_xyz_pos_dend_inh = child->get_cell().get_neuron_position_dendrites_inh();
                std::optional<Vec3d> temp_xyz_pos_ax_exc = child->get_cell().get_neuron_position_axons_exc();
                std::optional<Vec3d> temp_xyz_pos_ax_inh = child->get_cell().get_neuron_position_axons_inh();


                /**
					 * We can use position if it's valid or if corresponding num of dendrites is 0 
					 */
                RelearnException::check(temp_xyz_pos_dend_exc.has_value() || (0 == temp_num_dendrites_exc), "temp position dend_exc was bad");
                RelearnException::check(temp_xyz_pos_dend_inh.has_value() || (0 == temp_num_dendrites_inh), "temp position dend_inh was bad");
                RelearnException::check(temp_xyz_pos_ax_exc.has_value() || (0 == temp_num_axons_exc), "temp position ax_exc was bad");
                RelearnException::check(temp_xyz_pos_ax_inh.has_value() || (0 == temp_num_axons_inh), "temp position ax_inh was bad");

                for (auto j = 0; j < 3; j++) {
                    if (temp_xyz_pos_dend_exc.has_value()) {
                        xyz_pos_dend_exc[j] += static_cast<double>(temp_num_dendrites_exc) * temp_xyz_pos_dend_exc.value()[j];
                    }
                    if (temp_xyz_pos_dend_inh.has_value()) {
                        xyz_pos_dend_inh[j] += static_cast<double>(temp_num_dendrites_inh) * temp_xyz_pos_dend_inh.value()[j];
                    }
                     if (temp_xyz_pos_ax_exc.has_value()) {
                        xyz_pos_ax_exc[j] += static_cast<double>(temp_num_axons_exc) * temp_xyz_pos_ax_exc.value()[j];
                    }
                    if (temp_xyz_pos_ax_inh.has_value()) {
                        xyz_pos_ax_inh[j] += static_cast<double>(temp_num_axons_inh) * temp_xyz_pos_ax_inh.value()[j];
                    }
                }
            }

            node->set_cell_num_dendrites(my_number_dendrites_excitatory, my_number_dendrites_inhibitory);

            /**
			* For calculating the new weighted position, make sure that we don't
			* divide by 0. This happens if the total number of dendrites is 0.
			*/
            auto divisor_pos_dend_exc = num_dendrites_exc;
            auto divisor_pos_dend_inh = num_dendrites_inh;
            auto divisor_pos_ax_exc = num_axons_exc;
            auto divisor_pos_ax_inh = num_axons_inh;
            auto valid_pos_dend_exc = true;
            auto valid_pos_dend_inh = true;
            auto valid_pos_ax_exc = true;
            auto valid_pos_ax_inh = true;

            if (0 == num_dendrites_exc) {
                valid_pos_dend_exc = false; // Mark result as invald
                divisor_pos_dend_exc = 1;
            }

            if (0 == num_dendrites_inh) {
                valid_pos_dend_inh = false; // Mark result as invalid
                divisor_pos_dend_inh = 1;
            }

            if (0 == num_axons_exc) {
                valid_pos_ax_exc = false; // Mark result as invald
                divisor_pos_ax_exc = 1;
            }

            if (0 == num_axons_inh) {
                valid_pos_ax_inh = false; // Mark result as invalid
                divisor_pos_ax_inh = 1;
            }

            // Calc the average by dividing by the total number of dendrites
            for (auto j = 0; j < 3; j++) {
                xyz_pos_dend_exc[j] /= divisor_pos_dend_exc;
                xyz_pos_dend_inh[j] /= divisor_pos_dend_inh;
                xyz_pos_ax_exc[j] /= divisor_pos_ax_exc;
                xyz_pos_ax_inh[j] /= divisor_pos_ax_inh;
            }
            node->set_cell_num_dendrites(num_dendrites_exc, num_dendrites_inh);
            node->set_cell_num_axons(num_axons_exc,num_axons_inh);
            // Also mark if new position is valid using valid_pos_{exc,inh}

            std::optional<Vec3d> dend_ex_pos = valid_pos_dend_exc ? std::optional<Vec3d>{ xyz_pos_dend_exc } : std::optional<Vec3d>{};
            std::optional<Vec3d> dend_in_pos = valid_pos_dend_inh ? std::optional<Vec3d>{ xyz_pos_dend_inh } : std::optional<Vec3d>{};
            std::optional<Vec3d> ax_ex_pos = valid_pos_ax_exc ? std::optional<Vec3d>{ xyz_pos_ax_exc } : std::optional<Vec3d>{};
            std::optional<Vec3d> ax_in_pos = valid_pos_ax_inh ? std::optional<Vec3d>{ xyz_pos_ax_inh } : std::optional<Vec3d>{};


            node->set_cell_neuron_pos_dend_exc(dend_ex_pos);
            node->set_cell_neuron_pos_dend_inh(dend_in_pos);
            node->set_cell_neuron_pos_ax_exc(ax_ex_pos);
            node->set_cell_neuron_pos_ax_inh(ax_in_pos);
            
        }

    private:
        const std::vector<double>& dendrites_exc_cnts;
        const std::vector<unsigned int>& dendrites_exc_connected_cnts;
        const std::vector<double>& dendrites_inh_cnts;
        const std::vector<unsigned int>& dendrites_inh_connected_cnts;
        const std::vector<double>& axons_exc_cnts;
        const std::vector<unsigned int>& axons_exc_connected_cnts;
        const std::vector<double>& axons_inh_cnts;
        const std::vector<unsigned int>& axons_inh_connected_cnts;
        size_t num_neurons = 0;
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

public:
    Octree(const Vec3d& xyz_min, const Vec3d& xyz_max, size_t level_of_branch_nodes);
    Octree(const Vec3d& xyz_min, const Vec3d& xyz_max, size_t level_of_branch_nodes, double acceptance_criterion, double sigma);
    ~Octree() /*noexcept(false)*/;

    Octree(const Octree& other) = delete;
    Octree(Octree&& other) = delete;

    Octree& operator=(const Octree& other) = delete;
    Octree& operator=(Octree&& other) = delete;

    // Set simulation box size of the tree
    void set_size(const Vec3d& min, const Vec3d& max) {
        RelearnException::check(min.get_x() < max.get_x() && min.get_y() < max.get_y() && min.get_z() < max.get_z(), "In Octree::set_size, the minimum was not smaller than the maximum");

        xyz_min = min;
        xyz_max = max;
    }

    // Set acceptance criterion for cells in the tree
    void set_acceptance_criterion(double acceptance_criterion) {
        RelearnException::check(acceptance_criterion >= 0.0, "In Octree::set_acceptance_criterion, acceptance_criterion was less than 0");
        this->acceptance_criterion = acceptance_criterion;

        if (acceptance_criterion == 0.0) {
            naive_method = true;
        } else {
            naive_method = false;
        }
    }

    // Set probability parameter used to determine the probability
    // for a cell of being selected
    void set_probability_parameter(double sigma) {
        RelearnException::check(sigma > 0.0, "In Octree::set_probability_parameter, sigma was not greater than 0");
        this->sigma = sigma;
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

    [[nodiscard]] bool is_naive_method_used() const noexcept {
        return naive_method;
    }

    [[nodiscard]] double get_probabilty_parameter() const noexcept {
        return sigma;
    }

    [[nodiscard]] double get_acceptance_criterion() const noexcept {
        return acceptance_criterion;
    }

    [[nodiscard]] const Vec3d& get_xyz_min() const noexcept {
        return xyz_min;
    }

    [[nodiscard]] const Vec3d& get_xyz_max() const noexcept {
        return xyz_max;
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
        OctreeNode* local_tree = local_trees[local_id];
        return local_tree;
    }

    void print();

    void free();

    // Insert neuron into the tree
    [[nodiscard]] OctreeNode* insert(const Vec3d& position, size_t neuron_id, int rank);

    // Insert an octree node with its subtree into the tree
    void insert_local_tree(OctreeNode* node_to_insert, size_t index_1d) {
        *local_trees[index_1d] = *node_to_insert;
    }

    // The caller must ensure that only inner nodes are visited.
    // "max_level" must be chosen correctly for this
    void update_from_level(size_t max_level);

    void update_local_trees(const SynapticElements& dendrites_exc, const SynapticElements& dendrites_inh, size_t num_neurons);

    [[nodiscard]] std::optional<RankNeuronId> find_target_neuron(size_t src_neuron_id, const Vec3d& axon_pos_xyz, SignalType dendrite_type_needed);
   
   const std::optional<OctreeNode*> do_random_experiment(OctreeNode *source, const std::vector<double>& atractiveness);

    void empty_remote_nodes_cache();

    void synchronize_local_trees();

private:
    /**
	 * Do a postorder tree walk startring at "octree" and run the function "visit" for every node when it is visited
     * Does ignore every node which's level in the octree is greater than "max_level"
	 */
    template <typename Functor>
    void tree_walk_postorder(OctreeNode* root, Functor visit, size_t max_level = std::numeric_limits<size_t>::max()) {
        RelearnException::check(root != nullptr, "In tree_walk_postorder, octree was nullptr");

        std::stack<StackElement> stack{};
        // Push node onto stack
        stack.emplace(root, 0);

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

    void construct_global_tree_part();

   public: const std::vector<double> calc_attractiveness_to_connect_FMM(OctreeNode *source, const SignalType dendrite_type_needed);
    
    // Root of the tree
    OctreeNode* root{ nullptr };

    std::vector<OctreeNode*> local_trees{};

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
