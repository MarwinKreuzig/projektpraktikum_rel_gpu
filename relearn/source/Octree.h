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

#include "Cell.h"
#include "Commons.h"
#include "LogMessages.h"
#include "NeuronModels.h"
#include "OctreeNode.h"
#include "Parameters.h"
#include "RelearnException.h"
#include "SynapticElements.h"
#include "Vec3.h"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <stack>

// Forward declarations of Neurons.h
class SynapseCreationRequests;
using MapSynapseCreationRequests = std::map<int, SynapseCreationRequests>;

class Neurons;

class Partition;

class Octree {
public:
    friend class Partition;

    using AccessEpochsStarted = std::vector<bool>;

    /**
	 * Type for list elements used to create probability subinterval
	 */
    struct ProbabilitySubinterval {
        ProbabilitySubinterval() = default;

        explicit ProbabilitySubinterval(OctreeNode* node) noexcept
            : ptr(node) {
        }

        OctreeNode* ptr{ nullptr };
        double probability{ 0.0 };
        MPIWrapper::AsyncToken mpi_request{ MPIWrapper::get_null_request() };
        int request_rank{ -1 };
    };
    using ProbabilitySubintervalList = std::list<std::shared_ptr<ProbabilitySubinterval>>;

    /**
	 * Type for vacant axon for which a target neuron needs to be found
	 */
    struct VacantAxon {
        VacantAxon(size_t neuron_id, const Vec3d& pos, Cell::DendriteType dendrite_type_needed)
            : neuron_id(neuron_id)
            , xyz_pos(pos)
            , dendrite_type_needed(dendrite_type_needed) {
        }

        ProbabilitySubintervalList nodes_accepted;
        ProbabilitySubintervalList nodes_to_visit;
        size_t neuron_id;
        Vec3d xyz_pos;
        Cell::DendriteType dendrite_type_needed;
    };
    using VacantAxonList = std::list<std::shared_ptr<VacantAxon>>;

private:
    /**
	 * Type for stack used in postorder tree walk
	 */
    struct StackElement {

        StackElement(OctreeNode* oct_ptr, bool flag, size_t depth) noexcept
            : ptr(oct_ptr)
            , flag(flag)
            , depth(depth) {
        }

        OctreeNode* ptr;

        // True if node has been on stack already
        // twice and can be visited now
        bool flag;

        // Node's depth in the tree
        size_t depth;
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
        FunctorUpdateNode(const std::vector<double>& dendrites_exc_cnts, const std::vector<double>& dendrites_exc_connected_cnts,
            const std::vector<double>& dendrites_inh_cnts, const std::vector<double>& dendrites_inh_connected_cnts,
            size_t num_neurons) noexcept
            : dendrites_exc_cnts(dendrites_exc_cnts)
            , dendrites_exc_connected_cnts(dendrites_exc_connected_cnts)
            , dendrites_inh_cnts(dendrites_inh_cnts)
            , dendrites_inh_connected_cnts(dendrites_inh_connected_cnts)
            , num_neurons(num_neurons) {
        }

        //FunctorUpdateNode() noexcept :
        //	dendrites_exc_cnts(std::vector<double>{}),
        //	dendrites_exc_connected_cnts(std::vector<double>{}),
        //	dendrites_inh_cnts(std::vector<double>{}),
        //	dendrites_inh_connected_cnts(std::vector<double>{})
        //{}

        void operator()(OctreeNode* node) /*noexcept*/ {
            // I'm inner node, i.e., I have a super neuron
            if (node->is_parent) {
                Vec3d temp_xyz_pos_exc;
                Vec3d temp_xyz_pos_inh;
                Vec3d xyz_pos_exc = { 0., 0., 0. };
                Vec3d xyz_pos_inh = { 0., 0., 0. };
                bool valid_pos_exc = false;
                bool valid_pos_inh = false;

                // Sum of number of dendrites of all my children
                auto num_dendrites_exc = 0;
                auto num_dendrites_inh = 0;

                // For all my children
                for (const auto& child : node->children) {
                    if (child == nullptr) {
                        continue;
                    }

                    // Sum up number of dendrites
                    auto temp_num_dendrites_exc = child->cell.get_neuron_num_dendrites_exc();
                    auto temp_num_dendrites_inh = child->cell.get_neuron_num_dendrites_inh();
                    num_dendrites_exc += temp_num_dendrites_exc;
                    num_dendrites_inh += temp_num_dendrites_inh;

                    // Average the position by using the number of dendrites as weights
                    std::tie(temp_xyz_pos_exc, valid_pos_exc) = child->cell.get_neuron_position_exc();
                    std::tie(temp_xyz_pos_inh, valid_pos_inh) = child->cell.get_neuron_position_inh();
                    /**
					 * We can use position if it's valid or if corresponding num of dendrites is 0 (due to multiplying position with 0)
					 */

                    RelearnException::check(valid_pos_exc || (0 == temp_num_dendrites_exc));
                    RelearnException::check(valid_pos_inh || (0 == temp_num_dendrites_inh));

                    for (auto j = 0; j < 3; j++) {
                        xyz_pos_exc[j] += static_cast<double>(temp_num_dendrites_exc) * temp_xyz_pos_exc[j];
                        xyz_pos_inh[j] += static_cast<double>(temp_num_dendrites_inh) * temp_xyz_pos_inh[j];
                    }
                }
                /**
				 * For calculating the new weighted position, make sure that we don't
				 * divide by 0. This happens if the total number of dendrites is 0.
				 */
                auto divisor_pos_exc = num_dendrites_exc;
                auto divisor_pos_inh = num_dendrites_inh;
                valid_pos_exc = true;
                valid_pos_inh = true;

                if (0 == num_dendrites_exc) {
                    valid_pos_exc = false; // Mark result as invald
                    divisor_pos_exc = 1;
                }
                if (0 == num_dendrites_inh) {
                    valid_pos_inh = false; // Mark result as invalid
                    divisor_pos_inh = 1;
                }
                // Calc the average by dividing by the total number of dendrites
                for (auto j = 0; j < 3; j++) {
                    xyz_pos_exc[j] /= divisor_pos_exc;
                    xyz_pos_inh[j] /= divisor_pos_inh;
                }
                node->cell.set_neuron_num_dendrites_exc(num_dendrites_exc);
                node->cell.set_neuron_num_dendrites_inh(num_dendrites_inh);
                // Also mark if new position is valid using valid_pos_{exc,inh}
                node->cell.set_neuron_position_exc(xyz_pos_exc, valid_pos_exc);
                node->cell.set_neuron_position_inh(xyz_pos_inh, valid_pos_inh);
            }
            // I'm leaf node, i.e., I have a normal neuron
            else {
                // Get ID of the node's neuron
                const size_t neuron_id = node->cell.get_neuron_id();

                // Calculate number of vacant dendrites for my neuron
                RelearnException::check(neuron_id < num_neurons);

                const auto num_vacant_dendrites_exc = static_cast<unsigned int>(dendrites_exc_cnts[neuron_id] - dendrites_exc_connected_cnts[neuron_id]);
                const auto num_vacant_dendrites_inh = static_cast<unsigned int>(dendrites_inh_cnts[neuron_id] - dendrites_inh_connected_cnts[neuron_id]);

                node->cell.set_neuron_num_dendrites_exc(num_vacant_dendrites_exc);
                node->cell.set_neuron_num_dendrites_inh(num_vacant_dendrites_inh);
            }
        }

    private:
        const std::vector<double>& dendrites_exc_cnts;
        const std::vector<double>& dendrites_exc_connected_cnts;
        const std::vector<double>& dendrites_inh_cnts;
        const std::vector<double>& dendrites_inh_connected_cnts;
        size_t num_neurons = 0;
    };

    /**
	 * Visit function used with postorder tree walk
	 * Frees the node
	 */
    class FunctorFreeNode {
    public:
        FunctorFreeNode() noexcept = default;
        void operator()(OctreeNode* node) { MPIWrapper::delete_octree_node(node); }
    };

    Octree();

public:
    Octree(const Partition& part, double acceptance_criterion, double sigma, size_t max_num_pending_vacant_axons);
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

    void set_max_num_pending_vacant_axons(size_t max) noexcept {
        max_num_pending_vacant_axons = max;
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

    void update(const std::vector<double>& dendrites_exc_cnts, const std::vector<double>& dendrites_exc_connected_cnts,
        const std::vector<double>& dendrites_inh_cnts, const std::vector<double>& dendrites_inh_connected_cnts,
        size_t num_neurons);

    // The caller must ensure that only inner nodes are visited.
    // "max_level" must be chosen correctly for this
    void update_from_level(size_t max_level);

    void update_local_trees(const SynapticElements& dendrites_exc, const SynapticElements& dendrites_inh, const size_t& num_neurons) {
        for (auto& local_tree : local_trees) {
            local_tree->update(
                dendrites_exc.get_cnts(), dendrites_exc.get_connected_cnts(),
                dendrites_inh.get_cnts(), dendrites_inh.get_connected_cnts(),
                num_neurons);
        }
    }

    [[nodiscard]] bool find_target_neuron(size_t src_neuron_id, const Vec3d& axon_pos_xyz, Cell::DendriteType dendrite_type_needed, size_t& target_neuron_id, int& target_rank);

    void find_target_neurons(MapSynapseCreationRequests& map_synapse_creation_requests_outgoing, const Neurons& neurons);

    void empty_remote_nodes_cache();

private:
    /**
	 * Do a postorder tree walk and run the
	 * function "visit" for every node when
	 * it is visited
	 */
    template <typename Functor>
    void tree_walk_postorder(Functor visit, size_t max_level = std::numeric_limits<size_t>::max()) {
        std::stack<StackElement> stack{};

        // Tree is empty
        if (!root) {
            return;
        }

        // Push node onto stack
        stack.emplace(root, false, 0);

        while (!stack.empty()) {
            // Get top-of-stack node
            auto& elem = stack.top();
            const auto depth = elem.depth;

            // Node should be visited now?
            if (elem.flag) {
                RelearnException::check(elem.ptr->level <= max_level);

                // Apply action to node
                visit(elem.ptr);

                // Pop node from stack
                stack.pop();
            } else {
                // Mark node to be visited next time
                elem.flag = true;

                // Only push node's children onto stack if
                // they don't exceed "max_level"
                if (depth < max_level) {
                    // Push node's children onto stack

                    for (auto it = elem.ptr->children.crbegin(); it != elem.ptr->children.crend(); ++it) {
                        if (*it != nullptr) {
                            stack.emplace(*it, false, depth + 1);
                        }
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
    bool acceptance_criterion_test(const Vec3d& axon_pos_xyz,
        const OctreeNode* node_with_dendrite,
        Cell::DendriteType dendrite_type_needed,
        bool naive_method,
        bool& has_vacant_dendrites) const /*noexcept*/;

    /**
	 * Returns list with nodes for creating the probability interval
	 */
    void get_nodes_for_interval(
        const Vec3d& axon_pos_xyz,
        OctreeNode* root,
        Cell::DendriteType dendrite_type_needed,
        ProbabilitySubintervalList& list,
        bool naive_method);

    /**
	 * Returns probability interval, i.e., list with nodes where each node is assigned a probability.
	 * Nodes with probability 0 are removed from the list.
	 * The probabilities sum up to 1
	 */
    void create_interval(size_t src_neuron_id, const Vec3d& axon_pos_xyz, Cell::DendriteType dendrite_type_needed, ProbabilitySubintervalList& list) const;

    /**
	 * Returns attractiveness for connecting two given nodes
	 * NOTE: This is not a probability yet as it could be >1
	 */
    [[nodiscard]] double calc_attractiveness_to_connect(size_t src_neuron_id, const Vec3d& axon_pos_xyz, const OctreeNode& node_with_dendrite, Cell::DendriteType dendrite_type_needed) const /*noexcept*/;

    /**
	 * Randomly select node from probability interval
	 */
    [[nodiscard]] OctreeNode* select_subinterval(const ProbabilitySubintervalList& list);

    [[nodiscard]] static bool node_is_local(const OctreeNode& node) /*noexcept*/;

    static void append_node(OctreeNode* node, ProbabilitySubintervalList& list);
    void append_children(OctreeNode* node, ProbabilitySubintervalList& list, AccessEpochsStarted& epochs_started);

    // Root of the tree
    OctreeNode* root{ nullptr };

    std::vector<Octree*> local_trees;

    // Level which is assigned to the root of the tree (default = 0)
    size_t root_level{ Constants::uninitialized };

    // 'True' if destructor should not free the tree nodes
    bool no_free_in_destructor{ false };

    // Two points describe simulation box size of the tree
    Vec3d xyz_min;
    Vec3d xyz_max;

    double acceptance_criterion{ Constants::theta }; // Acceptance criterion
    double sigma{ Constants::sigma }; // Probability parameter
    bool naive_method{ false }; // If true, expand every cell regardless of whether dendrites are available or not
    size_t level_of_branch_nodes{ Constants::uninitialized };
    size_t max_num_pending_vacant_axons{ Constants::num_pend_vacant }; // Maximum number of vacant axons which are considered at the same time for
        // finding a target neuron

    // Cache with nodes owned by other ranks
    using NodesCacheKey = std::pair<int, OctreeNode*>;
    using NodesCacheValue = OctreeNode*;
    using NodesCache = std::map<NodesCacheKey, NodesCacheValue>;
    NodesCache remote_nodes_cache;

    // Randpm number generator for this class (C++11)
    std::mt19937& random_number_generator;
    // Random number distribution used together with "random_number_generator" (C++11)
    // Uniform distribution for interval [0, 1] (see constructor for initialization)
    std::uniform_real_distribution<double> random_number_distribution;
};
