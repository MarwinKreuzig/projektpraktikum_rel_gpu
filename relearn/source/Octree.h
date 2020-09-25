/*
 * File:   Octree.h
 * Author: rinke
 *
 * Created on October 10, 2014
 */

#ifndef OCTREE_H
#define	OCTREE_H

#include <cstddef>
#include <cmath>
#include <iostream>
#include <stack>
#include <map>
#include <list>
#include <limits>
#include <random>
#include <sstream>

#include "Parameters.h"
#include "OctreeNode.h"
#include "MPI_RMA_MemAllocator.h"
#include "Cell.h"
#include "MPIInfos.h"
#include "LogMessages.h"
#include "NeuronModels.h"
#include "SynapticElements.h"
#include "Vec3.h"

 // Forward declarations of Neurons.h
class SynapseCreationRequests;
typedef std::map<int, SynapseCreationRequests> MapSynapseCreationRequests;

class Neurons;

class Partition;

class Octree {
public:
	friend class Partition;

	typedef std::vector<bool> AccessEpochsStarted;

	/**
	 * Type for list elements used to create probability subinterval
	 */
	struct ProbabilitySubinterval {
		ProbabilitySubinterval() noexcept :
			ptr(nullptr),
			probability(0),
			mpi_request(MPI_REQUEST_NULL),
			request_rank(-1) {
		}

		ProbabilitySubinterval(OctreeNode* node) noexcept :
			ptr(node),
			probability(0),
			mpi_request(MPI_REQUEST_NULL),
			request_rank(-1) {
		}

		OctreeNode* ptr;
		double probability;
		MPI_Request mpi_request;
		int request_rank;
	};
	typedef std::list<ProbabilitySubinterval*> ProbabilitySubintervalList;

	/**
	 * Type for vacant axon for which a target neuron needs to be found
	 */
	struct VacantAxon {
		VacantAxon(size_t neuron_id, const Vec3d& pos, Cell::DendriteType dendrite_type_needed) :
			neuron_id(neuron_id),
			xyz_pos(pos),
			dendrite_type_needed(dendrite_type_needed) {
		}

		ProbabilitySubintervalList nodes_accepted;
		ProbabilitySubintervalList nodes_to_visit;
		size_t neuron_id;
		Vec3d xyz_pos;
		Cell::DendriteType dendrite_type_needed;
	};
	typedef std::list<VacantAxon*> VacantAxonList;

private:
	/**
	 * Type for stack used in postorder tree walk
	 */
	struct StackElement {

		StackElement(OctreeNode* oct_ptr, bool flag, size_t depth) noexcept :
			ptr(oct_ptr), flag(flag), depth(depth) {

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
		FunctorUpdateNode(double* dendrites_exc_cnts, double* dendrites_exc_connected_cnts,
			double* dendrites_inh_cnts, double* dendrites_inh_connected_cnts,
			size_t num_neurons) noexcept :
			dendrites_exc_cnts(dendrites_exc_cnts),
			dendrites_exc_connected_cnts(dendrites_exc_connected_cnts),
			dendrites_inh_cnts(dendrites_inh_cnts),
			dendrites_inh_connected_cnts(dendrites_inh_connected_cnts),
			num_neurons(num_neurons) {
		}
		FunctorUpdateNode() noexcept {}

		void operator()(OctreeNode* node) noexcept {
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
				for (auto i = 0; i < 8; i++) {
					if (node->children[i]) {
						// Sum up number of dendrites
						auto temp_num_dendrites_exc = node->children[i]->cell.get_neuron_num_dendrites_exc();
						auto temp_num_dendrites_inh = node->children[i]->cell.get_neuron_num_dendrites_inh();
						num_dendrites_exc += temp_num_dendrites_exc;
						num_dendrites_inh += temp_num_dendrites_inh;

						// Average the position by using the number of dendrites as weights
						node->children[i]->cell.get_neuron_position_exc(temp_xyz_pos_exc, valid_pos_exc);
						node->children[i]->cell.get_neuron_position_inh(temp_xyz_pos_inh, valid_pos_inh);
						/**
						 * We can use position if it's valid or if corresponding num of dendrites is 0 (due to multiplying position with 0)
						 */
						assert(valid_pos_exc || (0 == temp_num_dendrites_exc));
						assert(valid_pos_inh || (0 == temp_num_dendrites_inh));
						for (auto j = 0; j < 3; j++) {
							xyz_pos_exc[j] += static_cast<double>(temp_num_dendrites_exc) * temp_xyz_pos_exc[j];
							xyz_pos_inh[j] += static_cast<double>(temp_num_dendrites_inh) * temp_xyz_pos_inh[j];
						}
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
					valid_pos_exc = false;  // Mark result as invald
					divisor_pos_exc = 1;
				}
				if (0 == num_dendrites_inh) {
					valid_pos_inh = false;  // Mark result as invalid
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
				assert(neuron_id < num_neurons);
				const unsigned int num_vacant_dendrites_exc = static_cast<unsigned int>(dendrites_exc_cnts[neuron_id] - dendrites_exc_connected_cnts[neuron_id]);
				const unsigned int num_vacant_dendrites_inh = static_cast<unsigned int>(dendrites_inh_cnts[neuron_id] - dendrites_inh_connected_cnts[neuron_id]);

				node->cell.set_neuron_num_dendrites_exc(num_vacant_dendrites_exc);
				node->cell.set_neuron_num_dendrites_inh(num_vacant_dendrites_inh);
			}
		}

	private:
		double* dendrites_exc_cnts = nullptr;
		double* dendrites_exc_connected_cnts = nullptr;
		double* dendrites_inh_cnts = nullptr;
		double* dendrites_inh_connected_cnts = nullptr;
		size_t num_neurons = 0;
	};

	/**
	 * Visit function used with postorder tree walk
	 * Frees the node
	 */
	class FunctorFreeNode {
	public:
		// The functor needs to know the allocator before
		// it can use it to free objects with it
		FunctorFreeNode(MPI_RMA_MemAllocator<OctreeNode>& allocator) noexcept : allocator(allocator) {}

		void operator()(OctreeNode* node) { allocator.deleteObject(node); }
	private:
		MPI_RMA_MemAllocator<OctreeNode>& allocator;
	};

private:
	Octree();

public:
	Octree(const Partition& part, const Parameters& params);
	~Octree() noexcept(false);

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

	void set_mpi_rma_mem_allocator(MPI_RMA_MemAllocator<OctreeNode>* allocator) noexcept {
		this->mpi_rma_node_allocator = allocator;
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

	OctreeNode* Octree::get_root() const noexcept {
		return root;
	}

	size_t get_level_of_branch_nodes() const noexcept {
		return level_of_branch_nodes;
	}



	void print();

	void free();

	// Insert neuron into the tree
	OctreeNode* insert(const Vec3d& position, size_t neuron_id, int rank);

	// Insert an octree node with its subtree into the tree
	void insert(OctreeNode* node_to_insert);

	void update(double* dendrites_exc_cnts, double* dendrites_exc_connected_cnts,
		double* dendrites_inh_cnts, double* dendrites_inh_connected_cnts,
		size_t num_neurons);

	// The caller must ensure that only inner nodes are visited.
	// "max_level" must be chosen correctly for this
	void update_from_level(size_t max_level);

	bool find_target_neuron(size_t src_neuron_id, const Vec3d& axon_pos_xyz, Cell::DendriteType dendrite_type_needed, size_t& target_neuron_id, int& target_rank);

	void find_target_neurons(MapSynapseCreationRequests&, Neurons&);

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
				assert(elem.ptr->level <= max_level);

				// Apply action to node
				visit(elem.ptr);

				// Pop node from stack
				stack.pop();
			}
			else {
				// Mark node to be visited next time
				elem.flag = true;

				// Only push node's children onto stack if
				// they don't exceed "max_level"
				if (depth < max_level) {
					// Push node's children onto stack
					for (auto i = 7; i >= 0; i--) {
						if (elem.ptr->children[i]) {
							stack.emplace(elem.ptr->children[i], false, depth + 1);
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
		const OctreeNode* const node_with_dendrite,
		Cell::DendriteType dendrite_type_needed,
		bool naive_method,
		bool& has_vacant_dendrites) const noexcept;

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
	double calc_attractiveness_to_connect(size_t src_neuron_id, const Vec3d& axon_pos_xyz, const OctreeNode& node_with_dendrite, Cell::DendriteType dendrite_type_needed) const noexcept;

	/**
	 * Randomly select node from probability interval
	 */
	OctreeNode* select_subinterval(const ProbabilitySubintervalList& list);

	inline bool node_is_local(const OctreeNode& node) noexcept {
		return node.rank == MPIInfos::my_rank;
	}

	void append_node(OctreeNode* node, ProbabilitySubintervalList& list);
	void append_children(OctreeNode* node, ProbabilitySubintervalList& list, AccessEpochsStarted& epochs_started);


public:
	/**
	 * Neuron ID for parent (inner node)
	 * Used to easily recognize inner nodes during debugging
	 */
	static constexpr const size_t NEURON_ID_PARENT = 111222333444;
private:
	// Root of the tree
	OctreeNode* root;

	// Level which is assigned to the root of the tree (default = 0)
	size_t root_level;

	// 'True' if destructor should not free the tree nodes
	bool no_free_in_destructor;

	// Two points describe simulation box size of the tree
	Vec3d xyz_min;
	Vec3d xyz_max;

	double acceptance_criterion;  // Acceptance criterion
	double sigma;                 // Probability parameter
	bool   naive_method;          // If true, expand every cell regardless of whether dendrites are available or not
	size_t level_of_branch_nodes;
	size_t max_num_pending_vacant_axons;  // Maximum number of vacant axons which are considered at the same time for
										  // finding a target neuron

	// Allocator for MPI passive target sync. memory for tree nodes
	MPI_RMA_MemAllocator<OctreeNode>* mpi_rma_node_allocator;

	// Cache with nodes owned by other ranks
	typedef std::pair<int, OctreeNode*> NodesCacheKey;
	typedef OctreeNode* NodesCacheValue;
	typedef std::map<NodesCacheKey, NodesCacheValue> NodesCache;
	NodesCache remote_nodes_cache;

	// Randpm number generator for this class (C++11)
	std::mt19937& random_number_generator;
	// Random number distribution used together with "random_number_generator" (C++11)
	// Uniform distribution for interval [0, 1] (see constructor for initialization)
	std::uniform_real_distribution<double> random_number_distribution;
};

#endif	/* OCTTREE_H */
