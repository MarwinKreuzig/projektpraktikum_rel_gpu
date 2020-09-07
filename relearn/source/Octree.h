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
template<class NeuronModels, class Axons, class DendritesExc, class DendritesInh>
class Neurons;

class Partition;

class Octree {
public:
	friend class Partition;

private:
	Octree();

public:
	Octree(const Partition& part, const Parameters& params);

	~Octree();

	// See Octree.cpp for full declaration
	void find_target_neurons(MapSynapseCreationRequests&, Neurons<NeuronModels, SynapticElements, SynapticElements, SynapticElements>&);

	/**
	 * Neuron ID for parent (inner node)
	 * Used to easily recognize inner nodes during debugging
	 */
	const size_t NEURON_ID_PARENT = 111222333444;

	typedef std::vector<bool> AccessEpochsStarted;

	/**
	 * Type for list elements used to create probability subinterval
	 */
	struct ProbabilitySubinterval {
		ProbabilitySubinterval() :
			ptr(nullptr),
			probability(0),
			mpi_request(MPI_REQUEST_NULL),
			request_rank(-1) {
		}

		ProbabilitySubinterval(OctreeNode* node) :
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
		VacantAxon(size_t neuron_id, double x, double y, double z, Cell::DendriteType dendrite_type_needed) :
			xyz_pos{ x, y, z },
			neuron_id(neuron_id),
			dendrite_type_needed(dendrite_type_needed)
		{}

		ProbabilitySubintervalList nodes_accepted;
		ProbabilitySubintervalList nodes_to_visit;
		size_t neuron_id;
		double xyz_pos[3];
		Cell::DendriteType dendrite_type_needed;
	};
	typedef std::list<VacantAxon*> VacantAxonList;


	// Set simulation box size of the tree
	void set_size(Vec3d min, Vec3d max) {
		xyz_min = min;
		xyz_max = max;
	}

	// Set acceptance criterion for cells in the tree
	void set_acceptance_criterion(double acceptance_criterion) {
		this->acceptance_criterion = acceptance_criterion;
	}

	// Set probability parameter used to determine the probability
	// for a cell of being selected
	void set_probability_parameter(double sigma) {
		this->sigma = sigma;
	}

	// Set naive method parameter used to determine if all cells
	// should be expanded regardless of whether dendrites are available
	void set_naive_method_parameter(bool naive_method) {
		this->naive_method = naive_method;
	}

	void set_mpi_rma_mem_allocator(MPI_RMA_MemAllocator<OctreeNode>* allocator) {
		this->mpi_rma_node_allocator = allocator;
	}

	void set_root_level(size_t root_level) {
		this->root_level = root_level;
	}

	void set_no_free_in_destructor() {
		no_free_in_destructor = true;
	}

	void set_level_of_branch_nodes(size_t level) {
		level_of_branch_nodes = level;
	}

	void set_max_num_pending_vacant_axons(size_t max) {
		max_num_pending_vacant_axons = max;
	}

	// Insert neuron into the tree
	OctreeNode* insert(double pos_x,
		double pos_y,
		double pos_z,
		size_t neuron_id,
		int rank) {
		OctreeNode* prev = nullptr;
		OctreeNode* curr = root;
		unsigned char my_idx, idx;
		double xyz_min[3], xyz_max[3];

		// Create new tree node for the neuron
		OctreeNode* new_node = mpi_rma_node_allocator->newObject(); // new OctreeNode();
		new_node->cell.set_neuron_position(pos_x, pos_y, pos_z, true);
		new_node->cell.set_neuron_id(neuron_id);
		new_node->rank = rank;

		// Tree is empty
		if (nullptr == root) {
			// Init cell size with simulation box size
			new_node->cell.set_size(this->xyz_min[0], this->xyz_min[1], this->xyz_min[2],
				this->xyz_max[0], this->xyz_max[1], this->xyz_max[2]);

			// Init root with tree's root level
			new_node->level = root_level;
			root = new_node;
		}
		else {
			// Correct position for new node not found yet
			while (nullptr != curr) {
				/**
				 * My parent already exists.
				 * Calc which child to follow, i.e., determine octant
				 */
				my_idx = curr->cell.get_octant_for_position(pos_x, pos_y, pos_z);

				prev = curr;
				curr = curr->children[my_idx];
			}

			/**
			 * Found my octant, but
			 * I'm the very first child of that node.
			 * I.e., the node is a leaf.
			 */
			if (!prev->is_parent) {
				bool valid_pos;
				double x, y, z;
				size_t neuron_id;

				do {
					/**
					 * Make node containing my octant a parent by
					 * adding neuron in this node as child node
					 */

					 // Determine octant for neuron
					idx = prev->cell.get_neuron_octant();
					prev->children[idx] = mpi_rma_node_allocator->newObject();  // new OctreeNode();

					/**
					 * Init this new node properly
					 */
					 // Cell size
					prev->cell.get_size_for_octant(idx, xyz_min, xyz_max);

					prev->children[idx]->cell.set_size(xyz_min[0], xyz_min[1], xyz_min[2],
						xyz_max[0], xyz_max[1], xyz_max[2]);
					// Neuron position
					prev->cell.get_neuron_position(&x, &y, &z, &valid_pos);
					prev->children[idx]->cell.set_neuron_position(x, y, z, valid_pos);

					// Neuron ID
					neuron_id = prev->cell.get_neuron_id();
					prev->children[idx]->cell.set_neuron_id(neuron_id);
					/**
					 * Set neuron ID of parent (inner node) to NEURON_ID_PARENT.
					 * It is not used for inner nodes.
					 */
					prev->cell.set_neuron_id(NEURON_ID_PARENT);
					prev->is_parent = 1;  // Mark node as parent

					// MPI rank who owns this node
					prev->children[idx]->rank = prev->rank;

					// New node is one level below
					prev->children[idx]->level = prev->level + 1;

					// Determine my octant
					my_idx = prev->cell.get_octant_for_position(pos_x, pos_y, pos_z);

					if (my_idx == idx) {
						prev = prev->children[idx];
					}
				} while (my_idx == idx);
			}

			/**
			 * Found my position in children array,
			 * add myself to the array now
			 */
			prev->children[my_idx] = new_node;
			new_node->level = prev->level + 1;  // Now we know level of me

			// Init cell size
			prev->cell.get_size_for_octant(my_idx, xyz_min, xyz_max);
			prev->children[my_idx]->cell.set_size(xyz_min[0], xyz_min[1], xyz_min[2],
				xyz_max[0], xyz_max[1], xyz_max[2]);
		}
		assert(new_node);
		return new_node;
	}

	// Insert an octree node with its subtree into the tree
	void insert(OctreeNode* node_to_insert) {
		size_t target_level, next_level;
		double cell_xyz_min[3], cell_xyz_max[3], cell_xyz_mid[3];
		double cell_length_half;
		OctreeNode* curr;
		unsigned char my_idx;
		bool done = false;

		// Calc midpoint of node's cell
		node_to_insert->cell.get_size(&cell_xyz_min[0], &cell_xyz_min[1], &cell_xyz_min[2],
			&cell_xyz_max[0], &cell_xyz_max[1], &cell_xyz_max[2]);
		cell_length_half = node_to_insert->cell.get_length() / 2;
		cell_xyz_mid[0] = cell_xyz_min[0] + cell_length_half;
		cell_xyz_mid[1] = cell_xyz_min[1] + cell_length_half;
		cell_xyz_mid[2] = cell_xyz_min[2] + cell_length_half;

		// Level at which to insert the node
		target_level = node_to_insert->level;

		assert(target_level >= 0);

		// Tree is empty
		if (nullptr == root) {
			// Node should become root of the tree
			if (root_level == target_level) {
				root = node_to_insert;
				done = true;

				// NOTE: We assume that the tree's and the node's
				// box size are the same. That's why we don't set the tree's
				// box size explicitly here.

				//LogMessages::print_debug("ROOT: Me as root inserted.");
			}
			// Create tree's root
			else {
				// Create root node
				root = mpi_rma_node_allocator->newObject();

				// Init octree node
				root->rank = MPIInfos::my_rank;
				root->level = root_level;
				root->is_parent = 1;  // node will become parent

				// Init cell in octree node
				// cell size becomes tree's box size
				root->cell.set_size(this->xyz_min[0], this->xyz_min[1], this->xyz_min[2],
					this->xyz_max[0], this->xyz_max[1], this->xyz_max[2]);
				root->cell.set_neuron_id(NEURON_ID_PARENT);

				//LogMessages::print_debug("ROOT: new node as root inserted.");
			}
		}
		curr = root;
		next_level = curr->level + 1; // next_level is the current level we consider for inserting the node
									  // It's called next_level as it is the next level below the current node
									  // "curr" in the tree

		while (!done) {
			/**
			 * My parent already exists.
			 * Calc which child to follow, i.e., determine
			 * my octant (index in the children array)
			 * based on the midpoint of my cell
			 */
			my_idx = curr->cell.get_octant_for_position(cell_xyz_mid[0], cell_xyz_mid[1], cell_xyz_mid[2]);

			// Target level reached, so insert me
			if (next_level == target_level) {
				//LogMessages::print_debug("Target level reached.");

				// Make sure that no other node is already
				// on my index in the children array
				//
				// NOTE:
				// This assertion is not valid anymore as the same branch nodes
				// are inserted repeatedly at the same position
				// assert(curr->children[my_idx] == nullptr);

				curr->children[my_idx] = node_to_insert;
				done = true;

				//LogMessages::print_debug("  Target level reached... inserted me");
			}
			// Target level not yet reached
			else {
				//LogMessages::print_debug("Target level not yet reached.");

				// A node exists on my index in the
				// children array, so follow this node.
				if (curr->children[my_idx]) {
					curr = curr->children[my_idx];
					//LogMessages::print_debug("  I follow node on my index.");
				}
				// New node must be created which
				// I can then follow
				else {
					//LogMessages::print_debug("  New node must be created which I can then follow.");
					OctreeNode* new_node;
					double new_node_xyz_min[3], new_node_xyz_max[3];

					//LogMessages::print_debug("    Trying to allocate node.");
					// Create node
					new_node = mpi_rma_node_allocator->newObject();
					//LogMessages::print_debug("    Node allocated.");

					// Init octree node
					new_node->rank = MPIInfos::my_rank;
					new_node->level = next_level;
					new_node->is_parent = 1;  // node will become parent

					// Init cell in octree node
					// cell size becomes size of new node's octant
					curr->cell.get_size_for_octant(my_idx, new_node_xyz_min, new_node_xyz_max);
					new_node->cell.set_size(new_node_xyz_min[0], new_node_xyz_min[1], new_node_xyz_min[2],
						new_node_xyz_max[0], new_node_xyz_max[1], new_node_xyz_max[2]);
					new_node->cell.set_neuron_id(NEURON_ID_PARENT);

					curr->children[my_idx] = new_node;
					curr = new_node;
				}
				next_level++;
			} // target level not yet reached
		} // while
	}

	void print() {
		postorder_print();
	}

	void free() {
		// Provide allocator so that it can be used to free memory again
		FunctorFreeNode free_node(*mpi_rma_node_allocator);

		// The functor containing the visit function is of type FunctorFreeNode
		tree_walk_postorder<FunctorFreeNode>(free_node);
	}

	void update(double* dendrites_exc_cnts, double* dendrites_exc_connected_cnts,
		double* dendrites_inh_cnts, double* dendrites_inh_connected_cnts,
		size_t num_neurons) {
		// Init parameters to be used in function object
		FunctorUpdateNode update_node(dendrites_exc_cnts, dendrites_exc_connected_cnts,
			dendrites_inh_cnts, dendrites_inh_connected_cnts, num_neurons);

		// The functor containing the visit function is of type FunctorUpdateNode
		tree_walk_postorder<FunctorUpdateNode>(update_node);
	}

	// The caller must ensure that only inner nodes are visited.
	// "max_level" must be chosen correctly for this
	void update_from_level(size_t max_level) {
		FunctorUpdateNode update_node;

		/**
		 * NOTE: It *must* be ensured that in tree_walk_postorder() only inner nodes
		 * are visited as update_node cannot update leaf nodes here
		 */

		 // The functor containing the visit function is of type FunctorUpdateNode
		tree_walk_postorder<FunctorUpdateNode>(update_node, max_level);
	}

	bool find_target_neuron(size_t src_neuron_id, double axon_pos_xyz[3], Cell::DendriteType dendrite_type_needed, size_t& target_neuron_id, int& target_rank) {
		std::list<ProbabilitySubinterval*> list;
		std::list<ProbabilitySubinterval*>::iterator it;
		OctreeNode* node_selected;
		OctreeNode* root_of_subtree = root;
		bool done, found;

		do {
			/**
			 * Create list with nodes that have at least one dendrite and are
			 * precise enough given the position of an axon
			 */
			get_nodes_for_interval(axon_pos_xyz, root_of_subtree, dendrite_type_needed, list, naive_method);

			/**
			 * Assign a probability to each node in the list.
			 * The probability for connecting to the same neuron (i.e., the axon's neuron) is set 0.
			 * Nodes with 0 probability are removed.
			 * The probabilities of all list elements sum up to 1.
			 */
			create_interval(src_neuron_id, axon_pos_xyz, dendrite_type_needed, list);

			/**
			 * Select node with target neuron
			 */
			select_subinterval(list, node_selected);

			// Clear list for next interval creation
			for (it = list.begin(); it != list.end(); ) {
				delete* it;
				it = list.erase(it);
			}

			/**
			 * Leave loop if no node was selected OR
			 * the selected node is leaf node, i.e., contains normal neuron.
			 *
			 * No node is selected when all nodes in the interval, created in
			 * get_nodes_for_interval(), have probability 0 to connect.
			 */
			done = (nullptr == node_selected) || (!node_selected->is_parent);

			// Update root of subtree
			root_of_subtree = node_selected;
		} while (!done);

		// Return neuron ID and rank of target neuron
		if ((found = (nullptr != node_selected))) {
			target_neuron_id = node_selected->cell.get_neuron_id();
			target_rank = node_selected->rank;
		}

		return found;
	}

	OctreeNode* get_root() {
		return root;
	}

	size_t get_level_of_branch_nodes() {
		return level_of_branch_nodes;
	}

	void empty_remote_nodes_cache() {
		NodesCache::iterator it;

		for (it = remote_nodes_cache.begin(); it != remote_nodes_cache.end(); it++) {
			mpi_rma_node_allocator->deleteObject(it->second);
			remote_nodes_cache.erase(it);
		}
	}


private:
	/**
	 * Type for stack used in postorder tree walk
	 */
	struct StackElement {
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
			size_t num_neurons) :
			dendrites_exc_cnts(dendrites_exc_cnts),
			dendrites_exc_connected_cnts(dendrites_exc_connected_cnts),
			dendrites_inh_cnts(dendrites_inh_cnts),
			dendrites_inh_connected_cnts(dendrites_inh_connected_cnts),
			num_neurons(num_neurons) {
		}
		FunctorUpdateNode() {}

		void operator()(OctreeNode* node) {
			// I'm inner node, i.e., I have a super neuron
			if (node->is_parent) {
				unsigned int num_dendrites_exc, num_dendrites_inh;
				unsigned int temp_num_dendrites_exc, temp_num_dendrites_inh;
				unsigned int divisor_pos_exc, divisor_pos_inh;
				double temp_xyz_pos_exc[3];
				double temp_xyz_pos_inh[3];
				double xyz_pos_exc[3] = { 0., 0., 0. };
				double xyz_pos_inh[3] = { 0., 0., 0. };
				bool valid_pos_exc, valid_pos_inh;
				int i, j;

				// Sum of number of dendrites of all my children
				num_dendrites_exc = num_dendrites_inh = 0;

				// For all my children
				for (i = 0; i < 8; i++) {
					if (node->children[i]) {
						// Sum up number of dendrites
						temp_num_dendrites_exc = node->children[i]->cell.get_neuron_num_dendrites_exc();
						temp_num_dendrites_inh = node->children[i]->cell.get_neuron_num_dendrites_inh();
						num_dendrites_exc += temp_num_dendrites_exc;
						num_dendrites_inh += temp_num_dendrites_inh;

						// Average the position by using the number of dendrites as weights
						node->children[i]->cell.get_neuron_position_exc(&temp_xyz_pos_exc[0], &temp_xyz_pos_exc[1], &temp_xyz_pos_exc[2], &valid_pos_exc);
						node->children[i]->cell.get_neuron_position_inh(&temp_xyz_pos_inh[0], &temp_xyz_pos_inh[1], &temp_xyz_pos_inh[2], &valid_pos_inh);
						/**
						 * We can use position if it's valid or if corresponding num of dendrites is 0 (due to multiplying position with 0)
						 */
						assert(valid_pos_exc || (0 == temp_num_dendrites_exc));
						assert(valid_pos_inh || (0 == temp_num_dendrites_inh));
						for (j = 0; j < 3; j++) {
							xyz_pos_exc[j] += (double)temp_num_dendrites_exc * temp_xyz_pos_exc[j];
							xyz_pos_inh[j] += (double)temp_num_dendrites_inh * temp_xyz_pos_inh[j];
						}
					}
				}
				/**
				 * For calculating the new weighted position, make sure that we don't
				 * divide by 0. This happens if the total number of dendrites is 0.
				 */
				divisor_pos_exc = num_dendrites_exc;
				divisor_pos_inh = num_dendrites_inh;
				valid_pos_exc = valid_pos_inh = true;
				if (0 == num_dendrites_exc) {
					valid_pos_exc = false;  // Mark result as invald
					divisor_pos_exc = 1;
				}
				if (0 == num_dendrites_inh) {
					valid_pos_inh = false;  // Mark result as invalid
					divisor_pos_inh = 1;
				}
				// Calc the average by dividing by the total number of dendrites
				for (j = 0; j < 3; j++) {
					xyz_pos_exc[j] /= divisor_pos_exc;
					xyz_pos_inh[j] /= divisor_pos_inh;
				}
				node->cell.set_neuron_num_dendrites_exc(num_dendrites_exc);
				node->cell.set_neuron_num_dendrites_inh(num_dendrites_inh);
				// Also mark if new position is valid using valid_pos_{exc,inh}
				node->cell.set_neuron_position_exc(xyz_pos_exc[0], xyz_pos_exc[1], xyz_pos_exc[2], valid_pos_exc);
				node->cell.set_neuron_position_inh(xyz_pos_inh[0], xyz_pos_inh[1], xyz_pos_inh[2], valid_pos_inh);
			}
			// I'm leaf node, i.e., I have a normal neuron
			else {
				size_t neuron_id;
				unsigned int num_vacant_dendrites_exc;
				unsigned int num_vacant_dendrites_inh;

				// Get ID of the node's neuron
				neuron_id = node->cell.get_neuron_id();

				// Calculate number of vacant dendrites for my neuron
				assert(neuron_id < num_neurons);
				num_vacant_dendrites_exc = (unsigned int)(dendrites_exc_cnts[neuron_id] - dendrites_exc_connected_cnts[neuron_id]);
				num_vacant_dendrites_inh = (unsigned int)(dendrites_inh_cnts[neuron_id] - dendrites_inh_connected_cnts[neuron_id]);

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
		FunctorFreeNode(MPI_RMA_MemAllocator<OctreeNode>& allocator) : allocator(allocator) {}

		void operator()(OctreeNode* node) { allocator.deleteObject(node); }
	private:
		MPI_RMA_MemAllocator<OctreeNode>& allocator;
	};

	/**
	 * Do a postorder tree walk and run the
	 * function "visit" for every node when
	 * it is visited
	 */
	template <typename Functor>
	void tree_walk_postorder(Functor visit, size_t max_level = std::numeric_limits<size_t>::max()) {
		std::stack<StackElement*> stack;
		StackElement* elem, * tmp;
		int i, depth = 0;

		// Tree is not empty
		if (root) {
			// Push node onto stack
			elem = new StackElement;
			elem->ptr = root;
			elem->flag = 0;
			elem->depth = 0;

			stack.push(elem);

			while (!stack.empty()) {

				// Get top-of-stack node
				elem = stack.top();
				depth = (int)elem->depth;

				// Node should be visited now?
				if (elem->flag) {

					assert(elem->ptr->level <= max_level);

					// Apply action to node
					visit(elem->ptr);

					// Pop node from stack
					stack.pop();
					delete elem;
				}
				else {
					// Mark node to be visited next time
					elem->flag = 1;

					// Only push node's children onto stack if
					// they don't exceed "max_level"
					if (depth < max_level) {
						// Push node's children onto stack
						for (i = 7; i >= 0; i--) {
							if (elem->ptr->children[i]) {
								tmp = new StackElement;
								tmp->ptr = elem->ptr->children[i];
								tmp->flag = 0;
								tmp->depth = (size_t)depth + 1;
								stack.push(tmp);
							}
						}
					}
				}
			} /* while */
		}
	}

	/**
	 * Print tree in postorder
	 */
	void postorder_print() {
		std::stack<StackElement*> stack;
		StackElement* elem, * tmp;
		bool pos_valid;
		int i, j, depth = 0;

		// Tree is not empty
		if (root) {
			elem = new StackElement;
			elem->ptr = root;
			elem->flag = 0;
			elem->depth = 0;

			stack.push(elem);

			while (!stack.empty()) {

				elem = stack.top();
				depth = (int)elem->depth;

				// Visit node now
				if (elem->flag) {
					double xyz_min[3], xyz_max[3], xyz_pos[3];

					// Print node's address
					for (j = 0; j < depth; j++) {
						std::cout << " ";
					}
					std::cout << "Address: " << (OctreeNode*)elem->ptr << "\n";

					// Print cell extent
					elem->ptr->cell.get_size(&xyz_min[0], &xyz_min[1], &xyz_min[2], &xyz_max[0], &xyz_max[1], &xyz_max[2]);
					for (j = 0; j < depth; j++) {
						std::cout << " ";
					}
					std::cout << "Cell extent: (" << (double)xyz_min[0] << " .. " << (double)xyz_max[0] << ", "
						<< (double)xyz_min[1] << " .. " << (double)xyz_max[1] << ", "
						<< (double)xyz_min[2] << " .. " << (double)xyz_max[2] << ")\n";

					// Print neuron ID
					for (j = 0; j < depth; j++) {
						std::cout << " ";
					}
					std::cout << "Neuron ID: " << elem->ptr->cell.get_neuron_id() << "\n";

					// Print number of dendrites
					for (j = 0; j < depth; j++) {
						std::cout << " ";
					}
					std::cout << "Number dendrites (exc, inh): (" << elem->ptr->cell.get_neuron_num_dendrites_exc()
						<< ", " << elem->ptr->cell.get_neuron_num_dendrites_inh() << ")\n";

					// Print position EXCITATORY
					elem->ptr->cell.get_neuron_position_exc(&xyz_pos[0], &xyz_pos[1], &xyz_pos[2], &pos_valid);
					for (j = 0; j < depth; j++) {
						std::cout << " ";
					}
					std::cout << "Position exc: (" << (double)xyz_pos[0] << ", " << (double)xyz_pos[1] << ", " << (double)xyz_pos[2] << ") ";
					// Note if position is invalid
					if (!pos_valid) {
						std::cout << "-- invalid!";
					} std::cout << "\n";
					// Print position INHIBITORY
					elem->ptr->cell.get_neuron_position_inh(&xyz_pos[0], &xyz_pos[1], &xyz_pos[2], &pos_valid);
					for (j = 0; j < depth; j++) {
						std::cout << " ";
					}
					std::cout << "Position inh: (" << (double)xyz_pos[0] << ", " << (double)xyz_pos[1] << ", " << (double)xyz_pos[2] << ") ";
					// Note if position is invalid
					if (!pos_valid) {
						std::cout << "-- invalid!";
					} std::cout << "\n";
					std::cout << "\n";

					stack.pop();
					delete elem;
				}
				// Visit children first
				else {
					elem->flag = 1;

					for (j = 0; j < depth; j++) {
						std::cout << " ";
					}
					std::cout << "Child indices: ";
					for (i = 0; i < 8; i++) {
						if (elem->ptr->children[i]) {
							std::cout << i << " ";
						}
					}
					/**
					 * Push in reverse order so that visiting happens in
					 * increasing order of child indices
					 */
					for (i = 7; i >= 0; i--) {
						if (elem->ptr->children[i]) {
							tmp = new StackElement;
							tmp->ptr = elem->ptr->children[i];
							tmp->flag = 0;
							tmp->depth = (size_t)depth + 1;
							stack.push(tmp);
						}
					}
					std::cout << std::endl;
				}
			}
		}
	}

	/**
	 * If we use the naive method accept leaf cells only, otherwise
	 * test if cell has dendrites available and is precise enough.
	 * Returns true if accepted, false otherwise
	 */
	bool acceptance_criterion_test(double axon_pos_xyz[3], OctreeNode* node_with_dendrite,
		Cell::DendriteType dendrite_type_needed, bool naive_method,
		bool& has_vacant_dendrites) {
		double target_xyz[3];
		double distance, length;
		bool pos_valid, ret_val = false;
		int i;

		has_vacant_dendrites = node_with_dendrite->cell.get_neuron_num_dendrites_for(dendrite_type_needed);

		// Use naive method
		if (naive_method) {
			// Accept leaf only
			ret_val = !node_with_dendrite->is_parent;
		}
		else {
			// There are vacant dendrites available
			if (has_vacant_dendrites) {

				/**
				 * Node is leaf node, i.e., not super neuron.
				 * Thus the node is precise. Accept it.
				 */
				if (!node_with_dendrite->is_parent) {
					ret_val = true;
					//std::cout << __func__ << ": leaf node -> no test" << std::endl;
				}
				// Check distance between neuron with axon and neuron with dendrite
				else {
					node_with_dendrite->cell.get_neuron_position_for(dendrite_type_needed, target_xyz, &pos_valid);
					/**
					 * NOTE: This assertion fails when considering inner nodes that don't have dendrites.
					 */
					assert(pos_valid);

					/* Calc Euclidean distance between source and target neuron */
					distance = 0;
					for (i = 0; i < 3; i++) {
						distance += (target_xyz[i] - axon_pos_xyz[i]) * (target_xyz[i] - axon_pos_xyz[i]);
					}
					distance = sqrt(distance);
					length = node_with_dendrite->cell.get_length();

					//std::cout << __func__ << ": l / d: " << length/distance << std::endl;
					// Original Barnes-Hut acceptance criterion
					ret_val = (length / distance < acceptance_criterion);
				}
			}
			else {
				//std::cout << __func__ << ": no vacant dendrites -> no test" << std::endl;
			}
		}
		return ret_val;
	}

	/**
	 * Returns list with nodes for creating the probability interval
	 */
	void get_nodes_for_interval(double axon_pos_xyz[3], OctreeNode* root, Cell::DendriteType dendrite_type_needed,
		std::list<ProbabilitySubinterval*>& list, bool naive_method) {
		std::stack<OctreeNode*> stack;
		OctreeNode* stack_elem, * local_child_addr, * local_children[8];
		ProbabilitySubinterval* list_elem;
		const MPI_Aint* base_pointers = mpi_rma_node_allocator->get_base_pointers();
		MPI_Aint target_child_displ;
		NodesCacheKey rank_addr_pair;
		std::pair<NodesCache::iterator, bool> ret;
		std::pair<NodesCacheKey, NodesCacheValue> cache_key_val_pair;
		int i, target_rank;

		/* Subtree is not empty AND (Dendrites are available OR We use naive method) */
		if (root && (root->cell.get_neuron_num_dendrites_for(dendrite_type_needed) || naive_method)) {

			/**
			 * The root node is parent (i.e., contains a super neuron) and thus cannot be the target neuron.
			 * So, start considering its children.
			 */
			if (root->is_parent) {
				// Node is owned by this rank
				if (node_is_local(*root)) {
					// Push root's children onto stack
					for (i = 7; i >= 0; i--) {
						if (root->children[i]) {
							stack.push(root->children[i]);
						}
					}
				}
				// Node is owned by different rank
				else {
					target_rank = root->rank;
					rank_addr_pair.first = target_rank;

					// Start access epoch to remote rank
					MPI_Win_lock(MPI_LOCK_SHARED, target_rank, MPI_MODE_NOCHECK, mpi_rma_node_allocator->mpi_window);

					// Fetch remote children if they exist
					for (i = 7; i >= 0; i--) {
						if (nullptr != root->children[i]) {
							rank_addr_pair.second = root->children[i];

							cache_key_val_pair.first = rank_addr_pair;
							cache_key_val_pair.second = nullptr;

							// Get cache entry for "cache_key_val_pair"
							// It is created if it does not exist yet
							ret = remote_nodes_cache.insert(cache_key_val_pair);

							// Cache entry just inserted as it was not in cache
							// So, we still need to init the entry by fetching
							// from the target rank
							if (ret.second == true) {
								ret.first->second = mpi_rma_node_allocator->newObject();
								local_child_addr = ret.first->second;

								// Calc displacement from absolute address
								target_child_displ = (MPI_Aint)root->children[i] - base_pointers[target_rank];

								/*
								MPI_Request mpi_request;
								MPIX_Rget((char *) local_child_addr, sizeof(OctreeNode), MPI_CHAR,
										target_rank, target_child_displ, sizeof(OctreeNode), MPI_CHAR,
										mpi_rma_node_allocator->mpi_window, &mpi_request);
								MPI_Wait(&mpi_request, MPI_STATUS_IGNORE);
								*/
								MPI_Get((char*)local_child_addr, sizeof(OctreeNode), MPI_CHAR,
									target_rank, target_child_displ, sizeof(OctreeNode), MPI_CHAR,
									mpi_rma_node_allocator->mpi_window);
							}

							// Remember address of node
							local_children[i] = ret.first->second;
						}
						else {
							local_children[i] = nullptr;
						}
					}

					// Complete access epoch
					MPI_Win_unlock(target_rank, mpi_rma_node_allocator->mpi_window);

					// Push root's children onto stack
					for (i = 7; i >= 0; i--) {
						if (local_children[i]) {
							stack.push(local_children[i]);
						}
					}
				} // Node owned by different rank
			} // Root of subtree is parent
			else {
				/**
				 * The root node is a leaf and thus contains the target neuron.
				 *
				 * NOTE: Root is not intended to be a leaf but we handle this as well.
				 * Without pushing root onto the stack, it would not make it into the "list" of nodes.
				 */
				stack.push(root);
			}

			bool has_vacant_dendrites;
			while (!stack.empty()) {
				// Get top-of-stack node and remove it from stack
				stack_elem = stack.top();
				stack.pop();

				/**
				 * Should node be used for probability interval?
				 *
				 * Only take those that have dendrites available
				 */
				if (acceptance_criterion_test(axon_pos_xyz, stack_elem, dendrite_type_needed, naive_method,
					has_vacant_dendrites)) {
					//std::cout << "accepted: " << stack_elem->cell.get_neuron_id() << std::endl;
					// Insert node into list
					list_elem = new ProbabilitySubinterval;
					assert(list_elem);
					list_elem->ptr = stack_elem;
					list.push_back(list_elem);
				}
				else if (has_vacant_dendrites || naive_method) {
					// Node is owned by this rank
					if (node_is_local(*stack_elem)) {
						// Push node's children onto stack
						for (i = 7; i >= 0; i--) {
							if (stack_elem->children[i]) {
								stack.push(stack_elem->children[i]);
							}
						}
					}
					// Node is owned by different rank
					else {
						target_rank = stack_elem->rank;
						rank_addr_pair.first = target_rank;

						// Start access epoch to remote rank
						MPI_Win_lock(MPI_LOCK_SHARED, target_rank, MPI_MODE_NOCHECK, mpi_rma_node_allocator->mpi_window);

						// Fetch remote children if they exist
						for (i = 7; i >= 0; i--) {
							if (nullptr != stack_elem->children[i]) {
								rank_addr_pair.second = stack_elem->children[i];

								cache_key_val_pair.first = rank_addr_pair;
								cache_key_val_pair.second = nullptr;

								// Get cache entry for "rank_addr_pair"
								// It is created if it does not exist yet
								ret = remote_nodes_cache.insert(cache_key_val_pair);

								// Cache entry just inserted as it was not in cache
								// So, we still need to init the entry by fetching
								// from the target rank
								if (ret.second == true) {
									ret.first->second = mpi_rma_node_allocator->newObject();
									local_child_addr = ret.first->second;

									// Calc displacement from absolute address
									target_child_displ = (MPI_Aint)stack_elem->children[i] - base_pointers[target_rank];

									/*
									MPI_Request mpi_request;
									MPIX_Rget((char *) local_child_addr, sizeof(OctreeNode), MPI_CHAR,
											target_rank, target_child_displ, sizeof(OctreeNode), MPI_CHAR,
											mpi_rma_node_allocator->mpi_window, &mpi_request);
									MPI_Wait(&mpi_request, MPI_STATUS_IGNORE);
									 */

									MPI_Get((char*)local_child_addr, sizeof(OctreeNode), MPI_CHAR,
										target_rank, target_child_displ, sizeof(OctreeNode), MPI_CHAR,
										mpi_rma_node_allocator->mpi_window);
								}

								// Remember local address of node
								local_children[i] = ret.first->second;
							}
							else {
								local_children[i] = nullptr;
							}
						}

						// Complete access epoch
						MPI_Win_unlock(target_rank, mpi_rma_node_allocator->mpi_window);

						// Push node's children onto stack
						for (i = 7; i >= 0; i--) {
							if (local_children[i]) {
								stack.push(local_children[i]);
							}
						}
					} // Node owned by different rank

					/*
					 Old code without RMA
					//std::cout << "not accepted: " << stack_elem->cell.get_neuron_id() << std::endl;
					// Push node's children onto stack
					for (i = 7; i >= 0; i--) {
						if (stack_elem->children[i]) {
							stack.push(stack_elem->children[i]);
							//std::cout << __func__ << ": child pushed: " << i << std::endl;
						}
					}
					*/
				} // Acceptance criterion rejected
			} // while
		}
	}

	/**
	 * Returns probability interval, i.e., list with nodes where each node is assigned a probability.
	 * Nodes with probability 0 are removed from the list.
	 * The probabilities sum up to 1
	 */
	void create_interval(size_t src_neuron_id, double axon_pos_xyz[3], Cell::DendriteType dendrite_type_needed, std::list<ProbabilitySubinterval*>& list) {
		double sum = 0;
		std::list<ProbabilitySubinterval*>::iterator it;

		// Does list contain nodes?
		if (!list.empty()) {

			for (it = list.begin(); it != list.end(); it++) {
				(*it)->probability = calc_attractiveness_to_connect(src_neuron_id, axon_pos_xyz, *((*it)->ptr), dendrite_type_needed);
				sum += (*it)->probability;
			}
			/**
			 * Make sure that we don't divide by 0 in case
			 * all probabilities from above are 0.
			 */
			sum = (sum == 0) ? 1 : sum;
			//std::cout << __func__ << ": sum: " << (double) sum << std::endl;

			// Norm the values to [0,1] and thus get probabilities
			for (it = list.begin(); it != list.end();) {
				(*it)->probability /= sum;

				//std::cout << __func__ << ": probability: " << (double) (*it)->probability << " neuron: " << (*it)->ptr->cell.get_neuron_id() << std::endl;

				// Remove node that has probability 0
				if ((*it)->probability == 0) { // We want exact comparison of double to constant 0 here
					delete* it;                // Free memory for list element structure
					it = list.erase(it);       // "it" points now to successor element
					//std::cout << __func__ << ": probability == 0, node removed" << std::endl;
				}
				else {
					it++;
				}
			}
		}
	}

	/**
	 * Returns attractiveness for connecting two given nodes
	 * NOTE: This is not a probability yet as it could be >1
	 */
	double calc_attractiveness_to_connect(size_t src_neuron_id, double axon_pos_xyz[3], const OctreeNode& node_with_dendrite, Cell::DendriteType dendrite_type_needed) {
		double target_xyz[3];
		double numerator;
		double num_dendrites;
		double ret_val;
		bool pos_valid;
		int i;

		/**
		 * If the axon's neuron itself is considered as target neuron, set attractiveness to 0 to avoid forming an autapse (connection to itself).
		 * This can be done as the axon's neuron cells are always resolved until the normal (vs. super) axon's neuron is reached.
		 * That is, the dendrites of the axon's neuron are not included in any super neuron considered.
		 * However, this only works under the requirement that "acceptance_criterion" is <= 0.5.
		 */
		if ((!node_with_dendrite.is_parent) && (src_neuron_id == node_with_dendrite.cell.get_neuron_id())) {
			ret_val = 0;
			//std::cout << __func__ << ": neuron_id: " << src_neuron_id << " (src == tgt)" << std::endl;
		}
		else {
			node_with_dendrite.cell.get_neuron_position_for(dendrite_type_needed, target_xyz, &pos_valid);
			assert(pos_valid);
			num_dendrites = (double)node_with_dendrite.cell.get_neuron_num_dendrites_for(dendrite_type_needed);

			numerator = 0;
			for (i = 0; i < 3; i++) {
				numerator += (target_xyz[i] - axon_pos_xyz[i]) * (target_xyz[i] - axon_pos_xyz[i]);
			}

			// Criterion from Markus' paper with doi: 10.3389/fnsyn.2014.00007
			ret_val = (num_dendrites * exp(-numerator / (sigma * sigma)));
		}
		return ret_val;
	}

	/**
	 * Randomly select node from probability interval
	 */
	void select_subinterval(std::list<ProbabilitySubinterval*>& list, OctreeNode*& node_selected) {
		std::list<ProbabilitySubinterval*>::iterator it;
		double random_number, sum_probabilities;

		node_selected = nullptr;

		// Does list contain nodes?
		if (!list.empty()) {
			// Draw random number from [0,1]
			random_number = random_number_distribution(random_number_generator);

			/**
			 * Also check for it != list.end() to account for that, due to numeric inaccuracies in summation,
			 * it might happen that random_number > sum_probabilities in the end
			 */
			it = list.begin();
			sum_probabilities = (*it)->probability;
			it++; // Point to second element
			while (random_number > sum_probabilities && it != list.end()) {
				sum_probabilities += (*it)->probability;
				it++;
			}
			it--; // Undo it++ before or in loop to get correct subinterval
			node_selected = (*it)->ptr;
		}
	}

	inline bool node_is_local(const OctreeNode& node) {
		return node.rank == MPIInfos::my_rank;
	}

	void append_node(OctreeNode* node, ProbabilitySubintervalList& list);
	void append_children(OctreeNode* node, ProbabilitySubintervalList& list, AccessEpochsStarted& epochs_started);

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
