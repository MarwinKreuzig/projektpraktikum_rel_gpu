/*
 * File:   NetworkGraph.h
 * Author: rinke
 *
 * Created on May 2, 2016
 */

#ifndef NETWORKGRAPH_H
#define NETWORKGRAPH_H

#include <map>
#include <vector>
#include <set>
#include <string>
#include <ostream>

#include "Vec3.h"

class Partition;
class NeuronIdMap;

 /**
  * Network graph stores in and out edges for the neuron id range
  * an MPI process is responsible for: [my_neuron_id_start, my_neuron_id_end].
  */
class NetworkGraph {
public:
	/**
	 * Type definitions
	 */
	typedef std::pair<int, size_t> EdgesKey;    // Pair of (rank, neuron id)
	typedef int EdgesVal;
	typedef std::map<EdgesKey, EdgesVal> Edges; // Map of neuron id to edge weight
	struct Neighbors {                          // Neighbors of a neuron given by in and out edges
		Edges in_edges;
		Edges out_edges;
	};
	typedef std::vector<Neighbors> NeuronNeighborhood; // Neighbors for each neuron
													   // The index into the vector is the neuron's local id


	NetworkGraph(size_t my_num_neurons);

	// Return in edges of neuron "neuron_id"
	const Edges& get_in_edges(size_t neuron_id) const noexcept;

	// Return out edges of neuron "neuron_id"
	const Edges& get_out_edges(size_t neuron_id) const noexcept;

	/**
	 * Add weight to an edge.
	 * The edge is created if it does not exist yet and parameter weight != 0.
	 * The weight is added to the current weight.
	 * The edge is deleted if its weight becomes 0.
	 */
	void add_edge_weight(size_t target_neuron_id, int target_rank,
		size_t source_neuron_id, int source_rank,
		int weight);


	void add_edge(Edges& edges, int rank, int neuron_id, int weight);

	void add_edge_weights(const std::string& filename, const NeuronIdMap& neuron_id_map);

	// Print network using global neuron ids
	void print(std::ostream& os, const NeuronIdMap& neuron_id_map) const;

	void add_edges_from_file(const std::string& path_synapses, const std::string& path_neurons, const NeuronIdMap& neuron_id_map, const Partition& partition);

	void translate_global_to_local(const std::set<size_t>& global_ids, const std::map<size_t, size_t>& id_to_rank, const Partition& partition, std::map<size_t, size_t>& global_id_to_local_id);

	void load_neuron_positions(const std::string& path_neurons, std::set<size_t>& foreing_ids, std::map<size_t, Vec3d>& id_to_pos);

	void load_synapses(const std::string& path_synapses, const Partition& partition, std::set<size_t>& foreing_ids, std::vector<std::tuple<size_t, size_t, int>>& local_synapses, std::vector<std::tuple<size_t, size_t, int>>& out_synapses, std::vector<std::tuple<size_t, size_t, int>>& in_synapses);

	void write_synapses_to_file(const std::string& filename, const NeuronIdMap& neuron_id_map, const Partition& partition);


private:
	NeuronNeighborhood neuron_neighborhood;  // Neurons with their neighbors
	size_t my_num_neurons;                   // My number of neurons
	size_t my_neuron_id_start;               // Start neuron id I am allowed to create a neuron for in the graph
	size_t my_neuron_id_end;                 // End neuron id I am allowed to create a neuron for in the graph
											 // NOTE: This is necessary as only the MPI process which contains these neurons
											 // has full knowledge of their in and out edges. Other processes could only know
											 // some of their edges. That is why it does not make sense to store incomplete information
											 // on other processes
};

#endif /* NETWORKGRAPH_H */
