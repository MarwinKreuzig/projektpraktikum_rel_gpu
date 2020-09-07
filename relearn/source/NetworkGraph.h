/*
 * File:   NetworkGraph.h
 * Author: rinke
 *
 * Created on May 2, 2016
 */

#ifndef NETWORKGRAPH_H
#define NETWORKGRAPH_H

#include <map>
#include <fstream>
#include <cassert>
#include <sstream>

#include "MPIInfos.h"
#include "LogMessages.h"
#include "NeuronIdMap.h"


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


	NetworkGraph(size_t my_num_neurons) :
		neuron_neighborhood(my_num_neurons),
		my_num_neurons(my_num_neurons),
		my_neuron_id_start(0),
		my_neuron_id_end(my_num_neurons - 1) {
	}

	// Return in edges of neuron "neuron_id"
	const Edges& get_in_edges(size_t neuron_id) const {
		return neuron_neighborhood[neuron_id].in_edges;
	}

	// Return out edges of neuron "neuron_id"
	const Edges& get_out_edges(size_t neuron_id) const {
		return neuron_neighborhood[neuron_id].out_edges;
	}

	/**
	 * Add weight to an edge.
	 * The edge is created if it does not exist yet and parameter weight != 0.
	 * The weight is added to the current weight.
	 * The edge is deleted if its weight becomes 0.
	 */
	void add_edge_weight(size_t target_neuron_id, int target_rank,
		size_t source_neuron_id, int source_rank,
		int weight) {
		EdgesKey rank_neuron_id_pair;
		int sum;

		// Target neuron is mine
		if (target_rank == MPIInfos::my_rank) {
			rank_neuron_id_pair.first = source_rank;
			rank_neuron_id_pair.second = source_neuron_id;

			Edges& in_edges = neuron_neighborhood[target_neuron_id].in_edges;
			Edges::iterator it_in_edge = in_edges.find(rank_neuron_id_pair);

			// Edge found
			if (it_in_edge != in_edges.end()) {
				// Current edge weight + additional weight
				sum = it_in_edge->second + weight;

				// Edge weight becomes 0, so delete edge
				if (0 == sum) {
					in_edges.erase(it_in_edge);
					// Update edge weight
				}
				else {
					it_in_edge->second = sum;
				}
			}
			// Edge not found
			else {
				// Edge needs to be inserted as its weight != 0
				if (0 != weight) {
					in_edges[rank_neuron_id_pair] = weight;
				}
			}
		} // Target neuron is mine

		// Source neuron is mine
		if (source_rank == MPIInfos::my_rank) {
			rank_neuron_id_pair.first = target_rank;
			rank_neuron_id_pair.second = target_neuron_id;

			Edges& out_edges = neuron_neighborhood[source_neuron_id].out_edges;
			Edges::iterator it_out_edge = out_edges.find(rank_neuron_id_pair);

			// Edge found
			if (it_out_edge != out_edges.end()) {
				// Current edge weight + additional weight
				sum = it_out_edge->second + weight;

				// Edge weight becomes 0, so delete edge
				if (0 == sum) {
					out_edges.erase(it_out_edge);
					// Update edge weight
				}
				else {
					it_out_edge->second = sum;
				}
			}
			// Edge not found
			else {
				// Edge needs to be inserted as its weight != 0
				if (0 != weight) {
					out_edges[rank_neuron_id_pair] = weight;
				}
			}
		} // Source neuron id
	}

	void add_edge_weights(std::ifstream& file, const NeuronIdMap& neuron_id_map) {
		struct { double x, y, z; } src_pos, tgt_pos;
		NeuronIdMap::RankNeuronId src_id, tgt_id;
		std::string line;
		bool ret, success;

		while (std::getline(file, line)) {
			// Skip line with comments
			if (!line.empty() && '#' == line[0]) {
				continue;
			}

			std::stringstream sstream(line);
			success = (sstream >> src_pos.x) &&
				(sstream >> src_pos.y) &&
				(sstream >> src_pos.z) &&
				(sstream >> tgt_pos.x) &&
				(sstream >> tgt_pos.y) &&
				(sstream >> tgt_pos.z);

			assert(success);

			ret = neuron_id_map.pos2rank_neuron_id(src_pos.x, src_pos.y, src_pos.z, src_id);
			assert(ret);
			ret = neuron_id_map.pos2rank_neuron_id(tgt_pos.x, tgt_pos.y, tgt_pos.z, tgt_id);
			assert(ret);

			add_edge_weight(tgt_id.neuron_id, tgt_id.rank,
				src_id.neuron_id, src_id.rank, 1);

			if (!success) {
				std::cerr << "Skipping line: \"" << line << "\"\n";
				continue;
			}
		}
	}

	// Print network using global neuron ids
	void print(std::ostream& os, const NeuronIdMap& neuron_id_map) const {
		bool ret;
		size_t glob_tgt, glob_src;
		NeuronIdMap::RankNeuronId rank_neuron_id;

		// For my neurons
		for (size_t target_neuron_id = my_neuron_id_start; target_neuron_id <= my_neuron_id_end; target_neuron_id++) {
			// Walk through in-edges of my neuron
			const NetworkGraph::Edges& in_edges = get_in_edges(target_neuron_id);
			NetworkGraph::Edges::const_iterator it_in_edge;

			rank_neuron_id.rank = MPIInfos::my_rank;
			rank_neuron_id.neuron_id = target_neuron_id;
			ret = neuron_id_map.rank_neuron_id2glob_id(rank_neuron_id, glob_tgt);
			assert(ret);
			for (it_in_edge = in_edges.begin(); it_in_edge != in_edges.end(); ++it_in_edge) {
				rank_neuron_id.rank = it_in_edge->first.first;        // src rank
				rank_neuron_id.neuron_id = it_in_edge->first.second;  // src neuron id

				ret = neuron_id_map.rank_neuron_id2glob_id(rank_neuron_id, glob_src);
				assert(ret);

				// <target neuron id>  <source neuron id>  <weight>
				os << glob_tgt << " "
					<< glob_src << " "
					<< it_in_edge->second << "\n";
			}
		}
	}

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
