/*
 * File: Neurons.h
 * Author: naveau
 *
 * Created on September 26, 2014, 10:23 AM
 */

#ifndef NEURONS_H
#define	NEURONS_H

#include <random>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <numeric>
#include <mpi.h>


#include "Positions.h"
#include "Parameters.h"
#include "Octree.h"
#include "SynapticElements.h"
#include "LogFiles.h"
#include "Parameters.h"
#include "Timers.h"
#include "MPIInfos.h"
#include "NetworkGraph.h"
#include "NeuronIdMap.h"
#include "Random.h"

 /***************************************************************************************************
 * NOTE: The following two type declarations (SynapseCreationRequests, MapSynapseCreationRequests) *
 * are outside of the class Neurons so that the class Octree can use them *
 ***************************************************************************************************/

class Partition;
class NeuronMonitor;

// Types
typedef SynapticElements Axons;
typedef SynapticElements DendritesExc;
typedef SynapticElements DendritesInh;

/**
* Type for synapse creation requests which are used with MPI
*/
class SynapseCreationRequests {
public:
	SynapseCreationRequests() noexcept : num_requests(0) {}

	size_t size() const noexcept { return num_requests; }

	void resize(size_t size) {
		num_requests = size;
		requests.resize(3 * size);
		responses.resize(size);
	}

	void append(size_t source_neuron_id, size_t target_neuron_id, size_t dendrite_type_needed) {
		num_requests++;

		requests.push_back(source_neuron_id);
		requests.push_back(target_neuron_id);
		requests.push_back(dendrite_type_needed);

		responses.resize(responses.size() + 1);
	}

	void get_request(size_t request_index, size_t& source_neuron_id, size_t& target_neuron_id, size_t& dendrite_type_needed) const noexcept {
		const size_t base_index = 3 * request_index;

		source_neuron_id = requests[base_index];
		target_neuron_id = requests[base_index + 1];
		dendrite_type_needed = requests[base_index + 2];
	}

	void set_response(size_t request_index, char connected) noexcept {
		responses[request_index] = connected;
	}

	void get_response(size_t request_index, char& connected) const noexcept {
		connected = responses[request_index];
	}

	size_t* get_requests() noexcept {
		return requests.data();
	}

	const size_t* get_requests() const noexcept {
		return requests.data();
	}

	char* get_responses() noexcept {
		return responses.data();
	}

	const char* get_responses() const noexcept {
		return responses.data();
	}

	size_t get_requests_size_in_bytes() const noexcept {
		return requests.size() * sizeof(size_t);
	}

	size_t get_responses_size_in_bytes() const noexcept {
		return responses.size() * sizeof(char);
	}

private:
	size_t num_requests; // Number of synapse creation requests
	std::vector<size_t> requests; // Each request to form a synapse is a 3-tuple: (source_neuron_id, target_neuron_id, dendrite_type_needed)
								 // That is why requests.size() == 3*responses.size()
								 // Note, a more memory-efficient implementation would use a smaller data type (not size_t) for dendrite_type_needed.
								 // This vector is used as MPI communication buffer
	std::vector<char> responses; // Response if the corresponding request was accepted and thus the synapse was formed
								 // responses[i] refers to requests[3*i,...,3*i+2]
								 // This vector is used as MPI communication buffer
};

/**
 * Map of (MPI rank; SynapseCreationRequests)
 * The MPI rank specifies the corresponding process
 */
typedef std::map<int, SynapseCreationRequests> MapSynapseCreationRequests;


class Neurons {
	friend class NeuronMonitor;

	/**
	 * Identifies a neuron by the MPI rank of its owner
	 * and its neuron id on the owner, i.e., the pair <rank, neuron_id>
	 */
	struct RankNeuronId {
		const int rank;			// MPI rank of the owner
		const size_t neuron_id;	// Neuron id on the owner

		RankNeuronId(int rank, size_t neuron_id) noexcept :
			rank(rank), neuron_id(neuron_id) {
		}

		bool operator==(const RankNeuronId& other) const noexcept {
			return (this->rank == other.rank && this->neuron_id == other.neuron_id);
		}

		friend std::ostream& operator<< (std::ostream& os, const RankNeuronId& rni) {
			os << "Rank: " << rni.rank << "\t id: " << rni.neuron_id << "\n";
			return os;
		}
	};

	/**
	 * Type for list element used to represent a synapse for synapse selection
	 */
	struct Synapse {
		RankNeuronId rank_neuron_id;
		unsigned int synapse_id; // Id of the synapse. Used to distinguish multiple synapses between the same neuron pair

		Synapse(RankNeuronId rank_neuron_id, unsigned int synapse_id) noexcept :
			rank_neuron_id(rank_neuron_id), synapse_id(synapse_id) {
		}
	};

	/**
	 * Type for synapse deletion requests which are used with MPI
	 */
	struct SynapseDeletionRequests {
		SynapseDeletionRequests() noexcept : num_requests(0) {}

		size_t size() const noexcept { return num_requests; }

		void resize(size_t size) {
			num_requests = size;
			requests.resize(6 * size);
		}

		void append(size_t src_neuron_id, size_t tgt_neuron_id, size_t affected_neuron_id, size_t affected_element_type, size_t signal_type, size_t synapse_id) {
			num_requests++;

			requests.push_back(src_neuron_id);
			requests.push_back(tgt_neuron_id);
			requests.push_back(affected_neuron_id);
			requests.push_back(affected_element_type);
			requests.push_back(signal_type);
			requests.push_back(synapse_id);
		}

		void get_request(size_t request_index, size_t& src_neuron_id, size_t& tgt_neuron_id, size_t& affected_neuron_id, size_t& affected_element_type, size_t& signal_type, size_t& synapse_id) const noexcept {
			const size_t base_index = 6 * request_index;

			src_neuron_id = requests[base_index];
			tgt_neuron_id = requests[base_index + 1];
			affected_neuron_id = requests[base_index + 2];
			affected_element_type = requests[base_index + 3];
			signal_type = requests[base_index + 4];
			synapse_id = requests[base_index + 5];
		}

		// Get pointer to data
		size_t* get_requests() noexcept {
			return requests.data();
		}

		const size_t* get_requests() const noexcept {
			return requests.data();
		}

		size_t get_requests_size_in_bytes() const noexcept {
			return requests.size() * sizeof(size_t);
		}

	private:
		size_t num_requests;			// Number of synapse deletion requests
		std::vector<size_t> requests;	// Each request to delete a synapse is a 6-tuple:
										// (src_neuron_id, tgt_neuron_id, affected_neuron_id, affected_element_type, signal_type, synapse_id)
										// That is why requests.size() == 6*num_requests
										// Note, a more memory-efficient implementation would use a smaller data type (not size_t)
										// for affected_element_type, signal_type.
										// This vector is used as MPI communication buffer
	};

	/**
	* Type for list element used to store pending synapse deletion
	*/
	struct PendingSynapseDeletion {
		RankNeuronId src_neuron_id; // Synapse source neuron id
		RankNeuronId tgt_neuron_id; // Synapse target neuron id
		RankNeuronId affected_neuron_id; // Neuron whose synaptic element should be set vacant
		SynapticElements::ElementType affected_element_type; // Type of the element (axon/dendrite) to be set vacant
		SynapticElements::SignalType signal_type; // Signal type (exc/inh) of the synapse
		unsigned int synapse_id; // Synapse id of the synapse to be deleted
		bool affected_element_already_deleted; // "True" if the element to be set vacant was already deleted by the neuron owning it
															 // "False" if the element must be set vacant
	};

	template<typename T>
	struct StatisticalMeasures {
		T min;
		T max;
		double avg;
		double var;
		double std;
	};

public:
	/**
	 * Map of (MPI rank; SynapseDeletionRequests)
	 * The MPI rank specifies the corresponding process
	 */
	typedef std::map<int, SynapseDeletionRequests> MapSynapseDeletionRequests;

	Neurons(size_t num_neurons, const Parameters& params, const Partition&);
	Neurons(size_t num_neurons, const Parameters& params, const Partition&, std::unique_ptr<NeuronModels> model);
	~Neurons() = default;

	Neurons(const Neurons& other) = delete;
	Neurons(Neurons&& other) = default;

	Neurons& operator=(const Neurons& other) = delete;
	Neurons& operator=(Neurons&& other) = default;

	void set_model(std::unique_ptr<NeuronModels> model) noexcept {
		neuron_models = std::move(model);
	}

	size_t get_num_neurons() const noexcept { return num_neurons; }
	Positions& get_positions() noexcept { return positions; }
	std::vector<std::string>& get_area_names() noexcept { return area_names; }
	Axons& get_axons() noexcept { return axons; }
	const DendritesExc& get_dendrites_exc() const noexcept { return dendrites_exc; }
	const DendritesInh& get_dendrites_inh() const noexcept { return dendrites_inh; }
	NeuronModels& get_neuron_models() noexcept { return *neuron_models; }

	bool get_vacant_axon(size_t& neuron_id, Vec3d& xyz_pos, Cell::DendriteType& dendrite_type_needed) noexcept;

	void init_synaptic_elements();

	void update_electrical_activity(const NetworkGraph& network_graph) {
		neuron_models->update_electrical_activity(network_graph, calcium);
	}

	void update_number_synaptic_elements_delta() noexcept {
		axons.update_number_elements_delta(calcium);
		dendrites_exc.update_number_elements_delta(calcium);
		dendrites_inh.update_number_elements_delta(calcium);
	}

	void update_connectivity(Octree& global_tree,
		NetworkGraph& network_graph,
		size_t& num_synapses_deleted,
		size_t& num_synapses_created) {

		delete_synapses(num_synapses_deleted, network_graph);
		create_synapses(num_synapses_created, global_tree, network_graph);
	}

	void print_sums_of_synapses_and_elements_to_log_file_on_rank_0(size_t step, LogFiles& log_file, const Parameters& params, size_t sum_synapses_deleted, size_t sum_synapses_created);

	// Print global information about all neurons at rank 0
	void print_neurons_overview_to_log_file_on_rank_0(size_t step, LogFiles& log_file, const Parameters& params);

	void print_network_graph_to_log_file(LogFiles& log_file,
		const NetworkGraph& network_graph,
		const Parameters& params,
		const NeuronIdMap& neuron_id_map);

	void print_positions_to_log_file(LogFiles& log_file, const Parameters& params,
		const NeuronIdMap& neuron_id_map);

	void print();

	void print_info_for_barnes_hut();

private:
	void delete_synapses(size_t& num_synapses_deleted, NetworkGraph& network_graph);

	void create_synapses(size_t& num_synapses_created, Octree& global_tree, NetworkGraph& network_graph);

	void debug_check_counts();

	template<typename T>
	StatisticalMeasures<T> global_statistics(const T* local_values, size_t num_local_values, size_t total_num_values, int root, MPI_Comm mpi_comm) {
		const auto result = std::minmax_element(&local_values[0], &local_values[num_neurons]);
		const T my_min = *result.first;
		const T my_max = *result.second;

		double my_avg = std::accumulate(&local_values[0], &local_values[num_neurons], 0.0);
		my_avg /= total_num_values;

		// Get global min and max at rank "root"
		const double d_my_min = static_cast<double>(my_min);
		const double d_my_max = static_cast<double>(my_max);
		double d_min, d_max;
		MPI_Reduce(&d_my_min, &d_min, 1, MPI_DOUBLE, MPI_MIN, root, mpi_comm);
		MPI_Reduce(&d_my_max, &d_max, 1, MPI_DOUBLE, MPI_MAX, root, mpi_comm);

		// Get global avg at all ranks (needed for variance)
		double avg;
		MPI_Allreduce(&my_avg, &avg, 1, MPI_DOUBLE, MPI_SUM, mpi_comm);

		/**
		* Calc variance
		*/
		double my_var = 0;
		for (size_t neuron_id = 0; neuron_id < num_neurons; ++neuron_id) {
			my_var += (local_values[neuron_id] - avg) * (local_values[neuron_id] - avg);
		}
		my_var /= total_num_values;

		// Get global variance at rank "root"
		double var;
		MPI_Reduce(&my_var, &var, 1, MPI_DOUBLE, MPI_SUM, root, mpi_comm);

		// Calc standard deviation
		const double std = sqrt(var);

		return { static_cast<T>(d_min), static_cast<T>(d_max), avg, var, std };
	}


	/**
	 * Returns iterator to randomly chosen synapse from list
	 */
	void select_synapse(std::list<Synapse>& list, typename std::list<Synapse>::iterator& it);

	void add_synapse_to_pending_deletions(const RankNeuronId& src_neuron_id,
		const RankNeuronId& tgt_neuron_id,
		const RankNeuronId& affected_neuron_id,
		SynapticElements::ElementType affected_element_type,
		SynapticElements::SignalType signal_type,
		unsigned int synapse_id,
		std::list<PendingSynapseDeletion>& list);

	/**
	 * Determines which synapses should be deleted.
	 * The selected synapses connect with neuron "neuron_id" and the type of
	 * those synapses is given by "signal_type".
	 *
	 * NOTE: The semantics of the function is not nice but used to postpone all updates
	 * due to synapse deletion until all neurons have decided *independently* which synapse
	 * to delete. This should reflect how it's done for a distributed memory implementation.
	 */
	void find_synapses_for_deletion(size_t neuron_id,
		SynapticElements::ElementType element_type,
		SynapticElements::SignalType signal_type,
		unsigned int num_synapses_to_delete,
		const NetworkGraph& network_graph,
		std::list<PendingSynapseDeletion>& list_pending_deletions);

	void print_pending_synapse_deletions(std::list<PendingSynapseDeletion>& list);

	void delete_synapses(std::list<PendingSynapseDeletion>& list,
		SynapticElements& axons,
		SynapticElements& dendrites_exc,
		SynapticElements& dendrites_inh,
		NetworkGraph& network_graph,
		size_t& num_synapses_deleted);


	size_t num_neurons; // Local number of neurons
	std::vector<size_t> local_ids;

	const Partition& partition;

	std::unique_ptr<NeuronModels> neuron_models;

	Axons axons;
	DendritesExc dendrites_exc;
	DendritesInh dendrites_inh;

	Positions positions; // Position of every neuron
	std::vector<double> calcium; // Intracellular calcium concentration of every neuron
	std::vector<std::string> area_names; // Area name of every neuron

	// Random number generator for this class (C++11)
	std::mt19937& random_number_generator;
	// Random number distribution used together with "random_number_generator" (C++11)
	// Uniform distribution for interval [0, 1) (see constructor for initialization)
	std::uniform_real_distribution<double> random_number_distribution;
};

#endif	/* NEURONS_H */
