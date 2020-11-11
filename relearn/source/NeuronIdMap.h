#ifndef NEURONIDMAP_H
#define NEURONIDMAP_H

#include <vector>
#include <map>

#include <mpi.h>

#include "Vec3.h"

class NeuronIdMap {
public:
	// Rank and local neuron id
	struct RankNeuronId {
		int rank;
		size_t neuron_id;
	};

	NeuronIdMap(size_t my_num_neurons, const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& z, MPI_Comm mpi_comm);
	bool rank_neuron_id2glob_id(const RankNeuronId& rank_neuron_id, size_t& glob_id) const /*noexcept*/;
	bool pos2rank_neuron_id(const Vec3d& pos, RankNeuronId& result) const;

private:
	void create_rank_to_start_neuron_id_mapping(
		const std::vector<size_t>& rank_to_num_neurons,
		std::vector<size_t>& rank_to_start_neuron_id);

	void create_pos_to_rank_neuron_id_mapping(
		const std::vector<size_t>& rank_to_num_neurons,
		const std::vector<size_t>& rank_to_start_neuron_id,
		size_t my_num_neurons,
		const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& z,
		MPI_Comm mpi_comm,
		std::map<Vec3d, RankNeuronId>& pos_to_rank_neuron_id);

	std::vector<size_t> rank_to_start_neuron_id;  // Global neuron id of every rank's first local neuron
	std::map<Vec3d, RankNeuronId> pos_to_rank_neuron_id;
};

#endif /* NEURONIDMAP_H */
