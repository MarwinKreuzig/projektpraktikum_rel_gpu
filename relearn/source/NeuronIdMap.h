#ifndef NEURONIDMAP_H
#define NEURONIDMAP_H

#include <vector>
#include <map>

#include <mpi.h>

class NeuronIdMap {
public:
	// Rank and local neuron id
	struct RankNeuronId {
		int rank;
		size_t neuron_id;
	};

	NeuronIdMap(size_t my_num_neurons, const double* x, const double* y, const double* z, MPI_Comm mpi_comm);
	bool rank_neuron_id2glob_id(const RankNeuronId& rank_neuron_id, size_t& glob_id) const;
	bool pos2rank_neuron_id(double x, double y, double z, RankNeuronId& result) const;

private:
	// Helper class to store neuron positions
	struct Position {
		double x, y, z;

		struct less {
			bool operator() (const Position& lhs, const Position& rhs) const {
				return  lhs.x < rhs.x ||
					(lhs.x == rhs.x && lhs.y < rhs.y) ||
					(lhs.x == rhs.x && lhs.y == rhs.y && lhs.z < rhs.z);
			}
		};
	};

	void create_rank_to_start_neuron_id_mapping(
		const std::vector<size_t>& rank_to_num_neurons,
		std::vector<size_t>& rank_to_start_neuron_id);

	void create_pos_to_rank_neuron_id_mapping(
		const std::vector<size_t>& rank_to_num_neurons,
		const std::vector<size_t>& rank_to_start_neuron_id,
		size_t my_num_neurons,
		const double* x, const double* y, const double* z,
		MPI_Comm mpi_comm,
		std::map<Position, RankNeuronId, Position::less>& pos_to_rank_neuron_id);

	std::vector<size_t> rank_to_start_neuron_id;  // Global neuron id of every rank's first local neuron
	std::map<Position, RankNeuronId, Position::less> pos_to_rank_neuron_id;
};

#endif /* NEURONIDMAP_H */
