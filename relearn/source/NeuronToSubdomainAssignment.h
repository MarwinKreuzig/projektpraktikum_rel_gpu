#ifndef NEURON_TO_SUBDOMAIN_ASSIGNMENT_H_
#define NEURON_TO_SUBDOMAIN_ASSIGNMENT_H_

#include <vector>
#include <set>
#include <cmath>
#include <fstream>
#include <random>

#include "SynapticElements.h"
#include "Random.h"
#include "Vec3.h"

// Interface that has to be implemented by any class that
// wants to provide a neuron-to-subdomain assignment
class NeuronsInSubdomain {
public:
	// Helper class to store neuron positions
	using Position = Vec3<double>;

protected:
	struct Node {
		Position pos;
		SynapticElements::SignalType signal_type;
		std::string area_name;

		struct less {
			bool operator() (const Node& lhs, const Node& rhs) const noexcept {
				Position::less less;
				return  less(lhs.pos, rhs.pos);
			}
		};
	};

	using Nodes = std::set<Node, Node::less>;
	Nodes nodes_;

	double desired_frac_neurons_exc_;
	size_t desired_num_neurons_;

	double currently_frac_neurons_exc_;
	size_t currently_num_neurons_;

	bool position_in_box(const Position& pos, const Position& box_min, const Position& box_max) const noexcept;

	NeuronsInSubdomain() noexcept {
	}

public:
	virtual ~NeuronsInSubdomain() {};

	NeuronsInSubdomain(const NeuronsInSubdomain& other) = delete;
	NeuronsInSubdomain(NeuronsInSubdomain&& other) = delete;

	NeuronsInSubdomain& operator=(const NeuronsInSubdomain& other) = delete;
	NeuronsInSubdomain& operator=(NeuronsInSubdomain&& other) = delete;

	// Make sure that the length is larger than the largest coordinate of any neuron.
	// That is, for all neuron coordinates (x,y,z), x,y,z in [0 , length).
	virtual double simulation_box_length() const noexcept = 0;

	// Total number of neurons
	size_t desired_num_neurons() const noexcept {
		return desired_num_neurons_;
	}

	// Total number of neurons already placed
	size_t placed_num_neurons() const noexcept {
		return currently_num_neurons_;
	}

	// Ratio of EXCITATORY neurons
	double desired_ratio_neurons_exc() const noexcept {
		return desired_frac_neurons_exc_;
	}

	// Ratio of EXCITATORY neurons already placed
	double placed_ratio_neurons_exc() const noexcept {
		return currently_frac_neurons_exc_;
	}



	virtual void get_subdomain_boundaries(const Vec3<size_t>& subdomain_3idx, size_t num_subdomains_per_axis,
		Position& min, Position& max) const noexcept;

	virtual void get_subdomain_boundaries(const Vec3<size_t>& subdomain_3idx, const Vec3<size_t>& num_subdomains_per_axis,
		Position& min, Position& max) const noexcept;

	virtual void lazily_fill(size_t subdomain_idx, size_t num_subdomains,
		const Position& min, const Position& max) = 0;

	// Return number of neurons which have positions in the range [min, max) in every dimension
	virtual size_t num_neurons(size_t subdomain_idx, size_t num_subdomains,
		const Position& min, const Position& max);

	// Return neurons which have positions in the range [min, max) in every dimension
	virtual void neuron_positions(size_t subdomain_idx, size_t num_subdomains,
		const Position& min, const Position& max, std::vector<Position>& pos) const;

	// Return neurons which have positions in the range [min, max) in every dimension
	virtual void neuron_types(size_t subdomain_idx, size_t num_subdomains,
		const Position& min, const Position& max,
		std::vector<SynapticElements::SignalType>& types) const;

	// Return neurons which have positions in the range [min, max) in every dimension
	virtual void neuron_area_names(size_t subdomain_idx, size_t num_subdomains,
		const Position& min, const Position& max, std::vector<std::string>& areas) const;
};

// This class fills every subdomain with neurons at
// random positions. The size of the simulation box and the number of neurons per
// subdomain depend on the requested neuron density, i.e., micrometer per neuron
// in each of the three dimensions.
class SubdomainFromNeuronDensity : public NeuronsInSubdomain {
public:
	SubdomainFromNeuronDensity(size_t num_neurons, double frac_neurons_exc,
		double um_per_neuron = 26);

	SubdomainFromNeuronDensity(const SubdomainFromNeuronDensity& other) = delete;
	SubdomainFromNeuronDensity(SubdomainFromNeuronDensity&& other) = delete;

	SubdomainFromNeuronDensity& operator=(const SubdomainFromNeuronDensity& other) = delete;
	SubdomainFromNeuronDensity& operator=(SubdomainFromNeuronDensity&& other) = delete;

	~SubdomainFromNeuronDensity() {}

	double simulation_box_length() const noexcept override;

	void get_subdomain_boundaries(const Vec3<size_t>& subdomain_3idx, size_t num_subdomains_per_axis,
		Position& min, Position& max) const noexcept override;

	void get_subdomain_boundaries(const Vec3<size_t>& subdomain_3idx, const Vec3<size_t>& num_subdomains_per_axis,
		Position& min, Position& max) const noexcept override;

	void lazily_fill(size_t subdomain_idx, size_t num_subdomains, const Position& min, const Position& max) override;

private:
	const double um_per_neuron_;  // Micrometer per neuron in one dimension
	double simulation_box_length_;

	std::set<size_t> filled_subdomains_;

	void SubdomainFromNeuronDensity::place_neurons_in_area(
		const NeuronsInSubdomain::Position& offset,
		const NeuronsInSubdomain::Position& length_of_box,
		size_t num_neurons);

	std::mt19937& random_number_generator;
	std::uniform_real_distribution<double> random_number_distribution;
};

// This class reads the neurons with their positions from a file
// and, based on this, determines the size of the simulation box
// and the number of neurons in every individual subdomain.
class SubdomainFromFile : public NeuronsInSubdomain {
public:
	SubdomainFromFile(size_t num_neurons, std::ifstream& file);

	SubdomainFromFile(const SubdomainFromFile& other) = delete;
	SubdomainFromFile(SubdomainFromFile&& other) = delete;

	SubdomainFromFile& operator=(const SubdomainFromFile& other) = delete;
	SubdomainFromFile& operator=(SubdomainFromFile&& other) = delete;

	~SubdomainFromFile() {}

	// Make sure that the length is larger than the largest coordinate of any neuron.
	// That is, for all neuron coordinates (x,y,z), x,y,z in [0 , length).
	double simulation_box_length() const noexcept override;

	void lazily_fill(size_t subdomain_idx, size_t num_subdomains, const Position& min, const Position& max) override;

private:
	void read_nodes_from_file(std::ifstream& file, Nodes* ptr_nodes);
	double largest_dimension(const Nodes& nodes);

	double max_dimension_;
};

#endif /* NEURON_TO_SUBDOMAIN_ASSIGNMENT_H_ */
