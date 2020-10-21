#pragma once

#include <random>

#include "NeuronToSubdomainAssignment.h"
#include "Vec3.h"

// This class fills every subdomain with neurons at
// random positions. The size of the simulation box and the number of neurons per
// subdomain depend on the requested neuron density, i.e., micrometer per neuron
// in each of the three dimensions.
class SubdomainFromNeuronDensity : public NeuronToSubdomainAssignment {
public:
	SubdomainFromNeuronDensity(size_t num_neurons, double frac_neurons_exc,
		double um_per_neuron = 26);

	SubdomainFromNeuronDensity(const SubdomainFromNeuronDensity& other) = delete;
	SubdomainFromNeuronDensity(SubdomainFromNeuronDensity&& other) = delete;

	SubdomainFromNeuronDensity& operator=(const SubdomainFromNeuronDensity& other) = delete;
	SubdomainFromNeuronDensity& operator=(SubdomainFromNeuronDensity&& other) = delete;

	~SubdomainFromNeuronDensity() {}

	void get_subdomain_boundaries(const Vec3<size_t>& subdomain_3idx, size_t num_subdomains_per_axis,
		Position& min, Position& max) const noexcept override;

	void get_subdomain_boundaries(const Vec3<size_t>& subdomain_3idx, const Vec3<size_t>& num_subdomains_per_axis,
		Position& min, Position& max) const noexcept override;

	void fill_subdomain(size_t subdomain_idx, size_t num_subdomains, const Position& min, const Position& max) override;

private:
	const double um_per_neuron_;  // Micrometer per neuron in one dimension

	void SubdomainFromNeuronDensity::place_neurons_in_area(
		const NeuronToSubdomainAssignment::Position& offset,
		const NeuronToSubdomainAssignment::Position& length_of_box,
		size_t num_neurons);

	std::mt19937& random_number_generator;
	std::uniform_real_distribution<double> random_number_distribution;
};
