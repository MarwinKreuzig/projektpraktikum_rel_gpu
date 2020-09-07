#include <string>
#include <sstream>
#include <iostream>
#include <vector>
#include <fstream>

#include "NeuronToSubdomainAssignment.h"
#include "SynapticElements.h"
#include "LogMessages.h"

// Return number of neurons which have positions in the range [min, max) in every dimension
size_t NeuronsInSubdomain::num_neurons(size_t subdomain_idx, size_t num_subdomains,
	Position min, Position max) {
	size_t cnt = 0;

	for (const Node& node : nodes_) {
		if (position_in_box(node.pos, min, max)) {
			cnt++;
		}
	}
	return cnt;
}

// Return neuron positions in the range [min, max) in every dimension
void NeuronsInSubdomain::neuron_positions(size_t subdomain_idx, size_t num_subdomains,
	const Position& min, const Position& max, std::vector<Position>& pos) const {
	for (const Node& node : nodes_) {
		if (position_in_box(node.pos, min, max)) {
			pos.push_back(node.pos);
		}
	}
}

// For all neurons with positions in the range [min, max) in every dimension
void NeuronsInSubdomain::neuron_types(size_t subdomain_idx, size_t num_subdomains,
	const Position& min, const Position& max,
	std::vector<SynapticElements::SignalType>& types) const {
	for (const Node& node : nodes_) {
		if (position_in_box(node.pos, min, max)) {
			types.push_back(node.signal_type);
		}
	}
}

inline bool NeuronsInSubdomain::position_in_box(const Position& pos, const Position& box_min, const Position& box_max) const noexcept {
	return ((pos.x >= box_min.x && pos.x < box_max.x) &&
		(pos.y >= box_min.y && pos.y < box_max.y) &&
		(pos.z >= box_min.z && pos.z < box_max.z));
}

double NeuronsInSubdomain::ratio_neurons_exc() const noexcept {
	return frac_neurons_exc_;
}

// Return area names of neurons which have positions in the range [min, max) in every dimension
void NeuronsInSubdomain::neuron_area_names(size_t subdomain_idx, size_t num_subdomains,
	const Position& min, const Position& max, std::vector<std::string>& areas) const {
	for (const Node& node : nodes_) {
		if (position_in_box(node.pos, min, max)) {
			areas.push_back("no_area");
		}
	}
}



SubdomainFromFile::SubdomainFromFile(size_t num_neurons, std::ifstream& file) {
	read_nodes_from_file(file, &nodes_);
	max_dimension_ = largest_dimension(nodes_);

	frac_neurons_exc_ = 1.0;

	// Num neurons expected != num neurons read from file
	if (num_neurons != nodes_.size()) {
		std::stringstream sstring;
		sstring << __FUNCTION__ << ": expected number of neurons != number of neurons read from file";
		LogMessages::print_error(sstring.str().c_str());
		exit(EXIT_FAILURE);
	}
}

// Make sure that the length is larger than the largest coordinate of any neuron.
// That is, for all neuron coordinates (x,y,z), x,y,z in [0 , length).
double SubdomainFromFile::simulation_box_length() const {
	return std::nextafter(max_dimension_, max_dimension_ + 1);
}

void SubdomainFromFile::read_nodes_from_file(std::ifstream& file, Nodes* ptr_nodes) {
	Nodes& nodes = *ptr_nodes;
	std::string line;
	Node node;
	bool success;
	while (std::getline(file, line)) {
		// Skip line with comments
		if (!line.empty() && '#' == line[0]) {
			continue;
		}

		std::stringstream sstream(line);
		success = (sstream >> node.pos.x) &&
			(sstream >> node.pos.y) &&
			(sstream >> node.pos.z) &&
			(sstream >> node.area_name);

		node.signal_type = SynapticElements::EXCITATORY;

		nodes.insert(node);

		if (!success) {
			std::cerr << "Skipping line: \"" << line << "\"\n";
			continue;
		}
	}
}

double SubdomainFromFile::largest_dimension(const Nodes& nodes) {
	double max_of_node, max_of_all = 0;

	for (const Node& node : nodes) {
		max_of_node = std::max(node.pos.x, std::max(node.pos.y, node.pos.z));
		max_of_all = std::max(max_of_node, max_of_all);
	}

	return max_of_all;
}



SubdomainFromNeuronDensity::SubdomainFromNeuronDensity(size_t num_neurons, double desired_frac_neurons_exc, double um_per_neuron)
	: um_per_neuron_(um_per_neuron),
	random_number_generator(RandomHolder<SubdomainFromNeuronDensity>::get_random_generator()),
	random_number_distribution(0.0, 1.0) {

	// Calculate size of simulation box based on neuron density
	// num_neurons^(1/3) == #neurons per dimension
	double approx_number_of_neurons_per_dimension = ceil(pow((double)num_neurons, 1. / 3));
	simulation_box_length_ = approx_number_of_neurons_per_dimension * um_per_neuron;

	size_t expected_number_in = num_neurons - ceil(num_neurons * desired_frac_neurons_exc);
	size_t placed_neurons = 0;
	size_t placed_in_neurons = 0;

	for (auto x_it = 0; x_it < approx_number_of_neurons_per_dimension; x_it++) {
		for (auto y_it = 0; y_it < approx_number_of_neurons_per_dimension; y_it++) {
			for (auto z_it = 0; z_it < approx_number_of_neurons_per_dimension; z_it++) {
				double x_pos = random_number_distribution(random_number_generator) * um_per_neuron;
				double y_pos = random_number_distribution(random_number_generator) * um_per_neuron;
				double z_pos = random_number_distribution(random_number_generator) * um_per_neuron;

				Position pos{ x_pos, y_pos, z_pos };

				double type_indicator = random_number_distribution(random_number_generator);

				if (type_indicator < desired_frac_neurons_exc || expected_number_in == placed_in_neurons) {
					Node node{ pos, SynapticElements::EXCITATORY };
					nodes_.emplace(node);
				}
				else {
					Node node{ pos, SynapticElements::INHIBITORY };
					placed_in_neurons++;
					nodes_.emplace(node);
				}

				placed_neurons++;

				if (placed_neurons == num_neurons) {
					frac_neurons_exc_ = 1.0 - ((double)placed_in_neurons / (double)num_neurons);
					return;
				}
			}
		}
	}

	assert(false && "In SubdomainFromNeuronDensity, shouldn't be here");
}

inline double SubdomainFromNeuronDensity::simulation_box_length() const noexcept {
	return simulation_box_length_;
}
