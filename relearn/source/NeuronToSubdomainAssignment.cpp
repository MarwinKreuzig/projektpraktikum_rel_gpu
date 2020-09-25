#include <string>
#include <sstream>
#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>

#include "NeuronToSubdomainAssignment.h"
#include "SynapticElements.h"
#include "LogMessages.h"

// Return number of neurons which have positions in the range [min, max) in every dimension
size_t NeuronsInSubdomain::num_neurons(size_t subdomain_idx, size_t num_subdomains,
	const Position& min, const Position& max) {
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

bool NeuronsInSubdomain::position_in_box(const Position& pos, const Position& box_min, const Position& box_max) const noexcept {
	return ((pos.x >= box_min.x && pos.x < box_max.x) &&
		(pos.y >= box_min.y && pos.y < box_max.y) &&
		(pos.z >= box_min.z && pos.z < box_max.z));
}


void NeuronsInSubdomain::get_subdomain_boundaries(const Vec3<size_t>& subdomain_3idx, size_t num_subdomains_per_axis, Position& min, Position& max) const noexcept {
	get_subdomain_boundaries(subdomain_3idx, Vec3<size_t>{num_subdomains_per_axis}, min, max);
}

void NeuronsInSubdomain::get_subdomain_boundaries(const Vec3<size_t>& subdomain_3idx, const Vec3<size_t>& num_subdomains_per_axis, Position& min, Position& max) const noexcept {
	const auto length = simulation_box_length();
	const auto x_subdomain_length = length / num_subdomains_per_axis.x;
	const auto y_subdomain_length = length / num_subdomains_per_axis.y;
	const auto z_subdomain_length = length / num_subdomains_per_axis.z;

	min = Vec3d{ subdomain_3idx.x * x_subdomain_length, subdomain_3idx.y * y_subdomain_length, subdomain_3idx.z * z_subdomain_length };
	max = Vec3d{ (subdomain_3idx.x + 1) * x_subdomain_length, (subdomain_3idx.y + 1) * y_subdomain_length, (subdomain_3idx.z + 1) * z_subdomain_length };
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

	currently_frac_neurons_exc_ = 1.0;
	desired_frac_neurons_exc_ = 1.0;

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
double SubdomainFromFile::simulation_box_length() const noexcept {
	return std::nextafter(max_dimension_, max_dimension_ + 1);
}

void SubdomainFromFile::read_nodes_from_file(std::ifstream& file, Nodes* ptr_nodes) {
	Nodes& nodes = *ptr_nodes;
	std::string line;
	Node node;
	bool success = false;
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
	double max_of_all = 0.0;

	for (const Node& node : nodes) {
		auto max_of_node = std::max(node.pos.x, std::max(node.pos.y, node.pos.z));
		max_of_all = std::max(max_of_node, max_of_all);
	}

	return max_of_all;
}

void SubdomainFromFile::lazily_fill(size_t subdomain_idx, size_t num_subdomains, const Position& min, const Position& max) {
}



SubdomainFromNeuronDensity::SubdomainFromNeuronDensity(size_t num_neurons, double desired_frac_neurons_exc, double um_per_neuron)
	: um_per_neuron_(um_per_neuron),
	random_number_generator(RandomHolder<SubdomainFromNeuronDensity>::get_random_generator()),
	random_number_distribution(0.0, 1.0) {

	random_number_generator.seed(MPIInfos::my_rank);

	// Calculate size of simulation box based on neuron density
	// num_neurons^(1/3) == #neurons per dimension
	const double approx_number_of_neurons_per_dimension = ceil(pow(static_cast<double>(num_neurons), 1. / 3));
	simulation_box_length_ = approx_number_of_neurons_per_dimension * um_per_neuron;

	this->desired_frac_neurons_exc_ = desired_frac_neurons_exc;
	this->desired_num_neurons_ = num_neurons;

	this->currently_frac_neurons_exc_ = 0.0;
	this->currently_num_neurons_ = 0;

	//place_neurons_in_area(Position{ 0.0 }, Position{ simulation_box_length_ }, num_neurons);
}

void SubdomainFromNeuronDensity::place_neurons_in_area(
	const NeuronsInSubdomain::Position& offset,
	const NeuronsInSubdomain::Position& length_of_box,
	size_t num_neurons) {

	assert(length_of_box.x <= simulation_box_length_ && length_of_box.y <= simulation_box_length_ && length_of_box.z <= simulation_box_length_ &&
		"Requesting to fill neurons where no simulationbox is");

	const auto box = length_of_box - offset;

	const auto neurons_on_x = static_cast<size_t>(round(box.x / um_per_neuron_));
	const auto neurons_on_y = static_cast<size_t>(round(box.y / um_per_neuron_));
	const auto neurons_on_z = static_cast<size_t>(round(box.z / um_per_neuron_));

	const auto calculated_num_neurons = neurons_on_x * neurons_on_y * neurons_on_z;
	assert(calculated_num_neurons >= num_neurons && "Should emplace more neurons than space in box");
	assert(neurons_on_x < 65536 && neurons_on_y < 65536 && neurons_on_z < 65536 && "Should emplace more neurons in a dimension than possible");

	const size_t expected_number_in = num_neurons - static_cast<size_t>(ceil(num_neurons * desired_frac_neurons_exc_));
	const size_t expected_number_ex = num_neurons - expected_number_in;

	size_t placed_neurons = 0;
	size_t placed_in_neurons = 0;
	size_t placed_ex_neurons = 0;

	size_t random_counter = 0;
	std::vector<size_t> positions(calculated_num_neurons);
	for (size_t x_it = 0; x_it < neurons_on_x; x_it++) {
		for (size_t y_it = 0; y_it < neurons_on_y; y_it++) {
			for (size_t z_it = 0; z_it < neurons_on_z; z_it++) {
				size_t random_position = 0;
				random_position |= (z_it);
				random_position |= (y_it << 16);
				random_position |= (x_it << 32);
				positions[random_counter] = random_position;
				random_counter++;
			}
		}
	}

	std::shuffle(positions.begin(), positions.end(), random_number_generator);

	for (auto i = 0; i < num_neurons; i++) {
		const size_t pos_bitmask = positions[i];
		const size_t x_it = (pos_bitmask >> 32) & 0xFFFF;
		const size_t y_it = (pos_bitmask >> 16) & 0xFFFF;
		const size_t z_it = pos_bitmask & 0xFFFF;

		const double x_pos_rnd = random_number_distribution(random_number_generator) + x_it;
		const double y_pos_rnd = random_number_distribution(random_number_generator) + y_it;
		const double z_pos_rnd = random_number_distribution(random_number_generator) + z_it;

		Position pos_rnd{ x_pos_rnd, y_pos_rnd, z_pos_rnd };
		pos_rnd *= um_per_neuron_;

		const Position pos = pos_rnd + offset;

		const double type_indicator = random_number_distribution(random_number_generator);

		if (placed_ex_neurons < expected_number_ex && (type_indicator < desired_frac_neurons_exc_ || placed_in_neurons == expected_number_in)) {
			Node node{ pos, SynapticElements::EXCITATORY };
			placed_ex_neurons++;
			nodes_.emplace(node);
		}
		else {
			Node node{ pos, SynapticElements::INHIBITORY };
			placed_in_neurons++;
			nodes_.emplace(node);
		}

		placed_neurons++;

		if (placed_neurons == num_neurons) {
			const auto former_ex_neurons = this->currently_num_neurons_ * this->currently_frac_neurons_exc_;
			const auto former_in_neurons = this->currently_num_neurons_ - former_ex_neurons;

			this->currently_num_neurons_ += placed_neurons;

			const auto now_ex_neurons = former_ex_neurons + placed_ex_neurons;
			const auto now_in_neurons = former_in_neurons + placed_in_neurons;
			
			currently_frac_neurons_exc_ = static_cast<double>(now_ex_neurons) / static_cast<double>(this->currently_num_neurons_);
			return;
		}
	}

	assert(false && "In SubdomainFromNeuronDensity, shouldn't be here");
}

void SubdomainFromNeuronDensity::lazily_fill(size_t subdomain_idx, size_t num_subdomains, const Position& min, const Position& max) {
	if (filled_subdomains_.find(subdomain_idx) != filled_subdomains_.end()) {
		return;
	}

	filled_subdomains_.insert(subdomain_idx);

	const auto diff = max - min;
	const auto volume = diff.get_volume();

	const auto total_volume = simulation_box_length_ * simulation_box_length_ * simulation_box_length_;

	const auto neuron_portion = total_volume / volume;
	const auto neurons_in_subdomain = static_cast<size_t>(round(desired_num_neurons_ / neuron_portion));

	place_neurons_in_area(min, max, neurons_in_subdomain);
}

double SubdomainFromNeuronDensity::simulation_box_length() const noexcept {
	return simulation_box_length_;
}

void SubdomainFromNeuronDensity::get_subdomain_boundaries(
	const Vec3<size_t>& subdomain_3idx,
	size_t num_subdomains_per_axis,
	Position& min,
	Position& max) const noexcept {
	const auto length = simulation_box_length();
	const auto one_subdomain_length = length / num_subdomains_per_axis;

	min = ((Vec3d)subdomain_3idx) * one_subdomain_length;
	max = ((Vec3d)(subdomain_3idx + 1)) * one_subdomain_length;

	min.round_to_larger_multiple(um_per_neuron_);
	max.round_to_larger_multiple(um_per_neuron_);
}

void SubdomainFromNeuronDensity::get_subdomain_boundaries(
	const Vec3<size_t>& subdomain_3idx,
	const Vec3<size_t>& num_subdomains_per_axis,
	Position& min,
	Position& max) const noexcept {

	const auto length = simulation_box_length();
	const auto x_subdomain_length = length / num_subdomains_per_axis.x;
	const auto y_subdomain_length = length / num_subdomains_per_axis.y;
	const auto z_subdomain_length = length / num_subdomains_per_axis.z;

	min = Vec3d{ subdomain_3idx.x * x_subdomain_length, subdomain_3idx.y * y_subdomain_length, subdomain_3idx.z * z_subdomain_length };
	max = Vec3d{ (subdomain_3idx.x + 1) * x_subdomain_length, (subdomain_3idx.y + 1) * y_subdomain_length, (subdomain_3idx.z + 1) * z_subdomain_length };

	min.round_to_larger_multiple(um_per_neuron_);
	max.round_to_larger_multiple(um_per_neuron_);
}
