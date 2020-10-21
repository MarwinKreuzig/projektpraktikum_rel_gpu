//#include <string>
//#include <sstream>
//#include <iostream>
//#include <vector>
//#include <algorithm>

#include "NeuronToSubdomainAssignment.h"
#include "SynapticElements.h"
#include "LogMessages.h"

// Return number of neurons which have positions in the range [min, max) in every dimension
size_t NeuronToSubdomainAssignment::num_neurons(size_t subdomain_idx, size_t num_subdomains,
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
void NeuronToSubdomainAssignment::neuron_positions(size_t subdomain_idx, size_t num_subdomains,
	const Position& min, const Position& max, std::vector<Position>& pos) const {
	for (const Node& node : nodes_) {
		if (position_in_box(node.pos, min, max)) {
			pos.push_back(node.pos);
		}
	}
}

// For all neurons with positions in the range [min, max) in every dimension
void NeuronToSubdomainAssignment::neuron_types(size_t subdomain_idx, size_t num_subdomains,
	const Position& min, const Position& max,
	std::vector<SynapticElements::SignalType>& types) const {
	for (const Node& node : nodes_) {
		if (position_in_box(node.pos, min, max)) {
			types.push_back(node.signal_type);
		}
	}
}

bool NeuronToSubdomainAssignment::position_in_box(const Position& pos, const Position& box_min, const Position& box_max) const noexcept {
	return ((pos.x >= box_min.x && pos.x < box_max.x) &&
		(pos.y >= box_min.y && pos.y < box_max.y) &&
		(pos.z >= box_min.z && pos.z < box_max.z));
}


void NeuronToSubdomainAssignment::get_subdomain_boundaries(const Vec3<size_t>& subdomain_3idx, size_t num_subdomains_per_axis, Position& min, Position& max) const noexcept {
	get_subdomain_boundaries(subdomain_3idx, Vec3<size_t>{num_subdomains_per_axis}, min, max);
}

void NeuronToSubdomainAssignment::get_subdomain_boundaries(const Vec3<size_t>& subdomain_3idx, const Vec3<size_t>& num_subdomains_per_axis, Position& min, Position& max) const noexcept {
	const auto lengths = get_simulation_box_length();
	const auto x_subdomain_length = lengths.x / num_subdomains_per_axis.x;
	const auto y_subdomain_length = lengths.y / num_subdomains_per_axis.y;
	const auto z_subdomain_length = lengths.z / num_subdomains_per_axis.z;

	min = Vec3d{ subdomain_3idx.x * x_subdomain_length, subdomain_3idx.y * y_subdomain_length, subdomain_3idx.z * z_subdomain_length };
	max = Vec3d{ (subdomain_3idx.x + 1) * x_subdomain_length, (subdomain_3idx.y + 1) * y_subdomain_length, (subdomain_3idx.z + 1) * z_subdomain_length };
}

// Return area names of neurons which have positions in the range [min, max) in every dimension
void NeuronToSubdomainAssignment::neuron_area_names(size_t subdomain_idx, size_t num_subdomains,
	const Position& min, const Position& max, std::vector<std::string>& areas) const {
	for (const Node& node : nodes_) {
		if (position_in_box(node.pos, min, max)) {
			areas.push_back("no_area");
		}
	}
}
