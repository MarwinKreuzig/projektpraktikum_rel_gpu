#include <fstream>
#include <iomanip>
#include <cassert>

#include "NeuronToSubdomainAssignment.h"
#include "SynapticElements.h"
#include "LogMessages.h"

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


size_t NeuronToSubdomainAssignment::num_neurons(size_t subdomain_idx, size_t num_subdomains,
	const Position& min, const Position& max) const {

	const bool contains = neurons_in_subdomain.find(subdomain_idx) != neurons_in_subdomain.end();
	if (!contains) {
		assert(false && "Wanted to have num_neurons of subdomain_idx that is not present");
		return 0;
	}

	const Nodes& nodes = neurons_in_subdomain.at(subdomain_idx);
	size_t cnt = nodes.size();
	return cnt;
}

void NeuronToSubdomainAssignment::neuron_positions(size_t subdomain_idx, size_t num_subdomains,
	const Position& min, const Position& max, std::vector<Position>& pos) const {

	const bool contains = neurons_in_subdomain.find(subdomain_idx) != neurons_in_subdomain.end();
	if (!contains) {
		assert(false && "Wanted to have neuron_positions of subdomain_idx that is not present");
		return;
	}

	// TODO: This loads the positions, that are only used once, into another vector.

	const Nodes& nodes = neurons_in_subdomain.at(subdomain_idx);
	for (const Node& node : nodes) {
		pos.push_back(node.pos);
	}
}

void NeuronToSubdomainAssignment::neuron_types(size_t subdomain_idx, size_t num_subdomains,
	const Position& min, const Position& max,
	std::vector<SynapticElements::SignalType>& types) const {

	const bool contains = neurons_in_subdomain.find(subdomain_idx) != neurons_in_subdomain.end();
	if (!contains) {
		assert(false && "Wanted to have neuron_types of subdomain_idx that is not present");
		return;
	}

	const Nodes& nodes = neurons_in_subdomain.at(subdomain_idx);
	for (const Node& node : nodes) {
		types.push_back(node.signal_type);
	}
}

void NeuronToSubdomainAssignment::neuron_area_names(size_t subdomain_idx, size_t num_subdomains,
	const Position& min, const Position& max, std::vector<std::string>& areas) const {

	const bool contains = neurons_in_subdomain.find(subdomain_idx) != neurons_in_subdomain.end();
	if (!contains) {
		assert(false && "Wanted to have neuron_area_names of subdomain_idx that is not present");
		return;
	}

	const Nodes& nodes = neurons_in_subdomain.at(subdomain_idx);
	for (const Node& node : nodes) {
		areas.push_back("no_area");
	}
}

bool NeuronToSubdomainAssignment::position_in_box(const Position& pos, const Position& box_min, const Position& box_max) const noexcept {
	return ((pos.x >= box_min.x && pos.x < box_max.x) &&
		(pos.y >= box_min.y && pos.y < box_max.y) &&
		(pos.z >= box_min.z && pos.z < box_max.z));
}

void NeuronToSubdomainAssignment::write_neurons_to_file(const std::string& filename) const {
	std::ofstream of(filename, std::ios::binary | std::ios::out);

	of << std::setprecision(std::numeric_limits<double>::digits10);

	for (const auto it : neurons_in_subdomain) {
		const Nodes& nodes = it.second;

		for (const auto& node : nodes) {
			of << node.pos.x << " "
				<< node.pos.y << " "
				<< node.pos.z << " ";

			if (node.signal_type == SynapticElements::SignalType::EXCITATORY) {
				of << "ex\n";
			}
			else {
				of << "in\n";
			}
		}
	}

	of.flush();
	of.close();
}
