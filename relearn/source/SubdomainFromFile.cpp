#include "SubdomainFromFile.h"
#include "NeuronToSubdomainAssignment.h"
#include "LogMessages.h"

#include <iostream>
#include <sstream>
#include <cmath>

SubdomainFromFile::SubdomainFromFile(std::string file_path, size_t num_neurons) : file(file_path) {
	std::cout << "Loading: " << file_path << std::endl;
	const bool file_is_good = file.good();
	const bool file_is_not_good = file.fail() || file.eof();

	if (!file_is_good || file_is_not_good) {
		std::cout << "Opening the file was not successful" << std::endl;
		exit(EXIT_FAILURE);
	}

	read_dimensions_from_file();
}

void SubdomainFromFile::read_dimensions_from_file() {
	Vec3d minimum(std::numeric_limits<double>::max());
	Vec3d maximum(std::numeric_limits<double>::min());

	std::string line;
	bool success = false;

	Vec3d tmp;
	size_t id;
	std::string area_name;

	double found_ex_neurons = 0.0;
	double found_in_neurons = 0.0;

	while (std::getline(file, line)) {
		// Skip line with comments
		if (!line.empty() && '#' == line[0]) {
			continue;
		}

		std::stringstream sstream(line);
		success =
			(sstream >> id) &&
			(sstream >> tmp.x) &&
			(sstream >> tmp.y) &&
			(sstream >> tmp.z) &&
			(sstream >> area_name);

		minimum.calculate_pointwise_minimum(tmp);
		maximum.calculate_pointwise_maximum(tmp);

		if (area_name == "ex") {
			found_ex_neurons++;
		}
		else {
			found_in_neurons++;
		}

		if (!success) {
			std::cerr << "Skipping line: \"" << line << "\"\n";
			continue;
		}
	}

	{
		maximum.x = std::nextafter(maximum.x, maximum.x + 0.1);
		maximum.y = std::nextafter(maximum.y, maximum.y + 0.1);
		maximum.z = std::nextafter(maximum.z, maximum.z + 0.1);
	}

	const double total_neurons = found_ex_neurons + found_in_neurons;

	currently_frac_neurons_exc_ = found_ex_neurons / total_neurons;
	desired_frac_neurons_exc_ = found_ex_neurons / total_neurons;

	this->simulation_box_length = maximum - minimum;
	offset = minimum;
}

void SubdomainFromFile::read_nodes_from_file(const Position& min, const Position& max, Nodes& nodes) {
	std::string line;
	Node node;
	bool success = false;

	file.clear();
	file.seekg(0);

	double placed_ex_neurons = 0.0;
	double placed_in_neurons = 0.0;

	while (std::getline(file, line)) {
		// Skip line with comments
		if (!line.empty() && '#' == line[0]) {
			continue;
		}

		std::cout << line << std::endl;

		std::stringstream sstream(line);
		success =
			(sstream >> node.id) &&
			(sstream >> node.pos.x) &&
			(sstream >> node.pos.y) &&
			(sstream >> node.pos.z) &&
			(sstream >> node.area_name);

		if (!success) {
			std::cerr << "Skipping line: \"" << line << "\"\n";
			continue;
		}

		node.pos = node.pos - this->offset;

		bool is_in_subdomain = this->position_in_box(node.pos, min, max);

		if (!is_in_subdomain) {
			continue;
		}

		if (node.area_name == "ex") {
			node.signal_type = SynapticElements::EXCITATORY;
			placed_ex_neurons++;
		}
		else {
			node.signal_type = SynapticElements::INHIBITORY;
			placed_in_neurons++;
		}

		nodes.insert(node);
	}
}

void SubdomainFromFile::neuron_global_ids(size_t subdomain_idx, size_t num_subdomains,
	size_t local_id_start, size_t local_id_end, std::vector<size_t>& global_ids) const {

	const bool contains = neurons_in_subdomain.find(subdomain_idx) != neurons_in_subdomain.end();
	if (!contains) {
		assert(false && "Wanted to have neuron_global_ids of subdomain_idx that is not present");
		return;
	}

	const Nodes& nodes = neurons_in_subdomain.at(subdomain_idx);
	for (const Node& node : nodes) {
		//global_ids.push_back(node.id);
	}
}

void SubdomainFromFile::fill_subdomain(size_t subdomain_idx, size_t num_subdomains, const Position& min, const Position& max) {
	const bool subdomain_already_filled = neurons_in_subdomain.find(subdomain_idx) != neurons_in_subdomain.end();
	if (subdomain_already_filled) {
		assert(false && "Tried to fill an already filled subdomain.");
		return;
	}

	Nodes& nodes = neurons_in_subdomain[subdomain_idx];

	read_nodes_from_file(min, max, nodes);
}
