#include "SubdomainFromFile.h"
#include "LogMessages.h"

#include <iostream>
#include <sstream>


SubdomainFromFile::SubdomainFromFile(std::string file_path, size_t num_neurons) : file(file_path) {
	//read_nodes_from_file(nodes_);
	read_dimensions_from_file();
	//max_dimension_ = largest_dimension(nodes_);

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

//// Make sure that the length is larger than the largest coordinate of any neuron.
//// That is, for all neuron coordinates (x,y,z), x,y,z in [0 , length).
//double SubdomainFromFile::simulation_box_length() const noexcept {
//	return std::nextafter(max_dimension_, max_dimension_ + 1);
//}

void SubdomainFromFile::read_dimensions_from_file() {
	Vec3d minimum(std::numeric_limits<double>::max());
	Vec3d maximum(std::numeric_limits<double>::min());

	std::string line;
	bool success = false;

	Vec3d tmp;
	std::string area_name;

	while (std::getline(file, line)) {
		// Skip line with comments
		if (!line.empty() && '#' == line[0]) {
			continue;
		}

		std::stringstream sstream(line);
		success = (sstream >> tmp.x) &&
			(sstream >> tmp.y) &&
			(sstream >> tmp.z) &&
			(sstream >> area_name);

		minimum.calculate_pointwise_minimum(tmp);
		maximum.calculate_pointwise_maximum(tmp);

		if (!success) {
			std::cerr << "Skipping line: \"" << line << "\"\n";
			continue;
		}
	}

	this->simulation_box_length = maximum - minimum;
	offset = minimum;
}

void SubdomainFromFile::read_nodes_from_file(const Position& min, const Position& max, std::set<Node, Node::less>& nodes) {
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

		if (!success) {
			std::cerr << "Skipping line: \"" << line << "\"\n";
			continue;
		}

		node.pos = node.pos - this->offset;

		bool is_in_subdomain = this->position_in_box(node.pos, min, max);

		if (!is_in_subdomain) {
			continue;
		}

		node.signal_type = SynapticElements::EXCITATORY;
		nodes.insert(node);
	}
}

//double SubdomainFromFile::largest_dimension(const Nodes& nodes) {
//	double max_of_all = 0.0;
//
//	for (const Node& node : nodes) {
//		auto max_of_node = std::max(node.pos.x, std::max(node.pos.y, node.pos.z));
//		max_of_all = std::max(max_of_node, max_of_all);
//	}
//
//	return max_of_all;
//}

void SubdomainFromFile::fill_subdomain(size_t subdomain_idx, size_t num_subdomains, const Position& min, const Position& max) {
	if (filled_subdomains_.find(subdomain_idx) != filled_subdomains_.end()) {
		return;
	}

	filled_subdomains_.insert(subdomain_idx);

	read_nodes_from_file(min, max, this->nodes_);
}
