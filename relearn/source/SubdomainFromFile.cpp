/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "SubdomainFromFile.h"

#include "LogMessages.h"
#include "NeuronToSubdomainAssignment.h"
#include "RelearnException.h"

#include <cmath>
#include <iostream>
#include <sstream>

SubdomainFromFile::SubdomainFromFile(const std::string &file_path) : file(file_path) {
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

	size_t found_ex_neurons = 0;
	size_t found_in_neurons = 0;

	for (std::string line{}; std::getline(file, line);) {
		// Skip line with comments
		if (!line.empty() && '#' == line[0]) {
			continue;
		}

		size_t id{};
		Vec3d tmp{};
		std::string area_name{};
		std::string signal_type{};

		std::stringstream sstream(line);
		bool success =
			(sstream >> id) &&
			(sstream >> tmp.x) &&
			(sstream >> tmp.y) &&
			(sstream >> tmp.z) &&
			(sstream >> area_name) &&
			(sstream >> signal_type);

		minimum.calculate_componentwise_minimum(tmp);
		maximum.calculate_componentwise_maximum(tmp);

		if (signal_type == "ex") {
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

	desired_num_neurons_ = found_ex_neurons + found_in_neurons;
	desired_frac_neurons_exc_ = static_cast<double>(found_ex_neurons) / static_cast<double>(desired_num_neurons_);

	simulation_box_length = maximum - minimum;
	offset = minimum;
}

void SubdomainFromFile::read_nodes_from_file(const Position& min, const Position& max, Nodes& nodes) {
	file.clear();
	file.seekg(0);

	double placed_ex_neurons = 0.0;
	double placed_in_neurons = 0.0;

	for (std::string line{}; std::getline(file, line);) {
		// Skip line with comments
		if (!line.empty() && '#' == line[0]) {
			continue;
		}

		std::cout << line << "\n";

		std::string signal_type{};

		Node node{};

		std::stringstream sstream(line);
		bool success =
			(sstream >> node.id) &&
			(sstream >> node.pos.x) &&
			(sstream >> node.pos.y) &&
			(sstream >> node.pos.z) &&
			(sstream >> node.area_name) &&
			(sstream >> signal_type);

		if (!success) {
			std::cerr << "Skipping line: \"" << line << "\"\n";
			continue;
		}

		node.pos = node.pos - offset;

		bool is_in_subdomain = position_in_box(node.pos, min, max);

		if (!is_in_subdomain) {
			continue;
		}

		if (signal_type == "ex") {
			node.signal_type = SynapticElements::EXCITATORY;
			++placed_ex_neurons;
		}
		else {
			node.signal_type = SynapticElements::INHIBITORY;
			++placed_in_neurons;
		}

		++currently_num_neurons_;
		nodes.insert(node);
	}
	currently_frac_neurons_exc_ = placed_ex_neurons / static_cast<double>(currently_num_neurons_);
}

void SubdomainFromFile::neuron_global_ids(size_t subdomain_idx, size_t num_subdomains,
	size_t local_id_start, size_t local_id_end, std::vector<size_t>& global_ids) const {

	const bool contains = neurons_in_subdomain.find(subdomain_idx) != neurons_in_subdomain.end();
	if (!contains) {
		RelearnException::check(false, "Wanted to have neuron_global_ids of subdomain_idx that is not present");
		return;
	}

	const Nodes& nodes = neurons_in_subdomain.at(subdomain_idx);
	for (const Node& node : nodes) {
		global_ids.push_back(node.id);
	}
}

void SubdomainFromFile::fill_subdomain(size_t subdomain_idx, size_t num_subdomains, const Position& min, const Position& max) {
	const bool subdomain_already_filled = neurons_in_subdomain.find(subdomain_idx) != neurons_in_subdomain.end();
	if (subdomain_already_filled) {
		RelearnException::check(false, "Tried to fill an already filled subdomain.");
		return;
	}

	Nodes& nodes = neurons_in_subdomain[subdomain_idx];

	read_nodes_from_file(min, max, nodes);
}
