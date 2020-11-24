#include "graph.h"
#include "position.h"

#include <sys/stat.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

int main(int argc, char** argv) {
	// Check number of parameters
	if (argc < 2) {
		std::cout << "Usage: " << argv[0] << " <input network file with positions>" << std::endl;
		exit(EXIT_FAILURE);
	}

	// Open input file
	std::string filename_input(argv[1]);
	std::ifstream input_file(filename_input);
	if (input_file.fail()) {
		std::cout << "Opening file " << filename_input << " failed." << std::endl;
		exit(EXIT_FAILURE);
	}

	// Create output directory
	std::filesystem::path output_path("./output/");
	std::filesystem::create_directory(output_path);

	std::filesystem::path output_path_pos = output_path.concat("positions.txt");
	std::filesystem::path output_path_net = output_path.concat("network.txt");

	//mkdir(output_dir.c_str(), S_IRWXU);

	// Open output file for positions
	// std::string filename_positions(output_path_pos);
	std::ofstream file_positions(output_path_pos, std::ios::trunc);
	if (file_positions.fail()) {
		std::cout << "Opening file " << output_path_pos << " failed." << std::endl;
		exit(EXIT_FAILURE);
	}

	// Open output file for network
	//std::string filename_network(output_dir + "network.txt");
	std::ofstream file_network(output_path_net, std::ios::trunc);
	if (file_network.fail()) {
		std::cout << "Opening file " << output_path_net << " failed." << std::endl;
		exit(EXIT_FAILURE);
	}

	Graph graph;
	graph.init(input_file);

	// Print vertices
	file_positions << "# num_vertices: " << boost::num_vertices(graph.BGL_Graph()) << "\n";
	file_positions << "# num_edges: " << boost::num_edges(graph.BGL_Graph()) << "\n";

	double min_x, min_y, min_z;
	std::tie(min_x, min_y, min_z) = graph.smallest_coordinate_per_dimension();

	file_positions << "# min_x: " << min_x << "\n";
	file_positions << "# min_y: " << min_y << "\n";
	file_positions << "# min_z: " << min_z << "\n";

	Position offset;
	offset.x = min_x < 0 ? -min_x : 0;
	offset.y = min_y < 0 ? -min_y : 0;
	offset.z = min_z < 0 ? -min_z : 0;

	graph.add_offset_to_positions(offset);

	std::tie(min_x, min_y, min_z) = graph.smallest_coordinate_per_dimension();

	file_positions << "# Offset added\n";
	file_positions << "# min_x: " << min_x << "\n";
	file_positions << "# min_y: " << min_y << "\n";
	file_positions << "# min_z: " << min_z << "\n";

	int min_degree, max_degree;
	std::tie(min_degree, max_degree) = graph.min_max_degree();

	file_positions << "# min vertex degree: " << min_degree << "\n";
	file_positions << "# max vertex degree: " << max_degree << "\n";

	file_positions << std::fixed << std::setprecision(6);
	graph.print_vertices(file_positions);
	file_positions << std::defaultfloat;
	std::cout << "Created " << output_path_pos << "\n";

	// Print edges
	file_network << std::fixed << std::setprecision(6);
	graph.print_edges(file_network);
	file_network << std::defaultfloat;
	std::cout << "Created " << output_path_net << "\n";

	return 0;
}
