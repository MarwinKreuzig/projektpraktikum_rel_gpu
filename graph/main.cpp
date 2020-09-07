#include <iostream>
#include <set>
#include <map>
#include <string>
#include <vector>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <limits>

#include <sys/stat.h>

#include "graph.h"
#include "position.h"

int main(int argc, char** argv)
{
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
	std::string output_dir("./output/");
	mkdir(output_dir.c_str(), S_IRWXU);

	// Open output file for positions
	std::string filename_positions(output_dir + "positions.txt");
	std::ofstream file_positions(filename_positions, std::ios::trunc);
	if (file_positions.fail()) {
		std::cout << "Opening file " << filename_positions << " failed." << std::endl;
		exit(EXIT_FAILURE);
	}

	// Open output file for network
	std::string filename_network(output_dir + "network.txt");
	std::ofstream file_network(filename_network, std::ios::trunc);
	if (file_network.fail()) {
		std::cout << "Opening file " << filename_network << " failed." << std::endl;
		exit(EXIT_FAILURE);
	}

	Graph graph;
	graph.Init(input_file);

	// Print vertices
	file_positions << "# num_vertices: " << boost::num_vertices(graph.BGL_Graph()) << "\n";
	file_positions << "# num_edges: " << boost::num_edges(graph.BGL_Graph()) << "\n";

	double min_x, min_y, min_z;
	std::tie(min_x, min_y, min_z) = graph.SmallestCoordinatePerDimension();

	file_positions << "# min_x: " << min_x << "\n";
	file_positions << "# min_y: " << min_y << "\n";
	file_positions << "# min_z: " << min_z << "\n";

	Position offset;
	offset.x = min_x < 0 ? -min_x : 0;
	offset.y = min_y < 0 ? -min_y : 0;
	offset.z = min_z < 0 ? -min_z : 0;

	graph.AddOffsetToPositions(offset);

	std::tie(min_x, min_y, min_z) = graph.SmallestCoordinatePerDimension();

	file_positions << "# Offset added\n";
	file_positions << "# min_x: " << min_x << "\n";
	file_positions << "# min_y: " << min_y << "\n";
	file_positions << "# min_z: " << min_z << "\n";

	int min_degree, max_degree;
	std::tie(min_degree, max_degree) = graph.MinMaxDegree();

	file_positions << "# min vertex degree: " << min_degree << "\n";
	file_positions << "# max vertex degree: " << max_degree << "\n";

    file_positions << std::fixed << std::setprecision(6);
    graph.PrintVertices(file_positions);
    file_positions << std::defaultfloat;
	std::cout << "Created " << filename_positions << "\n";

	// Print edges
    file_network << std::fixed << std::setprecision(6);
	graph.PrintEdges(file_network);
    file_network << std::defaultfloat;
	std::cout << "Created " << filename_network << "\n";

    return 0;
}
