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
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <folder that contains positions and edges>" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<std::filesystem::path> position_paths;
    std::vector<std::filesystem::path> edges_paths;

    std::filesystem::path input_path(argv[1]);

    for (const auto& entry : std::filesystem::directory_iterator(input_path)) {
        const std::filesystem::path& p = entry.path();
        const std::filesystem::path filename = p.filename();
        const std::string filename_str = filename.string();

        if (filename_str.rfind("positions", 0) == 0) {
            position_paths.emplace_back(p);
        } else if (filename_str.rfind("network", 0) == 0) {
            edges_paths.emplace_back(p);
        }
    }

    // Create output directory
    std::filesystem::path output_path("./output/");
    std::filesystem::create_directory(output_path);

    std::filesystem::path output_path_pos = output_path.concat("positions.txt");
    std::filesystem::path output_path_net = output_path.replace_filename("network.txt");

    Graph full_graph;

    for (const auto& path : position_paths) {
        full_graph.add_vertices_from_file(path);
    }

    for (const auto& path : edges_paths) {
        full_graph.add_edges_from_file(path);
    }

    std::ofstream file_positions(output_path_pos, std::ios::trunc);
    std::ofstream file_network(output_path_net, std::ios::trunc);

    // Print vertices
    file_positions << "# num_vertices: " << full_graph.get_num_vertices() << "\n";
    file_positions << "# num_edges: " << full_graph.get_num_edges() << "\n";

    double min_x, min_y, min_z;
    std::tie(min_x, min_y, min_z) = full_graph.smallest_coordinate_per_dimension();

    file_positions << "# min_x: " << min_x << "\n";
    file_positions << "# min_y: " << min_y << "\n";
    file_positions << "# min_z: " << min_z << "\n";

    Position offset;
    offset.x = min_x < 0 ? -min_x : 0;
    offset.y = min_y < 0 ? -min_y : 0;
    offset.z = min_z < 0 ? -min_z : 0;

    full_graph.add_offset_to_positions(offset);

    std::tie(min_x, min_y, min_z) = full_graph.smallest_coordinate_per_dimension();

    file_positions << "# Offset added\n";
    file_positions << "# min_x: " << min_x << "\n";
    file_positions << "# min_y: " << min_y << "\n";
    file_positions << "# min_z: " << min_z << "\n";

    int min_degree, max_degree;
    std::tie(min_degree, max_degree) = full_graph.min_max_degree();

    file_positions << "# min vertex degree: " << min_degree << "\n";
    file_positions << "# max vertex degree: " << max_degree << "\n";

    file_positions << std::fixed << std::setprecision(6);
    full_graph.print_vertices(file_positions);
    file_positions << std::defaultfloat;
    std::cout << "Created " << output_path_pos << "\n";

    // Print edges
    file_network << std::fixed << std::setprecision(6);
    full_graph.print_edges(file_network);
    file_network << std::defaultfloat;
    std::cout << "Created " << output_path_net << "\n";

    full_graph.calculate_metrics(std::cout);

    return 0;
}
