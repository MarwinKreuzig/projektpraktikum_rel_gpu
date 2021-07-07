#include <optional>
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

#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include <CLI/Option.hpp>

#include "graph.h"
#include "position.h"

int main(int argc, char** argv) {
    CLI::App app{ "relearn-full_graph" };

    std::filesystem::path path{};
    app.add_option("-p,--path", path, "Path to folder that contains positions and edges")->required();

    std::filesystem::path output_path{ "./output/" };
    app.add_option("-o,--output-path", output_path, "Path to output folder")->capture_default_str();

    auto flag_cuda = [&]() -> std::optional<CLI::Option*> {
        if constexpr (CUDA_FOUND) {
            return { app.add_flag("--use-cuda", "Use CUDA algorithms") };
        }
        return std::nullopt;
    }();

    CLI11_PARSE(app, argc, argv);

    std::vector<std::filesystem::path> position_paths{};
    std::vector<std::filesystem::path> edges_paths{};

    for (const auto& entry : std::filesystem::directory_iterator(path)) {
        const std::filesystem::path& p = entry.path();
        const std::filesystem::path filename = p.filename();
        const std::string filename_str = filename.string();

        if (filename_str.rfind("positions") != std::string::npos) {
            position_paths.emplace_back(p);
        } else if (filename_str.rfind("network") != std::string::npos) {
            edges_paths.emplace_back(p);
        }
    }

    // Create output directory
    std::filesystem::create_directory(output_path);

    const std::filesystem::path output_path_pos = output_path.concat("positions.txt");
    const std::filesystem::path output_path_net = output_path.replace_filename("network.txt");

    Graph full_graph{};

    for (const auto& path : position_paths) {
        full_graph.add_vertices_from_file(path);
    }

    for (const auto& path : edges_paths) {
        full_graph.add_edges_from_file(path);
    }

    full_graph.set_use_cuda(static_cast<bool>(*flag_cuda.value()));

    std::ofstream file_positions(output_path_pos, std::ios::trunc);
    std::ofstream file_network(output_path_net, std::ios::trunc);

    // Print vertices
    file_positions << "# num_vertices: " << full_graph.get_num_vertices() << "\n";
    file_positions << "# num_edges: " << full_graph.get_num_edges() << "\n";

    auto [min_x, min_y, min_z] = full_graph.smallest_coordinate_per_dimension();
    file_positions << "# min_x: " << min_x << "\n";
    file_positions << "# min_y: " << min_y << "\n";
    file_positions << "# min_z: " << min_z << "\n";

    Position offset{};
    offset.x = min_x < 0 ? -min_x : 0;
    offset.y = min_y < 0 ? -min_y : 0;
    offset.z = min_z < 0 ? -min_z : 0;

    full_graph.add_offset_to_positions(offset);

    std::tie(min_x, min_y, min_z) = full_graph.smallest_coordinate_per_dimension();

    file_positions << "# Offset added\n";
    file_positions << "# min_x: " << min_x << "\n";
    file_positions << "# min_y: " << min_y << "\n";
    file_positions << "# min_z: " << min_z << "\n";

    auto [min_degree, max_degree] = full_graph.min_max_degree();
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
