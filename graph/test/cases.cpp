#include "GraphTest.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>

#define protected public
#define private public

#include "../source/graph.h"
#include "../source/apsp/apsp.h"

struct TestExample {
    Graph graph;
    double expected_avg{};
    double global_efficiency{};
};

static void doTest(TestExample& example, bool use_cuda_if_available) {
    auto& [graph, expected_avg, expected_glob_efficiency] = example;

    graph.set_use_cuda(use_cuda_if_available);

    const auto [avg, glob_efficiency] = graph.calculate_all_pairs_shortest_paths();

    EXPECT_NEAR(avg, expected_avg, 10e-5);
    EXPECT_NEAR(glob_efficiency, expected_glob_efficiency, 10e-5);
}

// Needed to get the correct path to the test cases, no matter the working directory
// Allows calling the tests from any dir (ctest)
[[nodiscard]] static std::string get_this_file_directory() {
    // Future: replace __FILE__ with std::source_location::current()'s file_name()
    std::filesystem::path p{ __FILE__ };
    return p.parent_path().string() + "/";
}

static const std::string path_to_cases = get_this_file_directory() + "../test/cases";

static Graph load_graph(const std::filesystem::path& path) {
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

    Graph graph{};

    for (const auto& path : position_paths) {
        graph.add_vertices_from_file(path);
    }

    for (const auto& path : edges_paths) {
        graph.add_edges_from_file(path);
    }

    return graph;
}

static void test(TestExample example) {
    doTest(example, false);

    // If CUDA is available do GPU test
    if (enable_cuda) {
        doTest(example, true);
    }
}

TEST(TestAPSPCases, case25) {
    test({ load_graph(path_to_cases + "/25"), 1.36333, 0.819444 });
}

TEST(TestAPSPCases, case100) {
    test({ load_graph(path_to_cases + "/100"), 1.495959, 0.754646 });
}

TEST(TestAPSPCases, case250) {
    test({ load_graph(path_to_cases + "/250"), 1.516417, 0.743194 });
}

TEST(TestAPSPCases, case500) {
    test({ load_graph(path_to_cases + "/500"), 1.49873, 0.751158 });
}
