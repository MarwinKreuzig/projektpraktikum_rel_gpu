#include "GraphTest.hpp"

#include <filesystem>
#include <gtest/gtest.h>

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

static const std::string path_to_cases = "../test/cases";

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
    doTest(example, true);

    // If CUDA is available do CPU test,
    // otherwise test already done on CPU
    if constexpr (CUDA_FOUND) {
        doTest(example, false);
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
