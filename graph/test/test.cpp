#include "GraphTest.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <limits>

#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>

#define protected public
#define private public

#include "../source/graph.h"
#include "../source/apsp/apsp.h"

struct TestExample {
    Graph graph;
    double expected_sum;
    double expected_avg;
    std::vector<double> expected_distances;
};

static void doTest(TestExample& example, bool use_cuda_if_available) {
    auto& [graph, expected_sum, expected_avg, expected_distances] = example;
    const auto distances = apsp::johnson(graph.full_graph, graph.get_num_vertices(), use_cuda_if_available);

    EXPECT_EQ(distances, expected_distances);

    const auto num_neurons = graph.get_num_vertices();

    size_t number_values = 0;
    double avg = 0.0;
    double sum = 0.0;
    for (size_t i = 0; i < num_neurons; i++) {
        for (size_t j = 0; j < num_neurons; j++) {
            // Consider pairs of different neurons only
            if (i != j) {
                const double val = distances[i * num_neurons + j];

                if (val == std::numeric_limits<double>::max()) {
                    continue;
                }

                // Average
                number_values++;
                const double delta = val - avg;
                avg += delta / static_cast<double>(number_values);

                // Sum
                if (val != 0.0) {
                    sum += 1 / val;
                }
            }
        }
    }

    EXPECT_NEAR(sum, expected_sum, 10e-5);
    EXPECT_NEAR(avg, expected_avg, 10e-5);
}

static void test(TestExample example) {
    doTest(example, false);

    // If CUDA is available do GPU test
    if (enable_cuda) {
        doTest(example, true);
    }
}

static TestExample createExample1() {
    // example from https://en.wikipedia.org/wiki/Johnson%27s_algorithm
    Graph graph{};

    Position posW{ 0, 0, 0 };
    graph.add_vertex(posW, "W", 0);

    Position posX{ 1, 0, 0 };
    graph.add_vertex(posX, "X", 1);

    Position posY{ 0, 1, 0 };
    graph.add_vertex(posY, "Y", 2);

    Position posZ{ 0, 0, 1 };
    graph.add_vertex(posZ, "Z", 3);

    graph.add_edge(0, 3, 2);

    graph.add_edge(1, 0, 6);
    graph.add_edge(1, 2, 3);

    graph.add_edge(2, 0, 4);
    graph.add_edge(2, 3, 5);

    graph.add_edge(3, 1, -7);
    graph.add_edge(3, 2, -3);

    return { graph, 2.58849, 5.91667, { 0, 9, 5, 2, 6, 0, 3, 8, 4, 12, 0, 5, 7, 7, 3, 0 } };
}

TEST(TestAPSP, testExampleGraph1) {
    test(createExample1());
}

static TestExample createExample2() {
    Graph graph{};

    Position posA{ 0, 0, 0 };
    graph.add_vertex(posA, "A", 0);

    Position posB{ 1, 0, 0 };
    graph.add_vertex(posB, "B", 1);

    Position posC{ 0, 1, 0 };
    graph.add_vertex(posC, "C", 2);

    Position posD{ 0, 0, 1 };
    graph.add_vertex(posD, "D", 3);

    Position posE{ 1, 0, 1 };
    graph.add_vertex(posE, "E", 4);

    Position posF{ 1, 1, 0 };
    graph.add_vertex(posF, "F", 5);

    Position posG{ 0, 1, 1 };
    graph.add_vertex(posG, "G", 6);

    Position posH{ 1, 1, 1 };
    graph.add_vertex(posH, "H", 7);

    graph.add_edge(0, 1, 1);
    graph.add_edge(0, 2, 1);
    graph.add_edge(1, 3, 1);
    graph.add_edge(2, 3, 1);
    graph.add_edge(2, 6, 1);
    graph.add_edge(2, 7, 1);
    graph.add_edge(3, 4, 1);
    graph.add_edge(4, 5, 1);
    graph.add_edge(6, 5, 1);
    graph.add_edge(7, 6, 1);

    // clang-format off
    return { graph, 15, 1.6667,
        {
        0,      1,      1,      2,      3,      3,    2,        2,
        max,    0,      max,    1,      2,      3,    max,      max,
        max,    max,    0,      1,      2,      2,    1,        1,
        max,    max,    max,    0,      1,      2,    max,      max,
        max,    max,    max,    max,    0,      1,    max,      max,
        max,    max,    max,    max,    max,    0,    max,      max,
        max,    max,    max,    max,    max,    1,    0,        max,
        max,    max,    max,    max,    max,    2,    1,        0
        }
    };
    // clang-format on
}

TEST(TestAPSP, testExampleGraph2) {
    test(createExample2());
}

static TestExample createExample3() {
    // circle
    Graph graph{};

    Position posX{ 1, 0, 0 };
    graph.add_vertex(posX, "X", 0);

    Position posY{ 0, 1, 0 };
    graph.add_vertex(posY, "Y", 1);

    Position posZ{ 0, 0, 1 };
    graph.add_vertex(posZ, "Z", 2);

    graph.add_edge(0, 1, 1);
    graph.add_edge(1, 2, 1);
    graph.add_edge(2, 0, 1);

    return { graph, 4.5, 1.5, { 0, 1, 2, 2, 0, 1, 1, 2, 0 } };
}

TEST(TestAPSP, testExampleGraph3) {
    test(createExample3());
}

static TestExample createExample4() {
    // circle with negative weight
    Graph graph{};

    Position posX{ 1, 0, 0 };
    graph.add_vertex(posX, "X", 0);

    Position posY{ 0, 1, 0 };
    graph.add_vertex(posY, "Y", 1);

    Position posZ{ 0, 0, 1 };
    graph.add_vertex(posZ, "Z", 2);

    graph.add_edge(0, 1, 1);
    graph.add_edge(1, 2, -1);
    graph.add_edge(2, 0, 1);

    return { graph, 4.5, 1.5, { 0, 1, 2, 2, 0, 1, 1, 2, 0 } };
}

TEST(TestAPSP, testExampleGraph4) {
    test(createExample4());
}

static TestExample createExample5() {
    // straight line
    Graph graph{};

    Position posX{ 1, 0, 0 };
    graph.add_vertex(posX, "X", 0);

    Position posY{ 0, 1, 0 };
    graph.add_vertex(posY, "Y", 1);

    Position posZ{ 0, 0, 1 };
    graph.add_vertex(posZ, "Z", 2);

    graph.add_edge(0, 1, 1);
    graph.add_edge(1, 2, 1);

    return { graph, 2.5, 1.3333, { 0, 1, 2, max, 0, 1, max, max, 0 } };
}

TEST(TestAPSP, testExampleGraph5) {
    test(createExample5());
}

static TestExample createExample6() {
    // https://en.wikipedia.org/wiki/Directed_graph
    // https://en.wikipedia.org/wiki/File:4-tournament.svg
    Graph graph{};

    graph.add_vertex({ 0, 0, 0 }, "A", 0);
    graph.add_vertex({ 0, 0, 1 }, "B", 1);
    graph.add_vertex({ 0, 1, 0 }, "C", 2);
    graph.add_vertex({ 1, 0, 0 }, "D", 3);

    graph.add_edge(0, 1, 1);
    graph.add_edge(0, 3, 1);
    graph.add_edge(1, 3, 1);
    graph.add_edge(2, 0, 1);
    graph.add_edge(2, 1, 1);
    graph.add_edge(3, 2, 1);

    return { graph, 8.83333, 1.583333, { 0, 1, 2, 1, 3, 0, 2, 1, 1, 1, 0, 2, 2, 2, 1, 0 } };
}

TEST(TestAPSP, testExampleGraph6) {
    test(createExample6());
}

static TestExample createExample7() {
    // https://en.wikipedia.org/wiki/Directed_graph
    // https://en.wikipedia.org/wiki/File:Directed_acyclic_graph_2.svg
    Graph graph{};

    graph.add_vertex({ 0, 0, 0 }, "A", 0);
    graph.add_vertex({ 0, 0, 1 }, "B", 1);
    graph.add_vertex({ 0, 1, 0 }, "C", 2);
    graph.add_vertex({ 1, 0, 0 }, "D", 3);
    graph.add_vertex({ 1, 0, 1 }, "E", 4);
    graph.add_vertex({ 1, 1, 0 }, "F", 5);
    graph.add_vertex({ 1, 1, 1 }, "G", 6);
    graph.add_vertex({ 0, 1, 1 }, "H", 7);

    graph.add_edge(0, 1, 1);
    graph.add_edge(1, 2, 1);
    graph.add_edge(1, 5, 1);
    graph.add_edge(1, 7, 1);
    graph.add_edge(3, 1, 1);
    graph.add_edge(3, 4, 1);
    graph.add_edge(4, 5, 1);
    graph.add_edge(6, 4, 1);
    graph.add_edge(6, 7, 1);

    // clang-format off
    return { graph, 12.5, 1.4375,
    {
        0,      1,      2,      max,    max,    2,      max,    2,
        max,    0,      1,      max,    max,    1,      max,    1,
        max,    max,    0,      max,    max,    max,    max,    max,
        max,    1,      2,      0,      1,      2,      max,    2,
        max,    max,    max,    max,    0,      1,      max,    max,
        max,    max,    max,    max,    max,    0,      max,    max,
        max,    max,    max,    max,    1,      2,      0,      1,
        max,    max,    max,    max,    max,    max,    max,    0
    }
    };
    // clang-format on
}

TEST(TestAPSP, testExampleGraph7) {
    test(createExample7());
}
