#include <gtest/gtest.h>

#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>

#define protected public
#define private public

#include "../source/graph.h"
#include "../source/apsp/johnson.hpp"

struct TestExample {
    Graph graph;
    double expected_sum;
    double expected_avg;
};

TestExample createExample1() {
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

    return { graph, 2.58849, 5.91667 };
}

void doTest(TestExample& example, bool use_cuda_if_available) {
    auto& [graph, expected_sum, expected_avg] = example;
    const auto distances = apsp::johnson(graph.full_graph, graph.get_num_vertices(), use_cuda_if_available);

    EXPECT_EQ(distances, (std::vector<double>{ 0, 9, 5, 2, 6, 0, 3, 8, 4, 12, 0, 5, 7, 7, 3, 0 }));

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
                sum += 1 / val;
            }
        }
    }

    EXPECT_NEAR(sum, expected_sum, 10e-5);
    EXPECT_NEAR(avg, expected_avg, 10e-5);
}

TEST(TestAPSP, testExampleGraph1) {
    auto example = createExample1();
    doTest(example, true);

    // If CUDA is available do CPU test,
    // otherwise test already done on CPU
    if constexpr (!CUDA_FOUND) {
        doTest(example, false);
    }
}
