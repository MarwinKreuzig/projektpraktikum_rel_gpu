#include "johnson.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <random> // mt19937_64, uniform_x_distribution
#include <vector>
#include <mutex>
#include <shared_mutex>
#include <atomic>

#include <spdlog/spdlog.h>
#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/johnson_all_pairs_shortest.hpp>

namespace apsp {

static bool bellman_ford(const graph_t& gr, std::vector<double>& dist) {
    const int& V = gr.V;
    const int& E = gr.E;
    const auto& edges = gr.edge_array;
    const auto& weights = gr.weights;

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < V; i++) {
        dist[i] = std::numeric_limits<double>::max();
    }
    dist.back() = 0;

    for (int i = 1; i < V; i++) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int j = 0; j < E; j++) {
            const auto [u, v] = edges[j];

            std::atomic_ref<double> dist_u{ dist[u] };
            const auto new_dist = weights[j] + dist_u.load(std::memory_order_relaxed);

            std::atomic_ref<double> dist_a{ dist[v] };
            auto current_dist = dist_a.load(std::memory_order_relaxed);
            while (new_dist < current_dist && !dist_a.compare_exchange_weak(current_dist, new_dist, std::memory_order_relaxed)) {
            }
        }
    }

    bool no_neg_cycle = true;
#ifdef _OPENMP
#pragma omp parallel for reduction(&& \
                                   : no_neg_cycle)
#endif
    for (int i = 0; i < E; i++) {
        const auto [u, v] = edges[i];
        const auto weight = weights[i];
        if (dist[u] != std::numeric_limits<double>::max() && dist[u] + weight < dist[v]) {
            no_neg_cycle = false;
        }
    }
    return no_neg_cycle;
}

void johnson_parallel_impl(graph_t& gr, std::vector<double>& output, const bool has_negative_edges) {

    const int V = gr.V;

    if (has_negative_edges) {
        // Make new graph for Bellman-Ford
        // First, a new node q is added to the graph, connected by zero-weight edges
        // to each of the other nodes.
        graph_t bf_graph{ V + 1, gr.E + V };
        std::copy(gr.edge_array.begin(), gr.edge_array.end(), bf_graph.edge_array.begin());
        std::copy(gr.weights.begin(), gr.weights.end(), bf_graph.weights.begin());
        std::fill(bf_graph.weights.begin() + gr.E, bf_graph.weights.begin() + gr.E + V, 0);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (int e = 0; e < V; e++) {
            bf_graph.edge_array[e + gr.E] = APSP_Edge(V, e);
        }

        // Check if there is a negative edge in the graph
        std::vector<double> h(bf_graph.V);
        if (const bool r = bellman_ford(bf_graph, h); !r) {
            spdlog::error("Johnson: Negative cycles deteced! Terminating program");
            std::terminate();
        }
    }

    APSP_Graph G(gr.edge_array.begin(), gr.edge_array.end(), gr.weights.begin(), V);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int s = 0; s < V; s++) {
        std::vector<double> d(V);
        boost::dijkstra_shortest_paths(G, s, boost::distance_map(d.data()));
        for (int v = 0; v < V; v++) {
            output[static_cast<size_t>(s) * static_cast<size_t>(V) + static_cast<size_t>(v)] = d[v];
        }
    }
}

} // namespace apsp
