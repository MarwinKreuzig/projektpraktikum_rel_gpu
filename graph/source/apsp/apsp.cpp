#include "apsp.h"

#include <algorithm>
#include <cassert>
#include <future>
#include <ranges>
#include <thread>
#include <vector>

#include <spdlog/fmt/bundled/core.h>

#include "johnson.h"
#include "../util.h"

namespace apsp {

std::vector<double> johnson(typename Graph::FullGraph& full_graph, const size_t num_neurons, const bool has_negative_edges, const bool use_cuda_if_available) {

    if constexpr (CUDA_FOUND) {
        if (use_cuda_if_available) {
            return johnson_cuda(full_graph, num_neurons, has_negative_edges);
        }
    }
    return johnson_parallel(full_graph, num_neurons, has_negative_edges);
}

static std::vector<int> johnson_get_weights_vector(auto edge_begin_it, const auto edge_end_it, const auto& weight_map) {
    std::vector<int> weights{};
    std::transform(edge_begin_it, edge_end_it, std::back_inserter(weights), [&](const auto& edge) {
        return weight_map(edge);
    });
    return weights;
}

template <typename EdgeType>
static std::vector<EdgeType> johnson_get_edge_vector(auto edge_begin_it, const auto edge_end_it, const auto E) {
    std::vector<EdgeType> edges(E);
    std::transform(edge_begin_it, edge_end_it, edges.begin(), [](const auto& edge) {
        return EdgeType{ static_cast<int>(edge.m_source), static_cast<int>(edge.m_target) };
    });
    return edges;
}

/**
 * @brief Create the starts vector required for johnson_cuda
 *
 * The starts vector holds the starting indices i for the range of edges that
 * start at the vertex i. If no such edge exists -1 is stored.
 * To get the index range for a vertex i, call starts[i] for the begin, to get
 * the end index iterate though the starts vector until a value other than -1 is found.
 * The returned vector is one larger than there are vertices to set the end index of the last element.
 *
 * @param cuda_edges Sorted edges to generate the starts vector from
 * @param num_neurons number of vertices
 * @return std::vector<int> starts vector
 */
static std::vector<int> johnson_cuda_generate_starts_vector(const auto& cuda_edges, const auto num_neurons) {
    auto starts = std::vector<int>(num_neurons + 1, -1); // Starting point for each edge

    auto edge_it = cuda_edges.cbegin();
    const auto edge_end_it = cuda_edges.cend();
    int starts_index_counter = 0;

    // Predicate: The current edge is ordered before the one we search for
    auto edge_range_for_idx_not_yet_reached = [](const edge_t& edge, const auto& idx) {
        return edge.u < static_cast<int>(idx);
    };

    // Predicate: The current edge is ordered after the one we search for
    auto edge_range_for_idx_passed = [](const edge_t& edge, const auto& idx) {
        return edge.u > static_cast<int>(idx);
    };

    for (size_t i = 0U; i < starts.size(); ++i) {
        // When the current edge's start (u) is smaller than i,
        // then we have not yet reached i's range of edges,
        // unless there is no edge for i and we skip it.
        while (edge_it != edge_end_it && edge_range_for_idx_not_yet_reached(*edge_it, i) && !edge_range_for_idx_passed(*edge_it, i)) {
            ++edge_it;
            ++starts_index_counter;
        }

        // This and all other edges have no edges
        if (edge_it == edge_end_it) {
            break;
        }

        // Found no edge for i, skipping
        if (edge_range_for_idx_passed(*edge_it, i)) {
            continue;
        }

        // Getting here means that edge_it points to the first edge that has idx as it's u value.
        // Set the starts index for vertex i to starts_index_counter
        starts[i] = starts_index_counter;
    }
    starts.back() = cuda_edges.size();

    return starts;
}

/**
 * @brief Sorts the edges and weights such that the u (starts) in cuda_edges are in ascending order.
 *
 * The CUDA version of Johnson relies on the edges being sorted. We have to sort both edges and weights together
 * to keep the correct weights for each edge at the same index as each edge.
 *
 * @param weights edge weights
 * @param cuda_edges edges to sort
 */
static void johnson_cuda_sort_edges_and_weights(auto& weights, auto& cuda_edges) {
    // Need to sort edges by their starting vertex id
    // Need to zip weights and edges to keep the correct weight for each edge
    std::vector<std::pair<int, edge_t>> zipped{};
    std::ranges::transform(
        weights,
        cuda_edges,
        std::back_inserter(zipped),
        [](int& weight, edge_t& edge) -> std::pair<int, edge_t> {
            return { weight, edge };
        });

    std::ranges::sort(
        zipped,
        [](const auto& a, const auto& b) -> bool {
            const auto& [weight_a, edge_a] = a;
            const auto& [weight_b, edge_b] = b;
            if (!(edge_a.u < edge_b.u)) {
                if (edge_a.u == edge_b.u) {
                    return edge_a.v < edge_b.v;
                }
                return false;
            }
            return true;
        });

    // Unzip
    std::ranges::transform(zipped, weights.begin(), [](const auto& a) { return std::get<0>(a); });
    std::ranges::transform(zipped, cuda_edges.begin(), [](const auto& a) { return std::get<1>(a); });
}

std::vector<double> johnson_cuda(typename Graph::FullGraph& full_graph, const size_t num_neurons, const bool has_negative_edges) {
    if constexpr (!CUDA_FOUND) {
        assert(false && "Tried calling CUDA function johnson_cuda, but CUDA was not found.");
        return {};
    }

    const auto E = boost::num_edges(full_graph);

    const auto weight_map = boost::get(&Graph::EdgeProperties::weight, full_graph);

    const auto [edge_begin_it, edge_end_it] = boost::edges(full_graph);

    std::vector<int> weights = johnson_get_weights_vector(edge_begin_it, edge_end_it, weight_map);
    std::vector<edge_t> cuda_edges = johnson_get_edge_vector<edge_t>(edge_begin_it, edge_end_it, E);

    johnson_cuda_sort_edges_and_weights(weights, cuda_edges);

    graph_cuda_t<std::vector<int>, std::vector<edge_t>> graph{
        static_cast<int>(num_neurons),
        static_cast<int>(E),
        johnson_cuda_generate_starts_vector(cuda_edges, num_neurons),
        std::move(weights),
        std::move(cuda_edges)
    };

    std::vector<double> distances(num_neurons * num_neurons);

    RAII_progress_status<timed_progress_status> status{
        0,
        100,
        false,
        [](auto& status) {
            return fmt::format("Johnson CUDA {}/{} Vertices", status.progress, status.total);
        }
    };
    johnson_cuda_impl(graph, distances, status.status, has_negative_edges);
    return distances;
}

std::vector<double> johnson_parallel(typename Graph::FullGraph& full_graph, const size_t num_neurons, const bool has_negative_edges) {
    const auto [edge_begin_it, edge_end_it] = boost::edges(full_graph);

    const auto E = boost::num_edges(full_graph);

    const auto weight_map = boost::get(&Graph::EdgeProperties::weight, full_graph);

    graph_t graph{
        static_cast<int>(num_neurons),
        static_cast<int>(E),
        johnson_get_weights_vector(edge_begin_it, edge_end_it, weight_map),
        johnson_get_edge_vector<APSP_Edge>(edge_begin_it, edge_end_it, E)
    };

    std::vector<double> distances(num_neurons * num_neurons);
    johnson_parallel_impl(graph, distances, has_negative_edges);
    return distances;
}

}
