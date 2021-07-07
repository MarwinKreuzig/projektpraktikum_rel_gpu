#include "apsp.h"

#include "johnson.h"

namespace apsp {

std::vector<double> johnson(typename Graph::FullGraph& full_graph, const size_t num_neurons, bool use_cuda_if_available) {

    if constexpr (CUDA_FOUND) {
        if (use_cuda_if_available) {
            return johnson_cuda(full_graph, num_neurons);
        }
    }
    return johnson_parallel(full_graph, num_neurons);
}

std::vector<double> johnson_cuda(typename Graph::FullGraph& full_graph, size_t num_neurons) {
    const auto [edge_begin_it, edge_end_it] = boost::edges(full_graph);

    const auto E = boost::num_edges(full_graph);

    const auto weight_map = boost::get(&Graph::EdgeProperties::weight, full_graph);

    std::vector<int> weights{};
    std::transform(edge_begin_it, edge_end_it, std::back_inserter(weights), [&](const auto& edge) {
        return weight_map(edge);
    });

    std::vector<edge_t> cuda_edges(E);
    std::transform(edge_begin_it, edge_end_it, cuda_edges.begin(), [](const auto& edge) {
        return edge_t{ static_cast<int>(edge.m_source), static_cast<int>(edge.m_target) };
    });

    // Need to sort edges by their starting vertex id
    // Need to zip weights and edges to keep the correct weight for each edge
    std::vector<std::pair<int, edge_t>> zipped{};
    std::transform(
        weights.begin(),
        weights.end(),
        cuda_edges.begin(),
        std::back_inserter(zipped),
        [](int& weight, edge_t& edge) -> std::pair<int, edge_t> {
            return { weight, edge };
        });

    std::sort(
        zipped.begin(),
        zipped.end(),
        [](const auto& a, const auto& b) -> bool {
            if (!(std::get<1>(a).u < std::get<1>(b).u)) {
                if (std::get<1>(a).u == std::get<1>(b).u) {
                    return std::get<1>(a).v < std::get<1>(b).v;
                }
                return false;
            }
            return true;
        });

    // Unzip
    std::transform(zipped.begin(), zipped.end(), weights.begin(), [](const auto& a) { return std::get<0>(a); });
    std::transform(zipped.begin(), zipped.end(), cuda_edges.begin(), [](const auto& a) { return std::get<1>(a); });

    auto starts = std::vector<int>(num_neurons + 1, -1); // Starting point for each edge

    auto edge_it = cuda_edges.cbegin();
    int c = 0;
    starts.front() = 0;
    for (size_t i = 1u; i < starts.size(); ++i) {
        while ((*edge_it).u < static_cast<int>(i)) {
            ++edge_it;
            ++c;
        }
        if ((*edge_it).u == static_cast<int>(i)) {
            starts[i] = c;
        }
    }
    starts.back() = cuda_edges.size();

    graph_cuda_t<std::vector<int>, std::vector<edge_t>> graph{
        static_cast<int>(num_neurons),
        static_cast<int>(E),
        std::move(starts),
        std::move(weights),
        std::move(cuda_edges)
    };

    std::vector<double> distances(num_neurons * num_neurons);

    johnson_cuda_impl(graph, distances);
    return distances;
}

std::vector<double> johnson_parallel(typename Graph::FullGraph& full_graph, size_t num_neurons) {
    const auto [edge_begin_it, edge_end_it] = boost::edges(full_graph);

    const auto E = boost::num_edges(full_graph);

    const auto weight_map = boost::get(&Graph::EdgeProperties::weight, full_graph);

    std::vector<int> weights{};
    std::transform(edge_begin_it, edge_end_it, std::back_inserter(weights), [&](const auto& edge) {
        return weight_map(edge);
    });

    std::vector<APSP_Edge> edges(E);
    std::transform(edge_begin_it, edge_end_it, edges.begin(), [](const auto& edge) {
        return APSP_Edge{ static_cast<int>(edge.m_source), static_cast<int>(edge.m_target) };
    });

    graph_t graph{
        static_cast<int>(num_neurons),
        static_cast<int>(E),
        std::move(weights),
        std::move(edges)
    };

    std::vector<double> distances(num_neurons * num_neurons);

    johnson_parallel_impl(graph, distances);
    return distances;
}

}
