#ifndef graph_johnson_h
#define graph_johnson_h

#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>

#include "../graph.h"

namespace apsp {

using APSP_Graph = boost::adjacency_list<boost::listS, boost::vecS, boost::directedS, boost::no_property, boost::property<boost::edge_weight_t, int>>;
using APSP_Vertex = boost::graph_traits<APSP_Graph>::vertex_descriptor;
using APSP_Edge = std::pair<int, int>;

struct graph_t {
    graph_t(int V_, int E_, std::vector<int> weights_, std::vector<APSP_Edge> edge_array_)
        : V{ V_ }
        , E{ E_ }
        , weights{ std::move(weights_) }
        , edge_array{ std::move(edge_array_) } { }

    graph_t(int V_, int E_)
        : graph_t(V_, E_, std::vector<int>(E_), std::vector<APSP_Edge>(E_)) { }

    int V; // NOLINT(misc-non-private-member-variables-in-classes)
    int E; // NOLINT(misc-non-private-member-variables-in-classes)
    std::vector<int> weights; // NOLINT(misc-non-private-member-variables-in-classes)
    std::vector<APSP_Edge> edge_array; // NOLINT(misc-non-private-member-variables-in-classes)
};

struct edge_t {
    int u;
    int v;
};

template <typename T, typename U>
struct graph_cuda_t {
    static_assert(std::is_same_v<int, typename T::value_type>, "value_type of T should be int");
    static_assert(std::is_same_v<edge_t, typename U::value_type>, "value_type of U should be edge_t");
    int V;
    int E;
    T starts;
    T weights;
    U edge_array;
};

#if CUDA_FOUND
void johnson_cuda(graph_cuda_t<std::vector<int>, std::vector<edge_t>>& gr, std::vector<double>& output);
#else
inline void johnson_cuda(graph_cuda_t<std::vector<int>, std::vector<edge_t>>& /* gr */, std::vector<double>& /* output */) { }
#endif

void johnson_parallel(graph_t& gr, std::vector<double>& output);

inline std::vector<double> johnson(typename Graph::FullGraph full_graph, const size_t num_neurons, bool use_cuda_if_available = true) {
    const auto [edge_begin_it, edge_end_it] = boost::edges(full_graph);

    const auto E = boost::num_edges(full_graph);

    const auto weight_map = boost::get(&Graph::EdgeProperties::weight, full_graph);

    std::vector<int> weights{};
    std::transform(edge_begin_it, edge_end_it, std::back_inserter(weights), [&](const auto& edge) {
        return weight_map(edge);
    });

    if constexpr (CUDA_FOUND) {
        if (use_cuda_if_available) {
            std::vector<edge_t> cuda_edges(E);
            std::transform(edge_begin_it, edge_end_it, cuda_edges.begin(), [](const auto& edge) {
                return edge_t{ static_cast<int>(edge.m_source), static_cast<int>(edge.m_target) };
            });

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

            std::transform(zipped.begin(), zipped.end(), weights.begin(), [](const auto& a) { return std::get<0>(a); });
            std::transform(zipped.begin(), zipped.end(), cuda_edges.begin(), [](const auto& a) { return std::get<1>(a); });

            auto starts = std::vector<int>(num_neurons + 1); // Starting point for each edge

            auto edge_it = cuda_edges.cbegin();
            int c = 0;
            starts.front() = 0;
            for (auto i = 1u; i < starts.size(); ++i) {
                const auto& my_edge = *edge_it;
                for (; my_edge.u == edge_it->u; ++edge_it, ++c) { }
                starts[i] = c;
            }

            graph_cuda_t<std::vector<int>, std::vector<edge_t>> graph{
                static_cast<int>(num_neurons),
                static_cast<int>(E),
                std::move(starts),
                std::move(weights),
                std::move(cuda_edges)
            };

            std::vector<double> distances(num_neurons * num_neurons);

            johnson_cuda(graph, distances);
            return distances;
        }
    }

    std::vector<APSP_Edge> edges(E);
    std::transform(edge_begin_it, edge_end_it, edges.begin(), [](const auto& edge) {
        return APSP_Edge{ static_cast<int>(edge.m_source), static_cast<int>(edge.m_target) };
    });

    auto edge_array = std::vector<edge_t>(E);

    graph_t graph{
        static_cast<int>(num_neurons),
        static_cast<int>(E),
        std::move(weights),
        std::move(edges)
    };

    std::vector<double> distances(num_neurons * num_neurons);

    johnson_parallel(graph, distances);
    return distances;
}

} // namespace apsp

#endif
