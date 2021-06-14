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

#if USE_CUDA
void johnson_cuda(graph_cuda_t<std::vector<int>, std::vector<edge_t>>& gr, std::vector<double>& output);
#else
inline void johnson_cuda(graph_cuda_t<std::vector<int>, std::vector<edge_t>>& /* gr */, std::vector<double>& /* output */) { }
#endif

void johnson_parallel(graph_t& gr, std::vector<double>& output);

inline std::vector<double> johnson(typename Graph::FullGraph full_graph, const size_t num_neurons) {
    const auto [edge_begin_it, edge_end_it] = boost::edges(full_graph);

    const auto E = boost::num_edges(full_graph);

    const auto weight_map = boost::get(&Graph::EdgeProperties::weight, full_graph);

    std::vector<int> weights{};
    std::transform(edge_begin_it, edge_end_it, std::back_inserter(weights), [&](const auto& edge) {
        return weight_map(edge);
    });

    if constexpr (USE_CUDA) {
        std::vector<edge_t> cuda_edges(E);
        std::transform(edge_begin_it, edge_end_it, cuda_edges.begin(), [](const auto& edge) {
            return edge_t{ static_cast<int>(edge.m_source), static_cast<int>(edge.m_target) };
        });

        auto edge_array = std::vector<edge_t>(E);
        auto starts = std::vector<int>(num_neurons + 1); // Starting point for each edge
        std::iota(starts.begin(), starts.end(), 0);

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
    } else {
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
}

} // namespace apsp
