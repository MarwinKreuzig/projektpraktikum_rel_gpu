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

void johnson_cuda(graph_cuda_t<std::vector<int>, std::vector<edge_t>>& gr, std::vector<double>& output);

void johnson_parallel(graph_t& gr, std::vector<int>& output);

} // namespace apsp
