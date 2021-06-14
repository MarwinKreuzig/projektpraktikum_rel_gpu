#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>

#include <type_traits>
#include <utility>
#include <vector>

namespace apsp {

using Graph = boost::adjacency_list<boost::listS, boost::vecS, boost::directedS, boost::no_property, boost::property<boost::edge_weight_t, int>>;
using Vertex = boost::graph_traits<Graph>::vertex_descriptor;
using Edge = std::pair<int, int>;

struct graph_t {
    graph_t(int V_, int E_, std::vector<Edge> edge_array_, std::vector<int> weights_)
        : V{ V_ }
        , E{ E_ }
        , weights{ std::move(weights_) }
        , edge_array{ std::move(edge_array_) } { }

    graph_t(int V_, int E_)
        : graph_t(V_, E_, std::vector<Edge>(E_), std::vector<int>(E_)) { }

    int V; // NOLINT(misc-non-private-member-variables-in-classes)
    int E; // NOLINT(misc-non-private-member-variables-in-classes)
    std::vector<int> weights; // NOLINT(misc-non-private-member-variables-in-classes)
    std::vector<Edge> edge_array; // NOLINT(misc-non-private-member-variables-in-classes)
};

struct edge_t {
    int u;
    int v;
};

template <typename T, typename U>
struct graph_cuda_t {
    static_assert(std::is_same_v<int, typename T::value_type>, "value_type of T should be int");
    static_assert(std::is_same_v<edge_t, typename U::value_type>, "value_type of U should be int");
    int V;
    int E;
    T starts;
    T weights;
    U edge_array;
};

void johnson_cuda(graph_cuda_t<std::vector<int>, std::vector<edge_t>>& gr, std::vector<double>& output);

void johnson_parallel(graph_t& gr, std::vector<int>& output);

} // namespace apsp
