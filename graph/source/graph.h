#pragma once

#include "position.h"

#include <boost/graph/adjacency_list.hpp>

#include <filesystem>
#include <tuple>
#include <utility>

class Graph {
public:
    // FullVertex properties (bundled properties)
    struct VertexProperties {
        std::string name;
        Position pos;
    };

    struct EdgeProperties {
        double weight;
        double weight_inverse;
        double weight_div_max_weight;
        double weight_one;
    };

    using FullGraph = boost::adjacency_list<
        boost::vecS, // OutEdgeList
        boost::vecS, // VertexList
        boost::bidirectionalS, // Bidirectional
        VertexProperties, // VertexProperties
        EdgeProperties // EdgeProperties
        >;

    using ConnectivityGraph = boost::adjacency_list<
        boost::vecS, // OutEdgeList
        boost::vecS, // VertexList
        boost::undirectedS // Unidirectional
        >;

    using FullVertex = boost::graph_traits<FullGraph>::vertex_descriptor;
    using FullVertexIterator = boost::graph_traits<FullGraph>::vertex_iterator;

    using FullEdge = boost::graph_traits<FullGraph>::edge_descriptor;
    using FullEdgeIterator = boost::graph_traits<FullGraph>::edge_iterator;

    using ConnectivityVertex = boost::graph_traits<ConnectivityGraph>::vertex_descriptor;
    using ConnectivityVertexIterator = boost::graph_traits<ConnectivityGraph>::vertex_iterator;

    using ConnectivityEdge = boost::graph_traits<ConnectivityGraph>::edge_descriptor;
    using ConnectivityEdgeIterator = boost::graph_traits<ConnectivityGraph>::edge_iterator;

    void add_vertices_from_file(const std::filesystem::path& file_path);

    void add_edges_from_file(const std::filesystem::path& file_path);

    void print_vertices(std::ostream& os);

    void print_edges(std::ostream& os);

    void calculate_metrics(std::ostream& os);

    std::tuple<double, double, double> smallest_coordinate_per_dimension();

    void add_offset_to_positions(const Position& offset);

    std::pair<int, int> min_max_degree();

    size_t get_num_vertices();

    size_t get_num_edges();

private:
    void init_edge_weight();

    double calculate_average_euclidean_distance();

    std::tuple<double, double> calculate_all_pairs_shortest_paths();

    double calculate_average_betweenness_centrality();

    double calculate_clustering_coefficient();

    void add_vertex(const Position& pos, const std::string& name, size_t id);

    void add_edge(size_t src_id, size_t dst_id, int weight);

    void print_vertex(FullVertex v, std::ostream& os);

    void print_edge(FullEdge e, std::ostream& os);

    FullGraph full_graph{};
    ConnectivityGraph conn_graph{};

    std::map<Position, FullVertex, Position::less> pos_to_vtx{};
    std::map<FullVertex, Position> vtx_to_pos{};
    std::map<size_t, FullVertex> id_to_vtx_full{};
    std::map<size_t, FullVertex> id_to_vtx_conn{};

    Position offset{};
};

template <typename Graph, typename Weight>
void average_clustering_coefficient(Graph& graph, const Weight& weight) {
    size_t num_denominator_zero = 0;
    size_t num_denominator_less_than_zero = 0;
    size_t num_denominator_greater_than_zero = 0;
    size_t num_bilateral_edges = 0;
    size_t num_vals = 0;
    double avg = 0;

    // For all vertices i
    for (auto [vertex_iter, vertex_iter_end] = vertices(graph); vertex_iter != vertex_iter_end; ++vertex_iter) {
        std::set<typename boost::graph_traits<Graph>::vertex_descriptor> neighbors_of_vertex_i{};
        typename std::set<typename boost::graph_traits<Graph>::vertex_descriptor>::iterator neighbors_of_vertex_i_iter;
        double numerator_clustering_coefficient_vertex_i = 0;
        size_t num_bilateral_edges_vertex_i = 0;

        const auto vertex_i = *vertex_iter;

        // Total degree (in + out) of vertex i
        const auto total_degree_vertex_i = out_degree(vertex_i, graph) + in_degree(vertex_i, graph);
        //std::cout << "total degree: " << total_degree_vertex_i << std::endl;

        // Number of bilateral edges between i and its neighbors j
        for (auto [adj_curr, adj_end] = adjacent_vertices(vertex_i, graph); adj_curr != adj_end; ++adj_curr) {
            const auto vertex_j = *adj_curr;
            if (auto found = boost::edge(vertex_j, vertex_i, graph).second; found) {
                num_bilateral_edges_vertex_i++;
                num_bilateral_edges++;
            }
        }
        //std::cout << "num bilateral edges: " << num_bilateral_edges_vertex_i << std::endl;

        // Gather all neighbors of vertex i (in and out neighbors)
        for (auto [adj_curr, adj_end] = adjacent_vertices(vertex_i, graph); adj_curr != adj_end; ++adj_curr) {
            neighbors_of_vertex_i.insert(*adj_curr);
        }
        for (auto [inv_adj_curr, inv_adj_end] = inv_adjacent_vertices(vertex_i, graph); inv_adj_curr != inv_adj_end; ++inv_adj_curr) {
            neighbors_of_vertex_i.insert(*inv_adj_curr);
        }

        for (const auto& vertex_j : neighbors_of_vertex_i) {
            std::set<typename boost::graph_traits<Graph>::vertex_descriptor> neighbors_of_vertex_j{};

            // Gather all neighbors of vertex j
            for (auto [adj_curr, adj_end] = adjacent_vertices(vertex_j, graph); adj_curr != adj_end; ++adj_curr) {
                neighbors_of_vertex_j.insert(*adj_curr);
            }
            for (auto [inv_adj_curr, inv_adj_end] = inv_adjacent_vertices(vertex_j, graph); inv_adj_curr != inv_adj_end; ++inv_adj_curr) {
                neighbors_of_vertex_j.insert(*inv_adj_curr);
            }

            for (const auto vertex_k : neighbors_of_vertex_j) {
                if ((vertex_i != vertex_j) && (vertex_j != vertex_k) && (vertex_i != vertex_k)) {
                    auto [edge, found] = boost::edge(vertex_i, vertex_j, graph);
                    const double weight_ij = found ? weight(edge) : 0;

                    std::tie(edge, found) = boost::edge(vertex_j, vertex_i, graph);
                    const double weight_ji = found ? weight(edge) : 0;

                    std::tie(edge, found) = boost::edge(vertex_j, vertex_k, graph);
                    const double weight_jk = found ? weight(edge) : 0;

                    std::tie(edge, found) = boost::edge(vertex_k, vertex_j, graph);
                    const double weight_kj = found ? weight(edge) : 0;

                    std::tie(edge, found) = boost::edge(vertex_i, vertex_k, graph);
                    const double weight_ik = found ? weight(edge) : 0;

                    std::tie(edge, found) = boost::edge(vertex_k, vertex_i, graph);
                    const double weight_ki = found ? weight(edge) : 0;

                    const double exponent = static_cast<double>(1) / 3;
                    numerator_clustering_coefficient_vertex_i += (pow(weight_ij, exponent) + pow(weight_ji, exponent)) * (pow(weight_jk, exponent) + pow(weight_kj, exponent)) * (pow(weight_ik, exponent) + pow(weight_ki, exponent));
                }
            } // for all k
        } // for all j
        const size_t denominator_clustering_coefficient_vertex_i = 2 * (total_degree_vertex_i * (total_degree_vertex_i - 1) - 2 * num_bilateral_edges_vertex_i);

        if (0 > denominator_clustering_coefficient_vertex_i) {
            num_denominator_less_than_zero++;
        } else if (0 == denominator_clustering_coefficient_vertex_i) {
            num_denominator_zero++;
        } else if (0 < denominator_clustering_coefficient_vertex_i) {
            num_denominator_greater_than_zero++;
        }

        const auto clustering_coefficient_vertex_i = numerator_clustering_coefficient_vertex_i / denominator_clustering_coefficient_vertex_i;

        // Include in average clustering coefficient
        num_vals++;
        const auto delta = clustering_coefficient_vertex_i - avg;
        avg += delta / static_cast<double>(num_vals);

    } // for all i

    //std::cout << "[" << wall_clock_time() << "] " << "Average clustering coefficient (" << weight.functor_version << "): " << avg << std::endl;
    //std::cout << "[" << wall_clock_time() << "] " << "    Number denominators == 0: " << num_denominator_zero << std::endl;
    //std::cout << "[" << wall_clock_time() << "] " << "    Number denominators <  0: " << num_denominator_less_than_zero << std::endl;
    //std::cout << "[" << wall_clock_time() << "] " << "    Number denominators >  0: " << num_denominator_greater_than_zero << std::endl;
    //std::cout << "[" << wall_clock_time() << "] " << "    Number bilateral edges  : " << num_bilateral_edges / 2 << std::endl;
}

template <typename Graph>
void average_clustering_coefficient_unweighted_undirected(Graph& graph) {
    size_t num_denominator_zero = 0;
    size_t num_denominator_less_than_zero = 0;
    size_t num_denominator_greater_than_zero = 0;
    size_t num_vals = 0;
    double avg = 0.0;

    // For all vertices i
    for (auto [vertex_iter, vertex_iter_end] = vertices(graph); vertex_iter != vertex_iter_end; ++vertex_iter) {
        std::set<typename boost::graph_traits<Graph>::vertex_descriptor> neighbors_of_vertex_i{};

        const auto vertex_i = *vertex_iter;

        // Gather all neighbors of vertex i (in and out neighbors)
        for (auto [adj_curr, adj_end] = adjacent_vertices(vertex_i, graph); adj_curr != adj_end; ++adj_curr) {
            neighbors_of_vertex_i.insert(*adj_curr);
        }
        for (auto [inv_adj_curr, inv_adj_end] = inv_adjacent_vertices(vertex_i, graph); inv_adj_curr != inv_adj_end; ++inv_adj_curr) {
            neighbors_of_vertex_i.insert(*inv_adj_curr);
        }

        size_t num_triangles_of_vertex_i = 0;
        for (auto iter_j = neighbors_of_vertex_i.begin(); iter_j != neighbors_of_vertex_i.end(); ++iter_j) {
            for (auto iter_k = std::next(iter_j); iter_k != neighbors_of_vertex_i.end(); ++iter_k) {
                const auto vertex_j = *iter_j;
                const auto vertex_k = *iter_k;

                const auto found_jk = boost::edge(vertex_j, vertex_k, graph).second;
                const auto found_kj = boost::edge(vertex_k, vertex_j, graph).second;

                if (found_jk || found_kj) {
                    num_triangles_of_vertex_i++;
                }
            }
        }

        const size_t num_neighbors_of_vertex_i = neighbors_of_vertex_i.size();
        const size_t max_num_triangles_of_vertex_i = (num_neighbors_of_vertex_i * (num_neighbors_of_vertex_i - 1)) / 2;

        if (0 > max_num_triangles_of_vertex_i) {
            num_denominator_less_than_zero++;
        } else if (0 == max_num_triangles_of_vertex_i) {
            num_denominator_zero++;
        } else {
            num_denominator_greater_than_zero++;
        }

        const double clustering_coefficient_vertex_i = static_cast<double>(num_triangles_of_vertex_i) / static_cast<double>(max_num_triangles_of_vertex_i);

        // Include in average clustering coefficient
        num_vals++;
        const auto delta = clustering_coefficient_vertex_i - avg;
        avg += delta / static_cast<double>(num_vals);
    } // for all i

    //std::cout << "[" << wall_clock_time() << "] " << "Average clustering coefficient (unweighted, undirected): " << avg << std::endl;
    //std::cout << "[" << wall_clock_time() << "] " << "    Number denominators == 0: " << num_denominator_zero << std::endl;
    //std::cout << "[" << wall_clock_time() << "] " << "    Number denominators <  0: " << num_denominator_less_than_zero << std::endl;
    //std::cout << "[" << wall_clock_time() << "] " << "    Number denominators >  0: " << num_denominator_greater_than_zero << std::endl;
}

// Base class template
template <typename Graph>
struct Weight {
    Weight(Graph& graph, std::string version)
        : graph(graph)
        , functor_version(std::move(version)){};
    Graph& graph;
    const std::string functor_version;
};

// Edge weight is 1/weight
template <typename Graph>
struct WeightInverse : public Weight<Graph> {
    explicit WeightInverse(Graph& graph)
        : Weight<Graph>(graph, "1/weight"){};

    double operator()(typename boost::graph_traits<Graph>::edge_descriptor edge) const {
        return this->graph[edge].weight_inverse;
    }
};

// Edge weight is weight/max{weights}
template <typename Graph>
struct WeightDivMaxWeight : public Weight<Graph> {
    explicit WeightDivMaxWeight(Graph& graph)
        : Weight<Graph>(graph, "weight/max{weights}"){};

    double operator()(typename boost::graph_traits<Graph>::edge_descriptor edge) const {
        return this->graph[edge].weight_div_max_weight;
    }
};

// Edge weight is 1
template <typename Graph>
struct WeightOne : public Weight<Graph> {
    explicit WeightOne(Graph& graph)
        : Weight<Graph>(graph, "weight = 1"){};

    double operator()(typename boost::graph_traits<Graph>::edge_descriptor edge) const {
        return this->graph[edge].weight_one;
    }
};
