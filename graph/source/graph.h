#pragma once

#include "position.h"

#include <boost/graph/adjacency_list.hpp>

#include <filesystem>
#include <tuple>

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

    FullGraph full_graph;
    ConnectivityGraph conn_graph;

    std::map<Position, FullVertex, Position::less> pos_to_vtx;
    std::map<FullVertex, Position> vtx_to_pos;
    std::map<size_t, FullVertex> id_to_vtx_full;
    std::map<size_t, FullVertex> id_to_vtx_conn;

    Position offset;
};

template <typename Graph, typename Weight>
void average_clustering_coefficient(Graph& graph, const Weight& weight) {
    using namespace boost;

    typename graph_traits<Graph>::vertex_descriptor vertex_i, vertex_j, vertex_k;
    size_t total_degree_vertex_i, num_bilateral_edges_vertex_i;
    size_t num_denominator_zero, num_denominator_less_than_zero, num_denominator_greater_than_zero;
    size_t num_bilateral_edges = 0;
    size_t num_vals = 0;
    double delta, avg = 0;

    num_denominator_zero = 0;
    num_denominator_less_than_zero = 0;
    num_denominator_greater_than_zero = 0;

    // For all vertices i
    typename graph_traits<Graph>::vertex_iterator vertex_iter, vertex_iter_end;
    for (boost::tie(vertex_iter, vertex_iter_end) = vertices(graph); vertex_iter != vertex_iter_end; ++vertex_iter) {
        typename graph_traits<Graph>::adjacency_iterator adj_curr, adj_end;
        typename Graph::inv_adjacency_iterator inv_adj_curr, inv_adj_end;
        typename graph_traits<Graph>::edge_descriptor edge;
        typename std::set<typename graph_traits<Graph>::vertex_descriptor> neighbors_of_vertex_i, neighbors_of_vertex_j;
        typename std::set<typename graph_traits<Graph>::vertex_descriptor>::iterator neighbors_of_vertex_i_iter;
        double clustering_coefficient_vertex_i, numerator_clustering_coefficient_vertex_i = 0;
        size_t num_bilateral_edges_vertex_i = 0;
        bool found;

        vertex_i = *vertex_iter;

        // Total degree (in + out) of vertex i
        total_degree_vertex_i = out_degree(vertex_i, graph) + in_degree(vertex_i, graph);
        //std::cout << "total degree: " << total_degree_vertex_i << std::endl;

        // Number of bilateral edges between i and its neighbors j
        for (boost::tie(adj_curr, adj_end) = adjacent_vertices(vertex_i, graph); adj_curr != adj_end; ++adj_curr) {
            vertex_j = *adj_curr;

            boost::tie(edge, found) = boost::edge(vertex_j, vertex_i, graph);

            if (found) {
                num_bilateral_edges_vertex_i++;
                num_bilateral_edges++;
            }
        }
        //std::cout << "num bilateral edges: " << num_bilateral_edges_vertex_i << std::endl;

        // Gather all neighbors of vertex i (in and out neighbors)
        for (boost::tie(adj_curr, adj_end) = adjacent_vertices(vertex_i, graph); adj_curr != adj_end; ++adj_curr)
            neighbors_of_vertex_i.insert(*adj_curr);
        for (boost::tie(inv_adj_curr, inv_adj_end) = inv_adjacent_vertices(vertex_i, graph); inv_adj_curr != inv_adj_end; ++inv_adj_curr)
            neighbors_of_vertex_i.insert(*inv_adj_curr);

        for (neighbors_of_vertex_i_iter = neighbors_of_vertex_i.begin();
             neighbors_of_vertex_i_iter != neighbors_of_vertex_i.end();
             ++neighbors_of_vertex_i_iter) {
            typename std::set<typename graph_traits<Graph>::vertex_descriptor> neighbors_of_vertex_j;
            typename std::set<typename graph_traits<Graph>::vertex_descriptor>::iterator neighbors_of_vertex_j_iter;

            vertex_j = *neighbors_of_vertex_i_iter;

            // Gather all neighbors of vertex j
            for (boost::tie(adj_curr, adj_end) = adjacent_vertices(vertex_j, graph); adj_curr != adj_end; ++adj_curr)
                neighbors_of_vertex_j.insert(*adj_curr);
            for (boost::tie(inv_adj_curr, inv_adj_end) = inv_adjacent_vertices(vertex_j, graph); inv_adj_curr != inv_adj_end; ++inv_adj_curr)
                neighbors_of_vertex_j.insert(*inv_adj_curr);

            for (neighbors_of_vertex_j_iter = neighbors_of_vertex_j.begin();
                 neighbors_of_vertex_j_iter != neighbors_of_vertex_j.end();
                 ++neighbors_of_vertex_j_iter) {
                vertex_k = *neighbors_of_vertex_j_iter;

                if ((vertex_i != vertex_j) && (vertex_j != vertex_k) && (vertex_i != vertex_k)) {
                    double weight_ij, weight_ji, weight_jk, weight_kj, weight_ik, weight_ki;

                    boost::tie(edge, found) = boost::edge(vertex_i, vertex_j, graph);
                    weight_ij = found ? weight(edge) : 0;

                    boost::tie(edge, found) = boost::edge(vertex_j, vertex_i, graph);
                    weight_ji = found ? weight(edge) : 0;

                    boost::tie(edge, found) = boost::edge(vertex_j, vertex_k, graph);
                    weight_jk = found ? weight(edge) : 0;

                    boost::tie(edge, found) = boost::edge(vertex_k, vertex_j, graph);
                    weight_kj = found ? weight(edge) : 0;

                    boost::tie(edge, found) = boost::edge(vertex_i, vertex_k, graph);
                    weight_ik = found ? weight(edge) : 0;

                    boost::tie(edge, found) = boost::edge(vertex_k, vertex_i, graph);
                    weight_ki = found ? weight(edge) : 0;

                    double exponent = ((double)1) / 3;
                    numerator_clustering_coefficient_vertex_i += (pow(weight_ij, exponent) + pow(weight_ji, exponent)) * (pow(weight_jk, exponent) + pow(weight_kj, exponent)) * (pow(weight_ik, exponent) + pow(weight_ki, exponent));
                }
            } // for all k
        } // for all j
        size_t denominator_clustering_coefficient_vertex_i = 2 * (total_degree_vertex_i * (total_degree_vertex_i - 1) - 2 * num_bilateral_edges_vertex_i);

        if (0 > denominator_clustering_coefficient_vertex_i)
            num_denominator_less_than_zero++;
        else if (0 == denominator_clustering_coefficient_vertex_i)
            num_denominator_zero++;
        else if (0 < denominator_clustering_coefficient_vertex_i)
            num_denominator_greater_than_zero++;

        clustering_coefficient_vertex_i = numerator_clustering_coefficient_vertex_i / denominator_clustering_coefficient_vertex_i;

        // Include in average clustering coefficient
        num_vals++;
        delta = clustering_coefficient_vertex_i - avg;
        avg += delta / num_vals;

    } // for all i

    //std::cout << "[" << wall_clock_time() << "] " << "Average clustering coefficient (" << weight.functor_version << "): " << avg << std::endl;
    //std::cout << "[" << wall_clock_time() << "] " << "    Number denominators == 0: " << num_denominator_zero << std::endl;
    //std::cout << "[" << wall_clock_time() << "] " << "    Number denominators <  0: " << num_denominator_less_than_zero << std::endl;
    //std::cout << "[" << wall_clock_time() << "] " << "    Number denominators >  0: " << num_denominator_greater_than_zero << std::endl;
    //std::cout << "[" << wall_clock_time() << "] " << "    Number bilateral edges  : " << num_bilateral_edges / 2 << std::endl;
}

template <typename Graph>
void average_clustering_coefficient_unweighted_undirected(Graph& graph) {
    using namespace boost;

    typename graph_traits<Graph>::vertex_descriptor vertex_i, vertex_j, vertex_k;
    typename graph_traits<Graph>::vertex_iterator vertex_iter, vertex_iter_end;
    size_t num_denominator_zero, num_denominator_less_than_zero, num_denominator_greater_than_zero;
    size_t num_vals = 0;
    double delta, avg = 0;

    num_denominator_zero = 0;
    num_denominator_less_than_zero = 0;
    num_denominator_greater_than_zero = 0;

    // For all vertices i
    for (boost::tie(vertex_iter, vertex_iter_end) = vertices(graph); vertex_iter != vertex_iter_end; ++vertex_iter) {
        typename graph_traits<Graph>::adjacency_iterator adj_curr, adj_end;
        typename Graph::inv_adjacency_iterator inv_adj_curr, inv_adj_end;
        typename std::set<typename graph_traits<Graph>::vertex_descriptor>::iterator iter_j, iter_k;
        typename std::set<typename graph_traits<Graph>::vertex_descriptor> neighbors_of_vertex_i;
        size_t max_num_triangles_of_vertex_i, num_triangles_of_vertex_i, num_neighbors_of_vertex_i;
        double clustering_coefficient_vertex_i;
        bool found_jk, found_kj;

        num_triangles_of_vertex_i = 0;
        num_neighbors_of_vertex_i = 0;

        vertex_i = *vertex_iter;

        // Gather all neighbors of vertex i (in and out neighbors)
        for (boost::tie(adj_curr, adj_end) = adjacent_vertices(vertex_i, graph); adj_curr != adj_end; ++adj_curr)
            neighbors_of_vertex_i.insert(*adj_curr);
        for (boost::tie(inv_adj_curr, inv_adj_end) = inv_adjacent_vertices(vertex_i, graph); inv_adj_curr != inv_adj_end; ++inv_adj_curr)
            neighbors_of_vertex_i.insert(*inv_adj_curr);

        num_neighbors_of_vertex_i = neighbors_of_vertex_i.size();

        for (iter_j = neighbors_of_vertex_i.begin(); iter_j != neighbors_of_vertex_i.end(); ++iter_j) {
            for (iter_k = std::next(iter_j); iter_k != neighbors_of_vertex_i.end(); ++iter_k) {
                vertex_j = *iter_j;
                vertex_k = *iter_k;

                std::tie(std::ignore, found_jk) = boost::edge(vertex_j, vertex_k, graph);
                std::tie(std::ignore, found_kj) = boost::edge(vertex_k, vertex_j, graph);

                if (found_jk || found_kj) {
                    num_triangles_of_vertex_i++;
                }
            }
        }

        max_num_triangles_of_vertex_i = (num_neighbors_of_vertex_i * (num_neighbors_of_vertex_i - 1)) / 2;

        if (0 > max_num_triangles_of_vertex_i)
            num_denominator_less_than_zero++;
        else if (0 == max_num_triangles_of_vertex_i)
            num_denominator_zero++;
        else if (0 < max_num_triangles_of_vertex_i)
            num_denominator_greater_than_zero++;

        clustering_coefficient_vertex_i = num_triangles_of_vertex_i / (double)max_num_triangles_of_vertex_i;

        // Include in average clustering coefficient
        num_vals++;
        delta = clustering_coefficient_vertex_i - avg;
        avg += delta / num_vals;
    } // for all i

    //std::cout << "[" << wall_clock_time() << "] " << "Average clustering coefficient (unweighted, undirected): " << avg << std::endl;
    //std::cout << "[" << wall_clock_time() << "] " << "    Number denominators == 0: " << num_denominator_zero << std::endl;
    //std::cout << "[" << wall_clock_time() << "] " << "    Number denominators <  0: " << num_denominator_less_than_zero << std::endl;
    //std::cout << "[" << wall_clock_time() << "] " << "    Number denominators >  0: " << num_denominator_greater_than_zero << std::endl;
}

// Base class template
template <typename Graph>
struct Weight {
    Weight(Graph& graph, const std::string& version)
        : graph(graph)
        , functor_version(version){};
    Graph& graph;
    const std::string functor_version;
};

// Edge weight is 1/weight
template <typename Graph>
struct WeightInverse : public Weight<Graph> {
    WeightInverse(Graph& graph)
        : Weight<Graph>(graph, "1/weight"){};

    double operator()(typename boost::graph_traits<Graph>::edge_descriptor edge) const {
        return this->graph[edge].weight_inverse;
    }
};

// Edge weight is weight/max{weights}
template <typename Graph>
struct WeightDivMaxWeight : public Weight<Graph> {
    WeightDivMaxWeight(Graph& graph)
        : Weight<Graph>(graph, "weight/max{weights}"){};

    double operator()(typename boost::graph_traits<Graph>::edge_descriptor edge) const {
        return this->graph[edge].weight_div_max_weight;
    }
};

// Edge weight is 1
template <typename Graph>
struct WeightOne : public Weight<Graph> {
    WeightOne(Graph& graph)
        : Weight<Graph>(graph, "weight = 1"){};

    double operator()(typename boost::graph_traits<Graph>::edge_descriptor edge) const {
        return this->graph[edge].weight_one;
    }
};
