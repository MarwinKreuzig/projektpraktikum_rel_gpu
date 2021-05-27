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

// Base class template
template <typename Graph>
struct Weight {
    Weight(Graph& graph, std::string version)
        : graph(graph)
        , functor_version(std::move(version)){};
    Graph& graph;
    const std::string functor_version;
    virtual double operator()(typename boost::graph_traits<Graph>::edge_descriptor edge) const = 0;
};

// Edge weight is 1/weight
template <typename Graph>
struct WeightInverse : public Weight<Graph> {
    explicit WeightInverse(Graph& graph)
        : Weight<Graph>(graph, "1/weight"){};

    double operator()(typename boost::graph_traits<Graph>::edge_descriptor edge) const final {
        return this->graph[edge].weight_inverse;
    }
};

// Edge weight is weight/max{weights}
template <typename Graph>
struct WeightDivMaxWeight : public Weight<Graph> {
    explicit WeightDivMaxWeight(Graph& graph)
        : Weight<Graph>(graph, "weight/max{weights}"){};

    double operator()(typename boost::graph_traits<Graph>::edge_descriptor edge) const final {
        return this->graph[edge].weight_div_max_weight;
    }
};

// Edge weight is 1
template <typename Graph>
struct WeightOne : public Weight<Graph> {
    explicit WeightOne(Graph& graph)
        : Weight<Graph>(graph, "weight = 1"){};

    double operator()(typename boost::graph_traits<Graph>::edge_descriptor edge) const final {
        return this->graph[edge].weight_one;
    }
};
