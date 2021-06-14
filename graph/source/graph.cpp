#include "graph.h"

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/betweenness_centrality.hpp>
#include <boost/graph/clustering_coefficient.hpp>
#include <boost/graph/exterior_property.hpp>
#include <boost/graph/johnson_all_pairs_shortest.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/variant/get.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "apsp/johnson.hpp"

static void average_clustering_coefficient(Graph::FullGraph& graph, const Weight<Graph::FullGraph>& weight);
static void average_clustering_coefficient_unweighted_undirected(typename Graph::FullGraph& graph);

void Graph::add_vertices_from_file(const std::filesystem::path& file_path) {
    std::ifstream file(file_path);

    std::string line{};

    while (std::getline(file, line)) {
        if (line[0] == '#') {
            continue;
        }

        std::stringstream sstream(line);

        size_t id{};
        Position pos{};
        std::string area{};
        std::string type{};

        bool success = (sstream >> id) && (sstream >> pos.x) && (sstream >> pos.y) && (sstream >> pos.z) && (sstream >> area) && (sstream >> type);

        if (!success) {
            continue;
        }

        add_vertex(pos, area.append(" ").append(type), id);
    }
}

void Graph::add_edges_from_file(const std::filesystem::path& file_path) {
    std::ifstream file(file_path);

    std::string line{};

    while (std::getline(file, line)) {
        if (line[0] == '#') {
            continue;
        }

        std::stringstream sstream(line);

        size_t src_id{};
        size_t dst_id{};
        int weight{};

        bool success = (sstream >> dst_id) && (sstream >> src_id) && (sstream >> weight);

        if (!success) {
            continue;
        }

        const int weight_in_boost = std::abs(weight);

        const FullVertex dst_vtx = id_to_vtx_full[dst_id];
        const FullVertex src_vtx = id_to_vtx_full[src_id];

        FullEdge edge{};
        std::tie(edge, success) = boost::edge(src_vtx, dst_vtx, full_graph);

        if (!success) {
            boost::add_edge(src_vtx, dst_vtx, full_graph);
            std::tie(edge, success) = boost::edge(src_vtx, dst_vtx, full_graph);
            full_graph[edge].weight = weight_in_boost;
        } else {
            std::cerr << "FullEdge already in full_graph\n";
            full_graph[edge].weight += weight_in_boost;
            continue;
        }
    }
}

void Graph::print_vertices(std::ostream& os) {
    os << "# Vertices: " << boost::num_vertices(full_graph) << "\n";
    os << "# Add offset (x y z) to get original positions: (" << -offset.x << " " << -offset.y << " " << -offset.z << ")\n";
    os << "# Position (x y z)\tArea"
       << "\n";

    for (auto [it, it_end] = boost::vertices(full_graph); it != it_end; ++it) {
        print_vertex(*it, os);
    }
}

void Graph::print_edges(std::ostream& os) {
    os << "# <src pos x> <src pos y> <src pos z>  "
       << "<tgt pos x> <tgt pos y> <tgt pos z>"
       << "\n";

    for (auto [it, it_end] = boost::edges(full_graph); it != it_end; ++it) {
        print_edge(*it, os);
    }
}

void Graph::calculate_metrics(std::ostream& os) {
    os << "Calculating different metrics\n";

    os << "There are " << get_num_vertices() << " many vertices and " << get_num_edges() << " many edges in the graph\n";
    const auto [min, max] = min_max_degree();
    os << "The minimum number of edges is: " << min << " and the maximum is: " << max << "\n";

    os << "Calculating average euclidean distance...\n";
    const double avg_eucl_dist = calculate_average_euclidean_distance();
    os << "It was: " << avg_eucl_dist << "\n";

    os << "Calculating all pairs shortest paths...\n";
    const auto [avg, glob_eff] = calculate_all_pairs_shortest_paths();
    os << "Average shortest path was: " << avg << "\n";
    os << "Global efficiency was: " << glob_eff << "\n";

    os << "Calculating average betweenness centrality...\n";
    const double avg_betw_cent = calculate_average_betweenness_centrality();
    os << "It was: " << avg_betw_cent << "\n";

    os << "Calculating clustering coefficient...\n";
    const double clust_coeff = calculate_clustering_coefficient();
    os << "It was: " << clust_coeff << "\n";
}

std::tuple<double, double, double> Graph::smallest_coordinate_per_dimension() {
    constexpr const double max_double = std::numeric_limits<double>::max();
    Position min_coords(max_double, max_double, max_double);

    for (auto [it_vtx, it_vtx_end] = boost::vertices(full_graph); it_vtx != it_vtx_end; ++it_vtx) {
        min_coords.MinForEachCoordinate(full_graph[*it_vtx].pos);
    }

    return std::make_tuple(min_coords.x, min_coords.y, min_coords.z);
}

void Graph::add_offset_to_positions(const Position& offset) {
    this->offset.Add(offset); // Update offset

    for (auto [it_vtx, it_vtx_end] = boost::vertices(full_graph); it_vtx != it_vtx_end; ++it_vtx) {
        full_graph[*it_vtx].pos.Add(offset);
    }
}

std::pair<int, int> Graph::min_max_degree() {
    int max_deg = 0;
    int min_deg = std::numeric_limits<int>::max();

    for (auto [it_vtx, it_vtx_end] = boost::vertices(full_graph); it_vtx != it_vtx_end; ++it_vtx) {
        max_deg = std::max(max_deg, static_cast<int>(boost::out_degree(*it_vtx, full_graph) + boost::in_degree(*it_vtx, full_graph)));
        min_deg = std::min(min_deg, static_cast<int>(boost::out_degree(*it_vtx, full_graph) + boost::in_degree(*it_vtx, full_graph)));
    }

    return std::make_pair(min_deg, max_deg);
}

size_t Graph::get_num_vertices() {
    return boost::num_vertices(full_graph);
}

size_t Graph::get_num_edges() {
    return boost::num_edges(full_graph);
}

void Graph::init_edge_weight() {
    double max_weight = std::numeric_limits<double>::min();
    double min_weight = std::numeric_limits<double>::max();

    for (auto [current, end] = boost::edges(full_graph); current != end; ++current) {
        auto& current_edge = full_graph[*current];
        const auto weight = current_edge.weight;

        min_weight = std::min(weight, min_weight);
        max_weight = std::max(weight, max_weight);

        current_edge.weight_inverse = 1.0 / weight;
        current_edge.weight_div_max_weight = weight / max_weight;
        current_edge.weight_one = 1.0;
    }
}

double Graph::calculate_average_euclidean_distance() {
    double avg_eucl_dist = 0.0;
    double sum_weights = 0.0;

    for (auto [current, end] = boost::edges(full_graph); current != end; ++current) {
        const auto current_prop = *current;
        auto& current_edge = full_graph[current_prop];
        const auto weight = current_edge.weight;

        const auto src = boost::source(current_prop, full_graph);
        const auto dst = boost::target(current_prop, full_graph);

        const auto src_vtx = full_graph[src];
        const auto dst_vtx = full_graph[dst];

        avg_eucl_dist += src_vtx.pos.CalcEuclDist(dst_vtx.pos);
        sum_weights += weight;
    }

    return avg_eucl_dist / sum_weights;
}

std::tuple<double, double> Graph::calculate_all_pairs_shortest_paths() {
    const auto num_neurons = get_num_vertices();

    const auto [edge_begin_it, edge_end_it] = boost::edges(full_graph);

    const auto E = boost::num_edges(full_graph);

    std::vector<apsp::edge_t> cuda_edges(E);
    std::transform(edge_begin_it, edge_end_it, cuda_edges.begin(), [](const auto& edge) {
        return apsp::edge_t{ static_cast<int>(edge.m_source), static_cast<int>(edge.m_target) };
    });

    const auto weight_map = boost::get(&EdgeProperties::weight, full_graph);

    std::vector<int> weights{};
    std::transform(edge_begin_it, edge_end_it, std::back_inserter(weights), [&](const auto& edge) {
        return weight_map(edge);
    });

    auto edge_array = std::vector<apsp::edge_t>(E);
    auto starts = std::vector<int>(num_neurons + 1); // Starting point for each edge
    std::iota(starts.begin(), starts.end(), 0);

    apsp::graph_cuda_t<std::vector<int>, std::vector<apsp::edge_t>> graph{
        static_cast<int>(num_neurons),
        static_cast<int>(E),
        std::move(starts),
        std::move(weights),
        std::move(cuda_edges)
    };

    std::vector<double> distances(num_neurons * num_neurons);

    apsp::johnson_cuda(graph, distances);

    size_t number_values = 0;

    double avg = 0.0;
    double sum = 0.0;

    for (size_t i = 0; i < num_neurons; i++) {
        for (size_t j = 0; j < num_neurons; j++) {
            // Consider pairs of different neurons only
            if (i != j) {
                const double val = distances[i * num_neurons + j];

                if (val == std::numeric_limits<double>::max()) {
                    continue;
                }

                // Average
                number_values++;
                const double delta = val - avg;
                avg += delta / static_cast<double>(number_values);

                // Sum
                sum += 1 / val;
            }
        }
    }

    const double global_efficiency = sum / static_cast<double>(num_neurons * (num_neurons - 1));

    return { avg, global_efficiency };
}

double Graph::calculate_average_betweenness_centrality() {
    const auto num_neurons = get_num_vertices();
    std::vector<double> v_centrality_vec(num_neurons, 0.0);

    boost::iterator_property_map<std::vector<double>::iterator, boost::identity_property_map>
        v_centrality_map(v_centrality_vec.begin());

    boost::brandes_betweenness_centrality(full_graph,
        centrality_map(v_centrality_map).weight_map(boost::get(&EdgeProperties::weight_inverse, full_graph)));

    const auto average_bc = std::reduce(v_centrality_vec.begin(), v_centrality_vec.end());
    return average_bc / static_cast<double>(num_neurons);
}

double Graph::calculate_clustering_coefficient() {
    average_clustering_coefficient(full_graph, WeightInverse<FullGraph>(full_graph));
    average_clustering_coefficient(full_graph, WeightOne<FullGraph>(full_graph));
    average_clustering_coefficient(full_graph, WeightDivMaxWeight<FullGraph>(full_graph));

    average_clustering_coefficient_unweighted_undirected(full_graph);

    using ClusteringProperty = boost::exterior_vertex_property<ConnectivityGraph, double>;
    using ClusteringContainer = ClusteringProperty::container_type;
    using ClusteringMap = ClusteringProperty::map_type;

    ClusteringContainer coefs(num_vertices(conn_graph));
    ClusteringMap cm(coefs, conn_graph);
    return all_clustering_coefficients(conn_graph, cm);
}

void Graph::add_vertex(const Position& pos, const std::string& name, size_t id) {
    // Add vertex to full_graph, if not there
    if (const auto it = pos_to_vtx.find(pos); it == pos_to_vtx.end()) {
        const FullVertex full_vtx = boost::add_vertex(full_graph);

        // Set vertex properties
        full_graph[full_vtx].name = name;
        full_graph[full_vtx].pos = pos;

        pos_to_vtx[pos] = full_vtx;
        vtx_to_pos[full_vtx] = pos;
        id_to_vtx_full[id] = full_vtx;

        const FullVertex conn_vtx = boost::add_vertex(conn_graph);
        id_to_vtx_conn[id] = conn_vtx;
    }
}

void Graph::add_edge(size_t src_id, size_t dst_id, int weight) {
    if (weight == 0) {
        return;
    }

    const int weight_in_boost = std::abs(weight);

    const FullVertex dst_vtx_full = id_to_vtx_full[dst_id];
    const FullVertex src_vtx_full = id_to_vtx_full[src_id];

    auto [edgefull, success] = boost::edge(src_vtx_full, dst_vtx_full, full_graph);

    if (!success) {
        boost::add_edge(src_vtx_full, dst_vtx_full, full_graph);
        full_graph[edgefull].weight = weight_in_boost;
    } else {
        std::cerr << "FullEdge already in full_graph\n";
        full_graph[edgefull].weight += weight_in_boost;
    }

    const ConnectivityVertex dst_vtx_conn = id_to_vtx_full[dst_id];
    const ConnectivityVertex src_vtx_conn = id_to_vtx_full[src_id];

    ConnectivityEdge edge_conn;
    std::tie(edge_conn, success) = boost::edge(src_vtx_conn, dst_vtx_conn, conn_graph);

    if (!success) {
        boost::add_edge(src_vtx_conn, dst_vtx_conn, conn_graph);
    }
}

void Graph::print_vertex(FullVertex v, std::ostream& os) {
    os << full_graph[v].pos.x << " " << full_graph[v].pos.y << " " << full_graph[v].pos.z << " "
       << "\t" << full_graph[v].name << "\n";
}

void Graph::print_edge(FullEdge e, std::ostream& os) {
    const FullVertex u = source(e, full_graph);
    const FullVertex v = target(e, full_graph);

    os << full_graph[u].pos.x << " " << full_graph[u].pos.y << " " << full_graph[u].pos.z << "  "
       << full_graph[v].pos.x << " " << full_graph[v].pos.y << " " << full_graph[v].pos.z
       << "\n";
}

static void average_clustering_coefficient(typename Graph::FullGraph& graph, const Weight<Graph::FullGraph>& weight) {
    size_t num_denominator_zero = 0;
    size_t num_denominator_greater_than_zero = 0;
    size_t num_bilateral_edges = 0;
    size_t num_vals = 0;
    double avg = 0;

    // For all vertices i
    for (auto [vertex_iter, vertex_iter_end] = boost::vertices(graph); vertex_iter != vertex_iter_end; ++vertex_iter) {
        std::set<typename boost::graph_traits<Graph::FullGraph>::vertex_descriptor> neighbors_of_vertex_i{};
        typename std::set<typename boost::graph_traits<Graph::FullGraph>::vertex_descriptor>::iterator neighbors_of_vertex_i_iter;
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
            std::set<typename boost::graph_traits<Graph::FullGraph>::vertex_descriptor> neighbors_of_vertex_j{};

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

        if (0 == denominator_clustering_coefficient_vertex_i) {
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

static void average_clustering_coefficient_unweighted_undirected(typename Graph::FullGraph& graph) {
    size_t num_denominator_zero = 0;
    size_t num_denominator_greater_than_zero = 0;
    size_t num_vals = 0;
    double avg = 0.0;

    // For all vertices i
    for (auto [vertex_iter, vertex_iter_end] = vertices(graph); vertex_iter != vertex_iter_end; ++vertex_iter) {
        std::set<typename boost::graph_traits<Graph::FullGraph>::vertex_descriptor> neighbors_of_vertex_i{};

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

        if (0 == max_num_triangles_of_vertex_i) {
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
