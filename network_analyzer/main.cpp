#include <boost/config.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <string>
#include <set>
#include <iterator>

#include <boost/property_map/property_map.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/johnson_all_pairs_shortest.hpp>
#include <boost/graph/betweenness_centrality.hpp>
#include <boost/graph/exterior_property.hpp>
#include <boost/graph/clustering_coefficient.hpp>

#include "time.h"

/**
 * Classes for functors to use different edge weights with average_clustering_coefficient()
 */
// Base class template
template <typename Graph>
struct Weight
{
    Weight(Graph& graph, const std::string& version) : graph(graph), functor_version(version) {};
    Graph& graph;
    const std::string functor_version;
};

// Edge weight is 1/weight
template <typename Graph>
struct WeightInverse : public Weight<Graph>
{
    WeightInverse(Graph& graph) : Weight<Graph>(graph, "1/weight") { };

    double operator()(typename boost::graph_traits<Graph>::edge_descriptor edge) const
    {
        return this->graph[edge].weight_inverse;
    }
};

// Edge weight is weight/max{weights}
template <typename Graph>
struct WeightDivMaxWeight : public Weight<Graph>
{
    WeightDivMaxWeight(Graph& graph) : Weight<Graph>(graph, "weight/max{weights}") { };

    double operator()(typename boost::graph_traits<Graph>::edge_descriptor edge) const
    {
        return this->graph[edge].weight_div_max_weight;
    }
};

// Edge weight is 1
template <typename Graph>
struct WeightOne : public Weight<Graph>
{
    WeightOne(Graph& graph) : Weight<Graph>(graph, "weight = 1") { };

    double operator()(typename boost::graph_traits<Graph>::edge_descriptor edge) const
    {
        return this->graph[edge].weight_one;
    }
};


template <typename Graph, typename Weight>
void average_clustering_coefficient(Graph& graph, const Weight& weight)
{
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
    for (boost::tie(vertex_iter, vertex_iter_end) = vertices(graph); vertex_iter != vertex_iter_end; ++vertex_iter)
    {
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
        for (boost::tie(adj_curr, adj_end) = adjacent_vertices(vertex_i, graph); adj_curr != adj_end; ++adj_curr)
        {
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
                ++neighbors_of_vertex_i_iter)
        {
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
                    ++neighbors_of_vertex_j_iter)
            {
                vertex_k = *neighbors_of_vertex_j_iter;

                if ((vertex_i != vertex_j) && (vertex_j != vertex_k) && (vertex_i != vertex_k))
                {
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

                    double exponent = ((double) 1) / 3;
                    numerator_clustering_coefficient_vertex_i +=
                            (pow(weight_ij, exponent) + pow(weight_ji, exponent)) *
                            (pow(weight_jk, exponent) + pow(weight_kj, exponent)) *
                            (pow(weight_ik, exponent) + pow(weight_ki, exponent));
                }
            } // for all k
        } // for all j
        size_t denominator_clustering_coefficient_vertex_i = 2 * (total_degree_vertex_i * (total_degree_vertex_i - 1) - 2*num_bilateral_edges_vertex_i);

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

    std::cout << "[" << wall_clock_time() << "] " << "Average clustering coefficient (" << weight.functor_version << "): " << avg << std::endl;
    std::cout << "[" << wall_clock_time() << "] " << "    Number denominators == 0: " << num_denominator_zero << std::endl;
    std::cout << "[" << wall_clock_time() << "] " << "    Number denominators <  0: " << num_denominator_less_than_zero << std::endl;
    std::cout << "[" << wall_clock_time() << "] " << "    Number denominators >  0: " << num_denominator_greater_than_zero << std::endl;
    std::cout << "[" << wall_clock_time() << "] " << "    Number bilateral edges  : " << num_bilateral_edges / 2 << std::endl;
}

template <typename Graph>
void average_clustering_coefficient_unweighted_undirected(Graph& graph)
{
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
    for (boost::tie(vertex_iter, vertex_iter_end) = vertices(graph); vertex_iter != vertex_iter_end; ++vertex_iter)
    {
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

        for (iter_j = neighbors_of_vertex_i.begin(); iter_j != neighbors_of_vertex_i.end(); ++iter_j)
        {
            for (iter_k = std::next(iter_j); iter_k != neighbors_of_vertex_i.end(); ++iter_k)
            {
                vertex_j = *iter_j;
                vertex_k = *iter_k;

                std::tie(std::ignore, found_jk) = boost::edge(vertex_j, vertex_k, graph);
                std::tie(std::ignore, found_kj) = boost::edge(vertex_k, vertex_j, graph);

                if (found_jk || found_kj)
                {
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

        clustering_coefficient_vertex_i = num_triangles_of_vertex_i / (double) max_num_triangles_of_vertex_i;

        // Include in average clustering coefficient
        num_vals++;
        delta = clustering_coefficient_vertex_i - avg;
        avg += delta / num_vals;
    } // for all i

    std::cout << "[" << wall_clock_time() << "] " << "Average clustering coefficient (unweighted, undirected): " << avg << std::endl;
    std::cout << "[" << wall_clock_time() << "] " << "    Number denominators == 0: " << num_denominator_zero << std::endl;
    std::cout << "[" << wall_clock_time() << "] " << "    Number denominators <  0: " << num_denominator_less_than_zero << std::endl;
    std::cout << "[" << wall_clock_time() << "] " << "    Number denominators >  0: " << num_denominator_greater_than_zero << std::endl;
}


int main(int argc, char** argv)
{
    using namespace boost;

    std::ifstream input_filestream;
    size_t num_neurons, src, tgt;
    std::string line;
    bool success;
    double dval;

    // Try to open network file
    if (argc > 1) {
        input_filestream.open(argv[1]);
    }
    if (!(argc > 1) || input_filestream.fail()) {
        std::cout << "Usage: " << argv[0] << " <network_file> [<positions>]" << std::endl;
        exit(EXIT_FAILURE);
    }

    // Read in number of neurons
    success = false;
    if (std::getline(input_filestream, line)) {
        std::stringstream sstream(line);
        char character;

        success = (sstream >> character) && (sstream >> num_neurons);
    }
    if (!success) {
        std::cout << "Could not read \"num_neurons\"." << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "[" << wall_clock_time() << "] [INFO] " << "START" << std::endl;

    std::cout << "num_neurons: " << num_neurons << std::endl;


    /**
     * Create graph from network file
     */

    // Graph edge properties (bundled properties)
    struct EdgeProperties
    {
        double weight;
        double weight_inverse;        // 1/weight (connection-length)
        double weight_div_max_weight; // weight/max{weights}
        double weight_one;
    };
    typedef adjacency_list<
            vecS,               // OutEdgeList
            vecS,               // VertexList
            bidirectionalS,     // Bidirectional
            no_property,        // VertexProperties
            EdgeProperties      // EdgeProperties
            > Graph;
    Graph graph(num_neurons);
    typedef adjacency_list<vecS, vecS, undirectedS> Graph_undirected;
    Graph_undirected graph_undirected(num_neurons);

    /**
     * Init graph
     */
    {
        graph_traits<Graph>::edge_descriptor edge;
        std::string line;
        int weight;

        while (std::getline(input_filestream, line)) {
            std::stringstream sstream(line);

            success = (sstream >> tgt) && (sstream >> src) && (sstream >> weight);

            // Consider only existing edges
            if (success && (weight != 0)) {
                //
                // Transpose matrix from (target, source) to (source, target)
                //
                tie(edge, success) = add_edge(src, tgt, graph);
                graph[edge].weight = fabs(weight);

                add_edge(src, tgt, graph_undirected);


                //std::cout << tgt << ' ' << src << ' ' << fabs(weight) << '\n';
            }
            else {
                std::cerr << "Skipping line: \"" << line << "\"\n";
            }
        }
        std::cout << "num_edges:  " << num_edges(graph) << std::endl;
    }

    /**
     * Init edge weights
     */
    {
        graph_traits<Graph>::edge_iterator edge_curr, edge_end;
        double weight, min_weight, max_weight;

        // Find min and max edge weight
        std::tie(edge_curr, edge_end) = edges(graph);
        min_weight = max_weight = graph[*edge_curr].weight;

        for (boost::tie(edge_curr, edge_end) = edges(graph); edge_curr != edge_end; ++edge_curr)
        {
            weight = graph[*edge_curr].weight;

            min_weight = std::min(min_weight, weight);
            max_weight = std::max(max_weight, weight);
        }
        std::cout << "\n" << "Edge weight (MIN, MAX): (" << min_weight << ", " << max_weight << ")\n";

        for (boost::tie(edge_curr, edge_end) = edges(graph); edge_curr != edge_end; ++edge_curr) {
            weight = graph[*edge_curr].weight;

            graph[*edge_curr].weight_inverse = 1/weight;
            graph[*edge_curr].weight_div_max_weight = weight/max_weight;
            graph[*edge_curr].weight_one = 1;

            //std::cout << graph[*edge_curr].weight << " ";
        }
    }


#if 1
    /**
     * Average Euclidean distance
     */
    std::cout << "[" << wall_clock_time() << "] [INFO] " << "Starting average Euclidean distance" << std::endl;
    {
        FILE* positions_file = NULL;
        size_t num_positions;

        // Try to open positions file
        if (argc > 2) {
            positions_file = std::fopen(argv[2], "r");
        }
        if (NULL == positions_file) {
            std::cout << "INFO: No positions file available... Skipping average Euclidean distance" << std::endl;
        }
        else {
            // Read in number of positions, should be equal to number of neurons
            if (0 == fscanf(positions_file, "# %zd", &num_positions)) {
                std::cout << "Could not read \"num_positions\"." << std::endl;
                exit(EXIT_FAILURE);
            }
            if (num_neurons != num_positions) {
                std::cout << "Number of neurons and number of positions not equal." << std::endl;
                exit(EXIT_FAILURE);
            }

            /**
             * Create positions matrix from positions file
             */
            // Create an uninitialized (num_neurons x 3) matrix
            // The columns correspond to the x, y, z coordinates, respectively.
            std::vector<std::vector<double>> positions_matrix(num_neurons);
            for (size_t i = 0; i < num_neurons; i++) {
                positions_matrix[i].resize(3);
            }

            // Initialize positions matrix with file
            for (size_t dim = 0; dim < 3; dim++) {
                for (size_t neuron_id = 0; neuron_id < num_neurons; neuron_id++) {
                    fscanf(positions_file, "%lf", &dval);
                    positions_matrix[neuron_id][dim] = dval;
                }
            }
            fclose(positions_file);
            std::cout << "[" << wall_clock_time() << "] [INFO] " << "Positions matrix created" << std::endl;

            // Do calculations
            double avg_eucl_dist = 0, sum = 0;
            double x_src, y_src, z_src, x_tgt, y_tgt, z_tgt;
            size_t src, tgt;
            graph_traits<Graph>::vertex_descriptor vertex_desc_src, vertex_desc_tgt;
            graph_traits<Graph>::edge_iterator edge_curr, edge_end;

            // Go through all edges
            for (boost::tie(edge_curr, edge_end) = edges(graph); edge_curr != edge_end; ++edge_curr) {
                dval = graph[*edge_curr].weight;

                // Get source vertex index for edge
                vertex_desc_src = source(*edge_curr, graph);
                src = get(get(vertex_index, graph), vertex_desc_src);

                // Get target vertex index for edge
                vertex_desc_tgt = target(*edge_curr, graph);
                tgt = get(get(vertex_index, graph), vertex_desc_tgt);

                //std::cout << "(" << src << "," << tgt << ")" << std::endl;

                x_src = positions_matrix[src][0];
                y_src = positions_matrix[src][1];
                z_src = positions_matrix[src][2];

                x_tgt = positions_matrix[tgt][0];
                y_tgt = positions_matrix[tgt][1];
                z_tgt = positions_matrix[tgt][2];

                avg_eucl_dist += sqrt(
                        (x_tgt - x_src)*(x_tgt - x_src) +
                        (y_tgt - y_src)*(y_tgt - y_src) +
                        (z_tgt - z_src)*(z_tgt - z_src)) *
                        dval;
                // Total number of synapses
                sum += dval;
            }
            std::cout << "[" << wall_clock_time() << "] " << "Average Euclidean distance: " << avg_eucl_dist/sum << std::endl;
        }
    }
#endif



#if 1
    /**
     * (i)  Average shortest path length (characteristic path length)
     * (ii) Global efficiency
     *
     * NOTE: Both require connection length as edge weight
     */
    std::cout << "[" << wall_clock_time() << "] [INFO] " << "Starting average shortest path length and global efficiency" << std::endl;
    {
        // Allocate distances matrix
        std::vector<std::vector<double>> distances(num_neurons);
        for (size_t i = 0; i < num_neurons; i++) {
            distances[i].resize(num_neurons);
        }

#if 0
        for (int i = 0; i < num_neurons; ++i) {
            for (int j = 0; j < num_neurons; ++j) {
                if (distances[i][j] == (std::numeric_limits<double>::max)())
                    std::cout << "inf";
                else
                    std::cout << distances[i][j];
                std::cout << " ";
            }
            std::cout << std::endl;
        }
#endif

        johnson_all_pairs_shortest_paths(graph, distances, weight_map(get(&EdgeProperties::weight_inverse, graph)));

        // Calc average shortest path length
        size_t num_vals = 0;
        double delta, avg = 0, sum = 0;
        double val;
        for (size_t i = 0; i < num_neurons; i++) {
            for (size_t j = 0; j < num_neurons; j++) {
                val = distances[i][j];

                // Consider pairs of different neurons only
                if (i != j)
                {
                    // Average
                    num_vals++;
                    delta = val - avg;
                    avg += delta / num_vals;

                    // Sum
                    sum += 1/val;
                }
            }
        }
        std::cout << "[" << wall_clock_time() << "] " << "Average shortest path length: " << avg << std::endl;
        std::cout << "[" << wall_clock_time() << "] " << "Global efficiency: " << sum/(num_neurons * (num_neurons - 1)) << std::endl;
    }
#endif


#if 1
    /**
     * Average betweenness centrality
     *
     * NOTE: Requires connection-length as edge weight
     */
    std::cout << "[" << wall_clock_time() << "] [INFO] " << "Starting average betweenness centrality" << std::endl;
    {
        std::vector<double> v_centrality_vec(num_vertices(graph), 0.0);

        // Create external vertex centrality property map
        iterator_property_map<std::vector<double>::iterator, identity_property_map>
        v_centrality_map(v_centrality_vec.begin());

        // We use named parameters of Boost: centrality_map(...).weight_map(...)
        brandes_betweenness_centrality(graph, centrality_map(v_centrality_map).weight_map(get(&EdgeProperties::weight_inverse, graph)));
        /**
         * Scales each absolute centrality by 2/((n-1)(n-2))
         * This is correct for undirected graphs
         */
        //relative_betweenness_centrality(graph, v_centrality_map);


        // Calc average betweenness centrality as sum
        double average_bc = 0;
        double n = num_vertices(graph);
        for (size_t i = 0; i < v_centrality_vec.size(); i++) {
            average_bc += v_centrality_vec[i];
        }
        average_bc /= n; // Average

        std::cout << "[" << wall_clock_time() << "] " << "Average betweenness centrality: " << average_bc << std::endl;
    }
#endif


#if 1
    /**
     * Average clustering coefficient
     *
     * NOTE: Requires weight/max{weights} as edge weight
     */
    std::cout << "[" << wall_clock_time() << "] [INFO] " << "Starting average clustering coefficient" << std::endl;
    {
        average_clustering_coefficient(graph, WeightInverse<Graph>(graph));
        average_clustering_coefficient(graph, WeightOne<Graph>(graph));
        average_clustering_coefficient(graph, WeightDivMaxWeight<Graph>(graph));

        average_clustering_coefficient_unweighted_undirected(graph);

        /**
         * Boost reference value for unweighted undirected clustering coefficient
         */
        // The clustering property, container, and map define the containment
        // and abstract accessor for the clustering coefficients of vertices.
        typedef exterior_vertex_property<Graph_undirected, double> ClusteringProperty;
        typedef ClusteringProperty::container_type ClusteringContainer;
        typedef ClusteringProperty::map_type ClusteringMap;

        // Compute the clustering coefficients of each vertex in the graph
        // and the mean clustering coefficient which is returned from the
        // computation.
        ClusteringContainer coefs(num_vertices(graph_undirected));
        ClusteringMap cm(coefs, graph_undirected);
        double cc = all_clustering_coefficients(graph_undirected, cm);
        std::cout << "[" << wall_clock_time() << "] " << "Boost: average clustering coefficient (unweighted, undirected): " << cc << "\n";
    }
#endif

#if 0
    graph_traits<Graph>::vertex_descriptor vertex_i, vertex_j, vertex_k;
    size_t total_degree_vertex_i, num_bilateral_edges_vertex_i;
    size_t num_denominator_zero, num_denominator_less_than_zero, num_denominator_greater_than_zero;
    size_t num_vals = 0;
    double delta, avg = 0;

    num_denominator_zero = 0;
    num_denominator_less_than_zero = 0;
    num_denominator_greater_than_zero = 0;

    // For all vertices i
    graph_traits<Graph>::vertex_iterator vertex_iter, vertex_iter_end;
    for (boost::tie(vertex_iter, vertex_iter_end) = vertices(graph); vertex_iter != vertex_iter_end; ++vertex_iter)
    {
        graph_traits<Graph>::adjacency_iterator adj_curr, adj_end;
        Graph::inv_adjacency_iterator inv_adj_curr, inv_adj_end;
        graph_traits<Graph>::edge_descriptor edge;
        std::set<graph_traits<Graph>::vertex_descriptor> neighbors_of_vertex_i, neighbors_of_vertex_j;
        std::set<graph_traits<Graph>::vertex_descriptor>::iterator neighbors_of_vertex_i_iter;
        double clustering_coefficient_vertex_i, numerator_clustering_coefficient_vertex_i = 0;
        size_t num_bilateral_edges_vertex_i = 0;
        bool found;

        vertex_i = *vertex_iter;

        // Total degree (in + out) of vertex i
        total_degree_vertex_i = out_degree(vertex_i, graph) + in_degree(vertex_i, graph);
        //std::cout << "total degree: " << total_degree_vertex_i << std::endl;


        // Number of bilateral edges between i and its neighbors j
        for (boost::tie(adj_curr, adj_end) = adjacent_vertices(vertex_i, graph); adj_curr != adj_end; ++adj_curr)
        {
            vertex_j = *adj_curr;

            boost::tie(edge, found) = boost::edge(vertex_j, vertex_i, graph); // TODO: Is edge() available??

            if (found) {
                num_bilateral_edges_vertex_i++;
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
                ++neighbors_of_vertex_i_iter)
        {
            std::set<graph_traits<Graph>::vertex_descriptor> neighbors_of_vertex_j;
            std::set<graph_traits<Graph>::vertex_descriptor>::iterator neighbors_of_vertex_j_iter;

            vertex_j = *neighbors_of_vertex_i_iter;

            // Gather all neighbors of vertex j
            for (boost::tie(adj_curr, adj_end) = adjacent_vertices(vertex_j, graph); adj_curr != adj_end; ++adj_curr)
                neighbors_of_vertex_j.insert(*adj_curr);
            for (boost::tie(inv_adj_curr, inv_adj_end) = inv_adjacent_vertices(vertex_j, graph); inv_adj_curr != inv_adj_end; ++inv_adj_curr)
                neighbors_of_vertex_j.insert(*inv_adj_curr);

            for (neighbors_of_vertex_j_iter = neighbors_of_vertex_j.begin();
                    neighbors_of_vertex_j_iter != neighbors_of_vertex_j.end();
                    ++neighbors_of_vertex_j_iter)
            {
                vertex_k = *neighbors_of_vertex_j_iter;

                if ((vertex_i != vertex_j) && (vertex_j != vertex_k) && (vertex_i != vertex_k))
                {
                    double weight_ij, weight_ji, weight_jk, weight_kj, weight_ik, weight_ki;

                    boost::tie(edge, found) = boost::edge(vertex_i, vertex_j, graph);
                    weight_ij = found ? graph[edge].weight_inverse : 0;

                    boost::tie(edge, found) = boost::edge(vertex_j, vertex_i, graph);
                    weight_ji = found ? graph[edge].weight_inverse : 0;

                    boost::tie(edge, found) = boost::edge(vertex_j, vertex_k, graph);
                    weight_jk = found ? graph[edge].weight_inverse : 0;

                    boost::tie(edge, found) = boost::edge(vertex_k, vertex_j, graph);
                    weight_kj = found ? graph[edge].weight_inverse : 0;

                    boost::tie(edge, found) = boost::edge(vertex_i, vertex_k, graph);
                    weight_ik = found ? graph[edge].weight_inverse : 0;

                    boost::tie(edge, found) = boost::edge(vertex_k, vertex_i, graph);
                    weight_ki = found ? graph[edge].weight_inverse : 0;

                    double exponent = ((double) 1) / 3;
                    numerator_clustering_coefficient_vertex_i +=
                            (pow(weight_ij, exponent) + pow(weight_ji, exponent)) *
                            (pow(weight_jk, exponent) + pow(weight_kj, exponent)) *
                            (pow(weight_ik, exponent) + pow(weight_ki, exponent));
                }
            } // for all k
        } // for all j
        size_t denominator_clustering_coefficient_vertex_i = 2 * (total_degree_vertex_i * (total_degree_vertex_i - 1) - 2*num_bilateral_edges_vertex_i);

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

    std::cout << "[" << wall_clock_time() << "] " << "Average clustering coefficient: " << avg << std::endl;
    std::cout << "[" << wall_clock_time() << "] " << "    Number denominators == 0: " << num_denominator_zero << std::endl;
    std::cout << "[" << wall_clock_time() << "] " << "    Number denominators <  0: " << num_denominator_less_than_zero << std::endl;
    std::cout << "[" << wall_clock_time() << "] " << "    Number denominators >  0: " << num_denominator_greater_than_zero << std::endl;
#endif


    /**
     * Boost Graph Library examples
     */

#if 0
    typedef adjacency_list<vecS, listS, directedS, property<vertex_index_t, std::size_t>,
            property<edge_weight_t, int>> Graph;


    property_map < Graph, vertex_index_t >::type idx = get(vertex_index, g);
    typedef graph_traits<Graph>::vertex_iterator vertex_iter;
    std::pair<vertex_iter, vertex_iter> vp;
    std::size_t i = 0;
    for (vp = vertices(g); vp.first != vp.second; ++vp.first) {
        put(idx, *vp.first, i);
        i++;
        std::cout << idx[*vp.first] <<  " ";
    }
    std::cout << std::endl;

    //graph_traits<Graph>::vertex_descriptor v1 = vertex(1, g);
    //graph_traits<Graph>::vertex_descriptor v2 = vertex(4, g);
    //add_edge(v1, v2, 400, g);

#endif

#if 0

    const int V = 6;
    typedef std::pair < int, int >Edge;

    Edge edge_array[] =
    { Edge(0, 1), Edge(0, 2), Edge(0, 3), Edge(0, 4), Edge(0, 5),
            Edge(1, 2), Edge(1, 5), Edge(1, 3), Edge(2, 4), Edge(2, 5),
            Edge(3, 2), Edge(4, 3), Edge(4, 1), Edge(5, 4)
    };
    const std::size_t E = sizeof(edge_array) / sizeof(Edge);
    Graph g(edge_array, edge_array + E, V);

    add_edge(1, 4, g);

    property_map < Graph, vertex_index_t >::type idx = get(vertex_index, g);

    typedef graph_traits<Graph>::vertex_iterator vertex_iter;
    std::pair<vertex_iter, vertex_iter> vp;
    std::size_t i = 0;
    for (vp = vertices(g); vp.first != vp.second; ++vp.first) {
        put(idx, *vp.first, i);
        i++;
        std::cout << idx[*vp.first] <<  " ";
    }
    std::cout << std::endl;

    property_map < Graph, edge_weight_t >::type w = get(edge_weight, g);
    int weights[] = { 0, 0, 0, 0, 0, 3, -4, 8, 1, 7, 4, -5, 2, 6 };
    int *wp = weights;

    graph_traits < Graph >::edge_iterator e, e_end;
    for (boost::tie(e, e_end) = edges(g); e != e_end; ++e)
        w[*e] = *wp++;

#if 1

    std::vector < int >d(V, (std::numeric_limits < int >::max)());
    int D[V][V];
    johnson_all_pairs_shortest_paths(g, D, distance_map(&d[0]));

    std::cout << "       ";
    for (int k = 0; k < V; ++k)
        std::cout << std::setw(5) << k;
    std::cout << std::endl;
    for (int i = 0; i < V; ++i) {
        std::cout << std::setw(3) << i << " -> ";
        for (int j = 0; j < V; ++j) {
            if (D[i][j] == (std::numeric_limits<int>::max)())
                std::cout << std::setw(5) << "inf";
            else
                std::cout << std::setw(5) << D[i][j];
        }
        std::cout << std::endl;
    }

    std::ofstream fout("figs/johnson-eg.dot");
    fout << "digraph A {\n"
            << "  rankdir=LR\n"
            << "size=\"5,3\"\n"
            << "ratio=\"fill\"\n"
            << "edge[style=\"bold\"]\n" << "node[shape=\"circle\"]\n";

    graph_traits < Graph >::edge_iterator ei, ei_end;
    for (boost::tie(ei, ei_end) = edges(g); ei != ei_end; ++ei)
        fout << source(*ei, g) << " -> " << target(*ei, g)
        << "[label=" << get(edge_weight, g)[*ei] << "]\n";

    fout << "}\n";
#endif

#endif

    return 0;
}
