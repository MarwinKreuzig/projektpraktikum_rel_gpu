#include "graph.h"

#include <boost/property_map/property_map.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/betweenness_centrality.hpp>
#include <boost/graph/clustering_coefficient.hpp>
#include <boost/graph/exterior_property.hpp>
#include <boost/graph/johnson_all_pairs_shortest.hpp>
#include <boost/variant/get.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

void Graph::add_vertices_from_file(const std::filesystem::path& file_path) {
	std::ifstream file(file_path);

	std::string line;

	while (std::getline(file, line)) {
		if (line[0] == '#') {
			continue;
		}

		std::stringstream sstream(line);

		size_t id;
		Position pos;
		std::string area, type;

		bool success = (sstream >> id) &&
			(sstream >> pos.x) &&
			(sstream >> pos.y) &&
			(sstream >> pos.z) &&
			(sstream >> area) &&
			(sstream >> type);

		if (!success) {
			continue;
		}

		add_vertex(pos, area + " " + type, id);
	}
}

void Graph::add_edges_from_file(const std::filesystem::path& file_path) {
	std::ifstream file(file_path);

	std::string line;

	while (std::getline(file, line)) {
		if (line[0] == '#') {
			continue;
		}

		std::stringstream sstream(line);

		size_t src_id;
		size_t dst_id;
		int weight;

		bool success = (sstream >> dst_id) &&
			(sstream >> src_id) &&
			(sstream >> weight);

		if (!success) {
			continue;
		}

		int weight_in_boost = weight > 0 ? weight : -weight;

		FullVertex dst_vtx = id_to_vtx_full[dst_id];
		FullVertex src_vtx = id_to_vtx_full[src_id];

		FullEdge edge;
		std::tie(edge, success) = boost::edge(src_vtx, dst_vtx, full_graph);

		if (!success) {
			boost::add_edge(src_vtx, dst_vtx, full_graph);
			std::tie(edge, success) = boost::edge(src_vtx, dst_vtx, full_graph);
			full_graph[edge].weight = weight_in_boost;
		}
		else {
			std::cerr << "FullEdge already in full_graph\n";
			full_graph[edge].weight += weight_in_boost;
			continue;
		}
	}
}

void Graph::print_vertices(std::ostream& os) {
	os << "# Vertices: " << boost::num_vertices(full_graph) << "\n";
	os << "# Add offset (x y z) to get original positions: (" <<
		-offset.x << " " << -offset.y << " " << -offset.z << ")\n";
	os << "# Position (x y z)\tArea" << "\n";

	FullVertexIterator it, it_end;
	std::tie(it, it_end) = boost::vertices(full_graph);
	for (; it != it_end; ++it) {
		print_vertex(*it, os);
	}
}

void Graph::print_edges(std::ostream& os) {
	os << "# <src pos x> <src pos y> <src pos z>  "
		<< "<tgt pos x> <tgt pos y> <tgt pos z>"
		<< "\n";

	FullEdgeIterator it, it_end;
	std::tie(it, it_end) = boost::edges(full_graph);
	for (; it != it_end; ++it) {
		print_edge(*it, os);
	}
}

void Graph::calculate_metrics(std::ostream& os) {
	os << "Calculating different metrics\n";
	
	os << "There are " << get_num_vertices() << " many vertices and " << get_num_edges() << " many edges in the graph\n";
	std::pair<int, int> min_max = min_max_degree();
	os << "The minimum number of edges is: " << min_max.first << " and the maximum is: " << min_max.second << "\n";

	os << "Calculating average euclidean distance...\n";
	const double avg_eucl_dist = calculate_average_euclidean_distance();
	os << "It was: " << avg_eucl_dist << "\n";

	os << "Calculating all pairs shortest paths...\n";
	double avg, glob_eff;
	std::tie(avg, glob_eff) = calculate_all_pairs_shortest_paths();
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

	FullVertexIterator it_vtx, it_vtx_end;

	std::tie(it_vtx, it_vtx_end) = boost::vertices(full_graph);
	for (; it_vtx != it_vtx_end; ++it_vtx) {
		min_coords.MinForEachCoordinate(full_graph[*it_vtx].pos);
	}

	return std::make_tuple(min_coords.x, min_coords.y, min_coords.z);
}

void Graph::add_offset_to_positions(const Position& offset) {
	this->offset.Add(offset);  // Update offset

	FullVertexIterator it_vtx, it_vtx_end;
	std::tie(it_vtx, it_vtx_end) = boost::vertices(full_graph);
	for (; it_vtx != it_vtx_end; ++it_vtx) {
		full_graph[*it_vtx].pos.Add(offset);
	}
}

std::pair<int, int> Graph::min_max_degree() {
	FullVertexIterator it_vtx, it_vtx_end;
	int max_deg = 0;
	int min_deg = std::numeric_limits<int>::max();

	std::tie(it_vtx, it_vtx_end) = boost::vertices(full_graph);
	for (; it_vtx != it_vtx_end; ++it_vtx) {
		max_deg =
			std::max(max_deg, static_cast<int>(boost::out_degree(*it_vtx, full_graph) +
				boost::in_degree(*it_vtx, full_graph)));
		min_deg =
			std::min(min_deg, static_cast<int>(boost::out_degree(*it_vtx, full_graph) +
				boost::in_degree(*it_vtx, full_graph)));
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

	FullEdgeIterator current, end;
	for (std::tie(current, end) = boost::edges(full_graph); current != end; ++current) {
		auto& current_edge = full_graph[*current];
		auto weight = current_edge.weight;

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
		auto current_prop = *current;
		auto& current_edge = full_graph[current_prop];
		auto weight = current_edge.weight;

		auto src = boost::source(current_prop, full_graph);
		auto dst = boost::target(current_prop, full_graph);

		auto src_vtx = full_graph[src];
		auto dst_vtx = full_graph[dst];

		avg_eucl_dist += src_vtx.pos.CalcEuclDist(dst_vtx.pos);
		sum_weights += weight;
	}

	return avg_eucl_dist / sum_weights;
}

std::tuple<double, double> Graph::calculate_all_pairs_shortest_paths() {
	auto num_neurons = get_num_vertices();

	std::vector<std::vector<double>> distances(num_neurons);
	for (size_t i = 0; i < num_neurons; i++) {
		distances[i].resize(num_neurons);
	}

	boost::johnson_all_pairs_shortest_paths(full_graph, distances, boost::weight_map(boost::get(&EdgeProperties::weight_inverse, full_graph)));

	size_t number_values = 0;

	double avg = 0.0;
	double sum = 0.0;

	for (size_t i = 0; i < num_neurons; i++) {
		for (size_t j = 0; j < num_neurons; j++) {
			// Consider pairs of different neurons only
			if (i != j) {
				double val = distances[i][j];

				// Average
				number_values++;
				double delta = val - avg;
				avg += delta / number_values;

				// Sum
				sum += 1 / val;
			}
		}
	}

	double global_efficiency = sum / (num_neurons * (num_neurons - 1));

	return std::make_tuple(avg, global_efficiency);
}

double Graph::calculate_average_betweenness_centrality() {
	auto num_neurons = get_num_vertices();
	std::vector<double> v_centrality_vec(num_neurons, 0.0);

	boost::iterator_property_map<std::vector<double>::iterator, boost::identity_property_map>
		v_centrality_map(v_centrality_vec.begin());

	boost::brandes_betweenness_centrality(full_graph, 
		centrality_map(v_centrality_map).weight_map(boost::get(&EdgeProperties::weight_inverse, full_graph)));

	double average_bc = 0;
	for (size_t i = 0; i < v_centrality_vec.size(); i++) {
		average_bc += v_centrality_vec[i];
	}

	return average_bc / num_neurons;
}

double Graph::calculate_clustering_coefficient() {
	average_clustering_coefficient(full_graph, WeightInverse<FullGraph>(full_graph));
	average_clustering_coefficient(full_graph, WeightOne<FullGraph>(full_graph));
	average_clustering_coefficient(full_graph, WeightDivMaxWeight<FullGraph>(full_graph));

	average_clustering_coefficient_unweighted_undirected(full_graph);

	typedef boost::exterior_vertex_property<ConnectivityGraph, double> ClusteringProperty;
	typedef ClusteringProperty::container_type ClusteringContainer;
	typedef ClusteringProperty::map_type ClusteringMap;

	ClusteringContainer coefs(num_vertices(conn_graph));
	ClusteringMap cm(coefs, conn_graph);
	double cc = all_clustering_coefficients(conn_graph, cm);

	return cc;
}

void Graph::add_vertex(const Position& pos, const std::string& name, size_t id) {
	// Add vertex to full_graph, if not there
	auto it = pos_to_vtx.find(pos);

	if (it == pos_to_vtx.end()) {
		FullVertex full_vtx = boost::add_vertex(full_graph);

		// Set vertex properties
		full_graph[full_vtx].name = name;
		full_graph[full_vtx].pos = pos;

		pos_to_vtx[pos] = full_vtx;
		vtx_to_pos[full_vtx] = pos;
		id_to_vtx_full[id] = full_vtx;

		FullVertex conn_vtx = boost::add_vertex(conn_graph);
		id_to_vtx_conn[id] = conn_vtx;
	}
}

void Graph::add_edge(size_t src_id, size_t dst_id, int weight) {
	if (weight == 0) {
		return;
	}

	int weight_in_boost = std::abs(weight);

	FullVertex dst_vtx_full = id_to_vtx_full[dst_id];
	FullVertex src_vtx_full = id_to_vtx_full[src_id];

	bool success;
	FullEdge edgefull;
	std::tie(edgefull, success) = boost::edge(src_vtx_full, dst_vtx_full, full_graph);

	if (!success) {
		boost::add_edge(src_vtx_full, dst_vtx_full, full_graph);
		full_graph[edgefull].weight = weight_in_boost;
	}
	else {
		std::cerr << "FullEdge already in full_graph\n";
		full_graph[edgefull].weight += weight_in_boost;
	}

	ConnectivityVertex dst_vtx_conn = id_to_vtx_full[dst_id];
	ConnectivityVertex src_vtx_conn = id_to_vtx_full[src_id];

	ConnectivityEdge edge_conn;
	std::tie(edge_conn, success) = boost::edge(src_vtx_conn, dst_vtx_conn, conn_graph);

	if (!success) {
		boost::add_edge(src_vtx_conn, dst_vtx_conn, conn_graph);
	}
}

void Graph::print_vertex(FullVertex v, std::ostream& os) {
	os << full_graph[v].pos.x << " " << full_graph[v].pos.y << " " << full_graph[v].pos.z << " " <<
		"\t" << full_graph[v].name << "\n";
}

void Graph::print_edge(FullEdge e, std::ostream& os) {
	FullVertex u, v;

	u = source(e, full_graph);
	v = target(e, full_graph);

	os << full_graph[u].pos.x << " " << full_graph[u].pos.y << " " << full_graph[u].pos.z << "  "
		<< full_graph[v].pos.x << " " << full_graph[v].pos.y << " " << full_graph[v].pos.z
		<< "\n";
}
