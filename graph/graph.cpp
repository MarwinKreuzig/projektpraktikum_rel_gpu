#include "graph.h"

#include <iostream>
#include <fstream>
#include <string>
#include <limits>
#include <tuple>
#include <algorithm>
#include <utility>

#include <boost/property_map/property_map.hpp>
#include <boost/graph/adjacency_list.hpp>

const Graph::BGLGraph& Graph::BGL_Graph() {
	return graph;
}

bool Graph::Init(std::ifstream& input_filestream) {
	std::string line;

	// Consume table headers
	if (!std::getline(input_filestream, line)) {
		std::cerr << "Could not read table headers\n";
		return false;
	}

	std::string origin_name, target_name;
	Position origin_pos, target_pos;
	bool success;
	while (std::getline(input_filestream, line)) {
		std::stringstream sstream(line);

		success = (sstream >> origin_name) &&
				  (sstream >> origin_pos.x) &&
				  (sstream >> origin_pos.y) &&
				  (sstream >> origin_pos.z) &&
				  (sstream >> target_name) &&
				  (sstream >> target_pos.x) &&
				  (sstream >> target_pos.y) &&
				  (sstream >> target_pos.z);

		if (!success) {
			std::cerr << "Skipping line: \"" << line << "\"\n";
			continue;
		}

		// Add origin and target to graph, if not there
		Vertex origin_vtx, target_vtx;
		std::tie(origin_vtx, std::ignore) = AddVertex(origin_pos, origin_name);
		std::tie(target_vtx, std::ignore) = AddVertex(target_pos, target_name);

		// Check whether edge already exists
		Edge edge;
		std::tie(edge, success) = boost::edge(origin_vtx, target_vtx, graph);

		// Add edge to graph, if not there
		if (!success) {
			boost::add_edge(origin_vtx, target_vtx, graph);
		} else {
			std::cerr << "Edge already in graph\n";
			continue;
		}
	}
	return true;
}

std::pair<Graph::Vertex, bool> Graph::AddVertex(const Position& pos, const std::string& name) {
	// Add vertex to graph, if not there
	auto it = pos_to_vtx.find(pos);

	if (it == pos_to_vtx.end()) {
		Vertex vtx = boost::add_vertex(graph);

		// Set vertex properties
		graph[vtx].name = name;
		graph[vtx].pos = pos;

		// Add key-value pairs (pos, vtx) and
		// (vtx, pos)
		pos_to_vtx[pos] = vtx;
		vtx_to_pos[vtx] = pos;

		return std::make_pair(vtx, true);
	}
	Vertex vtx = it->second;
//	std::cerr << "Vertex already in graph\n";
	return std::make_pair(vtx, false);
}

std::tuple<double, double, double>
	Graph::SmallestCoordinatePerDimension() {

	const double max_double = std::numeric_limits<double>::max();
	Position min_coords(max_double, max_double, max_double);

	VertexIterator it_vtx, it_vtx_end;

	std::tie(it_vtx, it_vtx_end) = boost::vertices(graph);
	for (; it_vtx != it_vtx_end; ++it_vtx) {
		min_coords.MinForEachCoordinate(graph[*it_vtx].pos);
	}

	return std::make_tuple(min_coords.x, min_coords.y, min_coords.z);
}

void Graph::AddOffsetToPositions(const Position& offset) {
	this->offset.Add(offset);  // Update offset

	VertexIterator it_vtx, it_vtx_end;
	std::tie(it_vtx, it_vtx_end) = boost::vertices(graph);
	for (; it_vtx != it_vtx_end; ++it_vtx) {
		graph[*it_vtx].pos.Add(offset);
	}
}

std::pair<int, int> Graph::MinMaxDegree() {
    VertexIterator it_vtx, it_vtx_end;
    int max_deg = 0;
    int min_deg = std::numeric_limits<int>::max();

    std::tie(it_vtx, it_vtx_end) = boost::vertices(graph);
    for (; it_vtx != it_vtx_end; ++it_vtx) {
    	max_deg =
    			std::max(max_deg, static_cast<int>(boost::out_degree(*it_vtx, graph) +
     					 boost::in_degree(*it_vtx, graph)));
    	min_deg =
    			std::min(min_deg, static_cast<int>(boost::out_degree(*it_vtx, graph) +
    					 boost::in_degree(*it_vtx, graph)));
    }

    return std::make_pair(min_deg, max_deg);
}

void Graph::PrintVertex(Vertex v, std::ostream& os) {
	os << graph[v].pos.x << " " << graph[v].pos.y << " " << graph[v].pos.z << " " <<
		  "\t" << graph[v].name << "\n";
}

void Graph::PrintVertices(std::ostream& os) {
	os << "# Vertices: " << boost::num_vertices(graph) << "\n";
	os << "# Add offset (x y z) to get original positions: (" <<
		  -offset.x << " " << -offset.y << " " << -offset.z << ")\n";
	os << "# Position (x y z)\tArea" << "\n";

	VertexIterator it, it_end;
	std::tie(it, it_end) = boost::vertices(graph);
	for (; it != it_end; ++it) {
		PrintVertex(*it, os);
	}
}

void Graph::PrintEdge(Edge e, std::ostream& os) {
	Vertex u, v;

	u = source(e, graph);
	v = target(e, graph);

	os << graph[u].pos.x << " " << graph[u].pos.y << " " << graph[u].pos.z << "  "
	   << graph[v].pos.x << " " << graph[v].pos.y << " " << graph[v].pos.z
	   << "\n";
}

void Graph::PrintEdges(std::ostream& os) {
	os << "# <src pos x> <src pos y> <src pos z>  "
	   << "<tgt pos x> <tgt pos y> <tgt pos z>"
	   << "\n";

	EdgeIterator it, it_end;
	std::tie(it, it_end) = boost::edges(graph);
	for (; it != it_end; ++it) {
		PrintEdge(*it, os);
	}
}
