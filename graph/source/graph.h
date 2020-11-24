#pragma once

#include "position.h"

#include <boost/property_map/property_map.hpp>
#include <boost/graph/adjacency_list.hpp>

class Graph {
public:
	// Vertex properties (bundled properties)
	struct VertexProperties {
		std::string name;
		Position pos;
	};

	using BGLGraph = boost::adjacency_list<
		boost::vecS,        	  // OutEdgeList
		boost::vecS,              // VertexList
		boost::bidirectionalS,    // Bidirectional
		VertexProperties,         // VertexProperties
		boost::no_property        // EdgeProperties
	>;

	using Edge = boost::graph_traits<BGLGraph>::edge_descriptor;
	using Vertex = boost::graph_traits<BGLGraph>::vertex_descriptor;
	using VertexIterator = boost::graph_traits<BGLGraph>::vertex_iterator;
	using EdgeIterator = boost::graph_traits<BGLGraph>::edge_iterator;
	using PositionToVtxMap = std::map<Position, Vertex, Position::less>;
	using VtxToPositionMap = std::map<Vertex, Position>;

	bool init(std::ifstream& input_filestream);

	std::pair<Vertex, bool> add_vertex(const Position& pos, const std::string& name);

	const BGLGraph& BGL_Graph();

	std::tuple<double, double, double> smallest_coordinate_per_dimension();

	void add_offset_to_positions(const Position& offset);

	std::pair<int, int> min_max_degree();

	void print_vertex(Vertex v, std::ostream& os);

	void print_vertices(std::ostream& os);

	void print_edge(Edge e, std::ostream& os);

	void print_edges(std::ostream& os);

private:
	BGLGraph graph;
	PositionToVtxMap pos_to_vtx;
	VtxToPositionMap vtx_to_pos;
	Position offset;
};
