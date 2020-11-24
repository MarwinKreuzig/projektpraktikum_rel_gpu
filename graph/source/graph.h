#ifndef GRAPH_H_
#define GRAPH_H_

#include <boost/property_map/property_map.hpp>
#include <boost/graph/adjacency_list.hpp>

#include "position.h"

class Graph {
public:
	// Vertex properties (bundled properties)
	struct VertexProperties
	{
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

	bool Init(std::ifstream& input_stream);
	std::pair<Vertex, bool> AddVertex(const Position& pos, const std::string& name);
	const BGLGraph& BGL_Graph();
	std::tuple<double, double, double> SmallestCoordinatePerDimension();
	void AddOffsetToPositions(const Position& offset);
	std::pair<int, int> MinMaxDegree();
	void PrintVertex(Vertex v, std::ostream& os);
	void PrintVertices(std::ostream& os);
	void PrintEdge(Edge e, std::ostream& os);
	void PrintEdges(std::ostream& os);

private:
	BGLGraph graph;
	PositionToVtxMap pos_to_vtx;
	VtxToPositionMap vtx_to_pos;
	Position offset;
};

#endif /* GRAPH_H_ */
