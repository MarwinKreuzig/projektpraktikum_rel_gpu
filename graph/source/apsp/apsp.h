#ifndef graph_apsp_h
#define graph_apsp_h

#include <vector>

#include "../graph.h"

namespace apsp {
std::vector<double> johnson(typename Graph::FullGraph& full_graph, size_t num_neurons, bool use_cuda_if_available = true);
std::vector<double> johnson_cuda(typename Graph::FullGraph& full_graph, size_t num_neurons);
std::vector<double> johnson_parallel(typename Graph::FullGraph& full_graph, size_t num_neurons);
}

#endif
