#ifndef graph_apsp_h
#define graph_apsp_h

#include <vector>

#include "../graph.h"

namespace apsp {
/**
 * @brief Johnson algorithm to get the avarage pair shortest paths
 *
 * @param full_graph the input graph
 * @param num_neurons number of neurons
 * @param use_cuda_if_available select the cuda implementation if the cuda implementation is available (compiled)
 * @return std::vector<double> distances
 */
std::vector<double> johnson(typename Graph::FullGraph& full_graph, size_t num_neurons, bool use_cuda_if_available = true);

/**
 * @brief Johnson algorithm to get the avarage pair shortest paths; CUDA implementation
 *
 * @warning Calling this function when CUDA_FOUND is 0 triggers an assertion and returns and empty vector
 *
 * @param full_graph the input graph
 * @param num_neurons number of neurons
 * @return std::vector<double> distances
 */
std::vector<double> johnson_cuda(typename Graph::FullGraph& full_graph, size_t num_neurons);

/**
 * @brief Johnson algorithm to get the avarage pair shortest paths; OpenMP implementation (serial if no OpenMP available)
 *
 * @param full_graph the input graph
 * @param num_neurons number of neurons
 * @return std::vector<double> distances
 */
std::vector<double> johnson_parallel(typename Graph::FullGraph& full_graph, size_t num_neurons);
}

#endif
