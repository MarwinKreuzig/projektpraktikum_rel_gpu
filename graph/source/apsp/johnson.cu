#include "johnson.h"

#include <algorithm>
#include <climits>
#include <ios>
#include <iostream>
#include <limits>
#include <vector>

#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/johnson_all_pairs_shortest.hpp>

#include "util.hpp"

#define THREADS_PER_BLOCK 32

namespace apsp {
const double double_max = std::numeric_limits<double>::max();
const float float_max = std::numeric_limits<float>::max();

__constant__ graph_cuda_t<View<int>, View<edge_t>> graph_const;

__forceinline__
    __device__ int
    min_distance(const double* dist, const char* visited, int n) {
    double min = double_max;
    int min_index = 0;
    for (int v = 0; v < n; v++) {
        if ((visited[v] == 0) && dist[v] <= min) {
            min = dist[v];
            min_index = v;
        }
    }
    return min_index;
}

__global__ void dijkstra_kernel(View<double> output, View<char> visited_global) {
    const auto s = blockIdx.x * blockDim.x + threadIdx.x; // NOLINT(readability-static-accessed-through-instance)
    const int V = graph_const.V;

    if (s >= V) {
        return;
    }

    const auto starts = graph_const.starts;
    const auto weights = graph_const.weights;
    const auto edge_array = graph_const.edge_array;

    double* dist = &output[s * V];
    char* visited = &visited_global[s * V];
    for (int i = 0; i < V; i++) {
        dist[i] = double_max;
        visited[i] = 0;
    }
    dist[s] = 0.0;

    for (int count = 0; count < V - 1; count++) {
        const auto u = min_distance(dist, visited, V);
        const auto u_start = starts[u];

        if (u_start == -1) {
            continue;
        }

        // find next non -1 index (the end of this vertecies edge list)
        auto u_end = starts[u + 1];

        for (int offset = 1; u_end == -1 && u + 1 + offset < V + 1; ++offset) {
            u_end = starts[u + 1 + offset];
        }

        const auto dist_u = dist[u];
        visited[u] = 1;

        for (int v_i = u_start; v_i < u_end; v_i++) {
            const auto v = edge_array[v_i].v;

            if ((visited[v] == 0) && dist_u != double_max && dist_u + weights[v_i] < dist[v]) {
                dist[v] = dist_u + weights[v_i];
            }
        }
    }
}

__global__ void bellman_ford_kernel(float* dist) {
    const int E = graph_const.E;
    const auto e = threadIdx.x + blockDim.x * blockIdx.x; // NOLINT(readability-static-accessed-through-instance)

    if (e >= E) {
        return;
    }
    const auto weights = graph_const.weights;
    const auto edges = graph_const.edge_array;
    const auto u = edges[e].u;
    const auto v = edges[e].v;
    const auto new_dist = weights[e] + dist[u];
    // Make ATOMIC
    // race condition?
    if (dist[u] != float_max && new_dist < dist[v]) {
        atomicExch(&dist[v], new_dist); // Needs to have conditional be atomic too
    }
}

__host__ bool bellman_ford_cuda(graph_cuda_t<std::vector<int>, std::vector<edge_t>>& gr, std::vector<float>& dist, int s) {
    const int E = gr.E;
    const int V = gr.V;

    std::fill(dist.begin(), dist.end(), std::numeric_limits<float>::max());
    dist[s] = 0;

    RAIIDeviceMemory<float> device_dist{ dist };

    const int blocks = (E + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    for (int i = 1; i <= V - 1; i++) {
        bellman_ford_kernel<<<blocks, THREADS_PER_BLOCK>>>(device_dist.data());
    }

    copy(dist, device_dist, cudaMemcpyDeviceToHost);

    // use OMP to parallelize. Not worth sending to GPU
    bool no_neg_cycle = true;
    const auto edges = gr.edge_array;
    const auto weights = gr.weights;
#ifdef _OPENMP
#pragma omp parallel for reduction(and \
                                   : no_neg_cycle)
#endif
    for (int i = 0; i < E; i++) {
        const auto [u, v] = edges[i];
        const int weight = weights[i];
        if (dist[u] != std::numeric_limits<float>::max()
            && dist[u] + weight < dist[v]) {
            no_neg_cycle = false;
        }
    }

    return no_neg_cycle;
}

/**************************************************************************
                        Johnson's Algorithm CUDA
**************************************************************************/

__host__ void johnson_cuda(graph_cuda_t<std::vector<int>, std::vector<edge_t>>& gr, std::vector<double>& output) {
    //cudaThreadSetCacheConfig(cudaFuncCachePreferL1);

    // Const Graph Initialization
    const int V = gr.V;
    const int E = gr.E;
    // Structure of the graph
    auto device_edge_array = RAIIDeviceMemory<edge_t>(gr.edge_array);
    auto device_weights = RAIIDeviceMemory<int>(gr.weights);
    auto device_output = RAIIDeviceMemory<double>(V * V);
    auto device_starts = RAIIDeviceMemory<int>(gr.starts);
    // Needed to run dijkstra
    auto device_visited = RAIIDeviceMemory<char>(V * V);

    auto graph_params = graph_cuda_t<View<int>, View<edge_t>>{
        V,
        E,
        device_starts,
        device_weights,
        device_edge_array
    };
    // Constant memory parameters
    cudaMemcpyToSymbol(graph_const, &graph_params, sizeof(decltype(graph_params)));
    // End initialization

    auto bf_graph = graph_cuda_t<std::vector<int>, std::vector<edge_t>>{
        V + 1,
        E,
        std::vector<int>(),
        std::vector<int>(E),
        std::vector<edge_t>(E)
    };

    std::memcpy(bf_graph.edge_array.data(), gr.edge_array.data(), gr.E * sizeof(edge_t));
    std::memcpy(bf_graph.weights.data(), gr.weights.data(), gr.E * sizeof(int));

    std::vector<float> h(bf_graph.V);

    if (bool r = bellman_ford_cuda(bf_graph, h, V); !r) {
        std::cerr << "\nNegative Cycles Detected! Terminating Early\n";
        exit(1);
    }

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int e = 0; e < E; e++) {
        const auto [u, v] = gr.edge_array[e];
        gr.weights[e] += h[u] - h[v];
    }

    const int blocks = (V + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    copy(device_weights, gr.weights, cudaMemcpyHostToDevice);

    dijkstra_kernel<<<blocks, THREADS_PER_BLOCK>>>(device_output, device_visited);

    copy(output, device_output, cudaMemcpyDeviceToHost);

    if (const cudaError_t errCode = cudaPeekAtLastError(); errCode != cudaSuccess) {
        std::cerr << "WARNING: A CUDA error occured: code=" << errCode << "," << cudaGetErrorString(errCode) << "\n";
    }

    // Remember to reweight edges back -- for every s reweight every v
    // Could do in a kernel launch or with OMP
}

} // namespace apsp
