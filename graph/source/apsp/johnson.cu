#include "johnson.hpp"

#include <algorithm>
#include <climits>
#include <ios>
#include <iostream>
#include <vector>

#include "util.hpp"

#define THREADS_PER_BLOCK 32

namespace apsp {

__constant__ graph_cuda_t<View<int>, View<edge_t>> graph_const;

__forceinline__
    __device__ int
    min_distance(const int* dist, const char* visited, int n) {
    int min = INT_MAX;
    int min_index = 0;
    for (int v = 0; v < n; v++) {
        if ((visited[v] == 0) && dist[v] <= min) {
            min = dist[v];
            min_index = v;
        }
    }
    return min_index;
}

__global__ void dijkstra_kernel(View<int> output, View<char> visited_global) {
    const auto s = blockIdx.x * blockDim.x + threadIdx.x; // NOLINT(readability-static-accessed-through-instance)
    const int V = graph_const.V;

    if (s >= V) {
        return;
    }

    const auto starts = graph_const.starts;
    const auto weights = graph_const.weights;
    const auto edge_array = graph_const.edge_array;

    int* dist = &output[s * V];
    char* visited = &visited_global[s * V];
    for (int i = 0; i < V; i++) {
        dist[i] = INT_MAX;
        visited[i] = 0;
    }
    dist[s] = 0;
    for (int count = 0; count < V - 1; count++) {
        const int u = min_distance(dist, visited, V);
        const int u_start = starts[u];
        const int u_end = starts[u + 1];
        const int dist_u = dist[u];
        visited[u] = 1;
        for (int v_i = u_start; v_i < u_end; v_i++) {
            const int v = edge_array[v_i].v;
            if ((visited[v] == 0) && dist_u != INT_MAX && dist_u + weights[v_i] < dist[v]) {
                dist[v] = dist_u + weights[v_i];
            }
        }
    }
}

__global__ void bellman_ford_kernel(int* dist) {
    const int E = graph_const.E;
    const auto e = threadIdx.x + blockDim.x * blockIdx.x; // NOLINT(readability-static-accessed-through-instance)

    if (e >= E) {
        return;
    }
    const auto weights = graph_const.weights;
    const auto edges = graph_const.edge_array;
    const int u = edges[e].u;
    const int v = edges[e].v;
    const int new_dist = weights[e] + dist[u];
    // Make ATOMIC
    if (dist[u] != INT_MAX && new_dist < dist[v]) {
        atomicExch(&dist[v], new_dist); // Needs to have conditional be atomic too
    }
}

__host__ bool bellman_ford_cuda(graph_cuda_t<std::vector<int>, std::vector<edge_t>>& gr, std::vector<int>& dist, int s) {
    const int V = gr.V;
    const int E = gr.E;

    std::fill(dist.begin(), dist.end(), INT_MAX);
    dist[s] = 0;

    RAIIDeviceMemory<int> device_dist{ dist };

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
        if (dist[u] != INT_MAX && dist[u] + weight < dist[v]) {
            no_neg_cycle = false;
        }
    }

    return no_neg_cycle;
}

/**************************************************************************
                        Johnson's Algorithm CUDA
**************************************************************************/

__host__ void johnson_cuda(graph_cuda_t<std::vector<int>, std::vector<edge_t>>& gr, std::vector<int>& output) {

    //cudaThreadSetCacheConfig(cudaFuncCachePreferL1);

    // Const Graph Initialization
    const int V = gr.V;
    const int E = gr.E;
    // Structure of the graph
    auto device_edge_array = RAIIDeviceMemory<edge_t>(gr.edge_array);
    auto device_weights = RAIIDeviceMemory<int>(gr.weights);
    auto device_output = RAIIDeviceMemory<int>(V * V);
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

    // errors on destruction
    auto bf_graph = graph_cuda_t<std::vector<int>, std::vector<edge_t>>{
        V + 1,
        E,
        std::vector<int>(),
        std::vector<int>(E),
        std::vector<edge_t>(E)
    };

    std::memcpy(bf_graph.edge_array.data(), gr.edge_array.data(), gr.E * sizeof(edge_t));
    std::memcpy(bf_graph.weights.data(), gr.weights.data(), gr.E * sizeof(int));

    std::vector<int> h(bf_graph.V);

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
