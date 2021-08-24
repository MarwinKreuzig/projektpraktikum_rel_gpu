#include "johnson.h"

#include <atomic>
#include <algorithm>
#include <limits>
#include <memory>
#include <random> // mt19937_64, uniform_x_distribution
#include <vector>
#include <mutex>
#include <shared_mutex>
#include <atomic>

#include <spdlog/spdlog.h>
#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/johnson_all_pairs_shortest.hpp>

namespace apsp {

	static bool bellman_ford(const graph_t& gr, std::vector<float>& dist) {
		const int& V = gr.V;
		const int& E = gr.E;
		const auto& edges = gr.edge_array;
		const auto& weights = gr.weights;

#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int i = 0; i < V; i++) {
			dist[i] = std::numeric_limits<float>::max();
		}
		dist.back() = 0;

		for (int i = 1; i < V; i++) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
			for (int j = 0; j < E; j++) {
				const auto [u, v] = edges[j];

				std::atomic_ref<double> dist_u{ dist[u] };
				const auto new_dist = weights[j] + dist_u.load(std::memory_order_relaxed);

				std::atomic_ref<double> dist_a{ dist[v] };
				auto current_dist = dist_a.load(std::memory_order_relaxed);
				while (new_dist < current_dist && !dist_a.compare_exchange_weak(current_dist, new_dist, std::memory_order_relaxed)) {
				}
			}
		}

		bool no_neg_cycle = true;
#ifdef _OPENMP
#pragma omp parallel for reduction(&& \
                                   : no_neg_cycle)
#endif
		for (int i = 0; i < E; i++) {
			const auto [u, v] = edges[i];
			const auto weight = weights[i];
			if (dist[u] != std::numeric_limits<float>::max() && dist[u] + weight < dist[v]) {
				no_neg_cycle = false;
			}
		}
		return no_neg_cycle;
	}

	void johnson_parallel_impl(graph_t& gr, std::vector<float>& output) {

		const int64_t  V = gr.V;

		APSP_Graph G(gr.edge_array.begin(), gr.edge_array.end(), gr.weights.begin(), V);

		std::cout << "Performing Dijkstra on nodes\n";

		std::atomic<int> counter(0);

#ifdef _OPENMP
#pragma omp parallel
#endif
		{
			std::vector<float> d(boost::num_vertices(G));

			// Second, the Bellman–Ford algorithm is used, starting from the new vertex q,
			// to find for each vertex v the minimum weight h(v) of a path from q to v. If
			// this step detects a negative cycle, the algorithm is terminated.
			// TODO Can run parallel version?
			std::vector<double> h(bf_graph.V);
			if (const bool r = bellman_ford(bf_graph, h); !r) {
				spdlog::error("Johnson: Negative cycles deteced! Terminating program");
				std::terminate();
			}
			// Next the edges of the original graph are reweighted using the values computed
			// by the Bellman–Ford algorithm: an edge from u to v, having length
			// w(u,v), is given the new length w(u,v) + h(u) − h(v).
#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
			for (int64_t s = 0; s < V; s++) {
				boost::dijkstra_shortest_paths(G, s, boost::distance_map(d.data()));

				for (int64_t v = 0; v < V; v++) {
					output[s * V + v] = d[v] /*+ h[v] - h[s]*/;
				}

				counter++;
			}
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
			for (int s = 0; s < V; s++) {
				std::vector<double> d(V);
				boost::dijkstra_shortest_paths(G, s, boost::distance_map(d.data()));
				for (int v = 0; v < V; v++) {
					output[static_cast<size_t>(s) * static_cast<size_t>(V) + static_cast<size_t>(v)] = d[v] + h[v] - h[s];
				}
			}
		}
	}
} // namespace apsp
