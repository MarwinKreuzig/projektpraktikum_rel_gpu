#include "johnson.h"

#include <atomic>
#include <algorithm>
#include <iostream> // cerr
#include <limits>
#include <memory>
#include <random> // mt19937_64, uniform_x_distribution
#include <vector>
#include <mutex>
#include <shared_mutex>

#include <omp.h>

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

#ifdef _OPENMP
		std::vector<std::shared_mutex> mv(dist.size());
#endif
		for (int i = 1; i <= V - 1; i++) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
			for (int j = 0; j < E; j++) {
				const auto [u, v] = edges[j];

#ifdef _OPENMP
				std::shared_lock l{ mv[u], std::defer_lock };
				std::unique_lock l2{ mv[v], std::defer_lock };
				std::lock(l, l2);
#endif

				const auto new_dist = weights[j] + dist[u];
				if (dist[u] != std::numeric_limits<float>::max() && new_dist < dist[v]) {
					dist[v] = new_dist;
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

#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
			for (int64_t s = 0; s < V; s++) {
				boost::dijkstra_shortest_paths(G, s, boost::distance_map(d.data()));

				for (int64_t v = 0; v < V; v++) {
					output[s * V + v] = d[v] /*+ h[v] - h[s]*/;
				}

				counter++;

				if (omp_get_thread_num() == 0) {
					std::cout << counter << '\n';
				}
			}
		}

		std::cout << std::endl;
	}

} // namespace apsp
