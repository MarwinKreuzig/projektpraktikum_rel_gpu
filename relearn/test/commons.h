#pragma once

#include <random>
#include <vector>

#include "../../../source/neurons/NetworkGraph.h"

//#define private public
//#define protected public

constexpr int iterations = 10;
constexpr double eps = 0.00001;
constexpr size_t num_neurons_test = 1000;


extern std::mt19937 mt;

void setup();

NetworkGraph generate_random_network_graph(size_t num_neurons, size_t num_synapses, double threshold_exc);

std::vector<size_t> generate_random_ids(size_t id_low, size_t id_high, size_t num_disables);
