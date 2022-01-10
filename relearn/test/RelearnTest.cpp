/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "gtest/gtest.h"

#include "RelearnTest.hpp"

double RelearnTest::position_bounary = 10000;

int RelearnTest::iterations = 10;
double RelearnTest::eps = 0.00001;

bool RelearnTest::use_predetermined_seed = false;
unsigned int RelearnTest::predetermined_seed = 1366809696;

std::uniform_int_distribution<size_t> RelearnTest::uid_num_ranks(1, upper_bound_num_ranks);
std::uniform_int_distribution<size_t> RelearnTest::uid_num_neurons(1, upper_bound_num_neurons);
std::uniform_int_distribution<size_t> RelearnTest::uid_num_synapses(1, upper_bound_num_synapses);

std::uniform_int_distribution<int> RelearnTest::uid_synapse_weight(-bound_synapse_weight, bound_synapse_weight);

int NetworkGraphTest::num_ranks = 17;
int NetworkGraphTest::num_synapses_per_neuron = 2;

std::uniform_real_distribution<double> VectorTest::uniform_vector_elements(lower_bound, upper_bound);

std::uniform_int_distribution<unsigned short> RelearnTest::uid_refinement(0, max_refinement_level);
std::uniform_int_distribution<unsigned short> RelearnTest::uid_small_refinement(0, small_refinement_level);
std::uniform_int_distribution<unsigned short> RelearnTest::uid_large_refinement(small_refinement_level + 1, max_refinement_level);
