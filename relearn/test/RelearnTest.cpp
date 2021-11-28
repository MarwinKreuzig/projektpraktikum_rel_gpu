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
unsigned int RelearnTest::predetermined_seed = 2572984436;

std::uniform_int_distribution<size_t> RelearnTest::uid_num_ranks(1, upper_bound_num_ranks);
std::uniform_int_distribution<size_t> RelearnTest::uid_num_neurons(1, upper_bound_num_neurons);

std::uniform_real_distribution<double> RelearnTest::urd_percentage(0.0, std::nextafter(1.0, 2.0)); // [0.0, 1.0]

size_t NetworkGraphTest::upper_bound_num_neurons = 10000;
int NetworkGraphTest::bound_synapse_weight = 10;
int NetworkGraphTest::num_ranks = 17;
int NetworkGraphTest::num_synapses_per_neuron = 2;
