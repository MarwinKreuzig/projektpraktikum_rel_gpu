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
size_t RelearnTest::num_neurons_test = 1000;
double RelearnTest::eps = 0.00001;

bool RelearnTest::use_predetermined_seed = false;
unsigned int RelearnTest::predetermined_seed = 4260214720;

size_t NetworkGraphTest::upper_bound_num_neurons = 10000;
int NetworkGraphTest::bound_synapse_weight = 10;
int NetworkGraphTest::num_ranks = 17;
int NetworkGraphTest::num_synapses_per_neuron = 2;
