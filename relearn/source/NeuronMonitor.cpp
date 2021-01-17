/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "NeuronMonitor.h"

std::shared_ptr<Neurons> NeuronMonitor::neurons_to_monitor {};
size_t NeuronMonitor::current_step = 0;
size_t NeuronMonitor::max_steps = 0;
