/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#pragma once

<<<<<<< HEAD:relearn/source/Random.cpp
namespace randomNumberSeeds {
unsigned int partition;
unsigned int octree;
} // namespace randomNumberSeeds

std::map<RandomHolderKey, std::mt19937> RandomHolder::random_number_generators;
=======
enum class ElementType { AXON,
    DENDRITE };
>>>>>>> merged2:relearn/source/neurons/ElementType.h
