#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "RelearnTest.hpp"
#include "util/MPIRank.h"

#include <memory>
#include <utility>

class Neurons;

class NetworkGraph;

class Partition;

class NeuronsTest : public RelearnTest {
protected:
    static std::tuple <std::shared_ptr<Neurons>, std::shared_ptr<NetworkGraph>>
    create_neurons_object(std::shared_ptr <Partition> &partition, MPIRank rank);
};
