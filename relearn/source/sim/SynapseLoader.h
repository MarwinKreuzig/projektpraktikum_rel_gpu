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

#include "Types.h"

#include <memory>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

class Essentials;
class Partition;

/**
 * SynapseLoader is a type that abstracts away the mechanics of how synapses are loaded.
 */
class SynapseLoader {
protected:
    using synapses_tuple_type = std::tuple<LocalSynapses, DistantInSynapses, DistantOutSynapses>;
    /*
     * <Static synapses, plastic synapses>
     */
    using synapses_pair_type = std::pair<synapses_tuple_type, synapses_tuple_type>;

public:
    /**
     * @brief Constructs a SynapseLoader with the given Partition
     * @param partition The partition to use
     */
    explicit SynapseLoader(std::shared_ptr<Partition> partition)
        : partition(std::move(partition)) { }

    virtual ~SynapseLoader() = default;

    SynapseLoader(const SynapseLoader& other) = default;
    SynapseLoader(SynapseLoader&& other) = default;

    SynapseLoader& operator=(const SynapseLoader& other) = default;
    SynapseLoader& operator=(SynapseLoader&& other) = default;

    /**
     * @brief Loads all synapses that affect the local neurons, which are
     *      (1) local synapses (local neuron to local neuron)
     *      (2) in synapses (non-local neuron to local neuron)
     *      (3) out synpases (local neuron to non-local neuron)
     *      for both plastic and static synapses
     * @param essentials The essentials to print the metrics
     * @return Two tuples of (local, in, out) synapes, the first one being static synapses, the second one being plastic synapses
     */
    synapses_pair_type load_synapses(const std::unique_ptr<Essentials>& essentials);

protected:
    std::shared_ptr<Partition> partition{};

    virtual synapses_pair_type internal_load_synapses() = 0;
};
