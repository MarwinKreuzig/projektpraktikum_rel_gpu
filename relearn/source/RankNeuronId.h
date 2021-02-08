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

#include "Config.h"
#include "RelearnException.h"

#include <ostream>

/**
* Identifies a neuron by the MPI rank of its owner
* and its neuron id on the owner, i.e., the pair <rank, neuron_id>
*/
class RankNeuronId {
    int rank{ -1 }; // MPI rank of the owner
    size_t neuron_id{ Constants::uninitialized }; // Neuron id on the owner

public:
    RankNeuronId(int rank, size_t neuron_id) noexcept
        : rank(rank)
        , neuron_id(neuron_id) { }

    [[nodiscard]] int get_rank() const {
        RelearnException::check(rank >= 0, "RankNeuronId::get_rank, it was: %d", rank);
        return rank;
    }

    [[nodiscard]] size_t get_neuron_id() const {
        RelearnException::check(neuron_id < Constants::uninitialized, "RankNeuronId::get_neuron_id, it was: %u", Constants::uninitialized);
        return neuron_id;
    }

    bool operator==(const RankNeuronId& other) const noexcept {
        return (this->rank == other.rank && this->neuron_id == other.neuron_id);
    }

    friend std::ostream& operator<<(std::ostream& os, const RankNeuronId& rni) {
        os << "Rank: " << rni.get_rank() << "\t id: " << rni.get_neuron_id() << "\n";
        return os;
    }
};
