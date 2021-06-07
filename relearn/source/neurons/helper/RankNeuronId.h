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

#include "../../Config.h"
#include "../../mpi/MPIWrapper.h"
#include "../../util/RelearnException.h"

#include <ostream>

/**
* Identifies a neuron by the MPI rank of its owner
* and its neuron id on the owner, i.e., the pair <rank, neuron_id>
*/
class RankNeuronId {
    int rank{ -1 }; // MPI rank of the owner
    size_t neuron_id{ Constants::uninitialized }; // Neuron id on the owner

public:
    /**
     * @brief Constructs a new RankNeuronId with invalid rank and id
     */
    RankNeuronId() = default;

    /**
     * @brief Constructs a new RankNeuronId with specified inputs (not validated)
     * @param rank The MPI rank
     * @param neuron_id The neuron id
     */
    RankNeuronId(int rank, size_t neuron_id) noexcept
        : rank(rank)
        , neuron_id(neuron_id) {
    }

    /** 
     * @brief Returns the associated MPI rank
     * @return The MPI rank
     * @exception Throws a RelearnException if the rank is negative
     */
    [[nodiscard]] int get_rank() const {
        RelearnException::check(rank >= 0, "RankNeuronId::get_rank, it was: %d", rank);
        //const auto num_ranks = MPIWrapper::get_num_ranks();
        //RelearnException::check(rank < num_ranks, "RankNeuronId::get_rank, it was %d but the number of ranks was only %d%", rank, num_ranks);
        return rank;
    }

    /**
     * @brief Returns the associated neuron id
     * @return The neuron id
     * @exception Throws a RelearnException if the id is not smaller than Constants::uninitialized
     */
    [[nodiscard]] size_t get_neuron_id() const {
        RelearnException::check(neuron_id < Constants::uninitialized, "RankNeuronId::get_neuron_id, it was: %u", Constants::uninitialized);
        return neuron_id;
    }

    /**
     * @brief Compares two objects by rank and id
     * @param other The other RankNeuronId
     * @return True iff both ranks and both neuron ids are equal
     */
    bool operator==(const RankNeuronId& other) const noexcept {
        return (this->rank == other.rank && this->neuron_id == other.neuron_id);
    }

    /**
     * @brief Compares two objects by rank and id
     * @param other The other RankNeuronId
     * @return False iff both ranks and both neuron ids are equal
     */
    bool operator!=(const RankNeuronId& other) const noexcept {
        return !(*this == other);
    }

    /**
     * @brief Prints the object's rank and id; inserts \n
     * @param os The out-stream in which the object is printed
     * @return The argument os to allow chaining
     */
    friend std::ostream& operator<<(std::ostream& os, const RankNeuronId& rni) {
        os << "Rank: " << rni.get_rank() << "\t id: " << rni.get_neuron_id() << "\n";
        return os;
    }
};
