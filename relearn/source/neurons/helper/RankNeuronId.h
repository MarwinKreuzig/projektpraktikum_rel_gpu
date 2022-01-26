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
#include "../../util/TaggedID.h"

#include <ostream>
#include <utility>

/**
 * Identifies a neuron by the MPI rank of its owner
 * and its neuron id on the owner, i.e., the pair <rank, neuron_id>
 */
class RankNeuronId {
public:
    using rank_type = int;
    using neuron_id_type = NeuronID;

private:
    rank_type rank{ -1 }; // MPI rank of the owner
    NeuronID neuron_id{ NeuronID::uninitialized_id() }; // Neuron id on the owner

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
    RankNeuronId(const rank_type rank, const neuron_id_type neuron_id) noexcept
        : rank(rank)
        , neuron_id(neuron_id) {
    }

    /**
     * @brief Returns the associated MPI rank
     * @return The MPI rank
     * @exception Throws a RelearnException if the rank is negative
     */
    [[nodiscard]] rank_type get_rank() const {
        RelearnException::check(rank >= 0, "RankNeuronId::get_rank: It was negative: {}", rank);
        return rank;
    }

    /**
     * @brief Returns the associated neuron id
     * @return The neuron id
     * @exception Throws a RelearnException if the id is not smaller than Constants::uninitialized
     */
    [[nodiscard]] neuron_id_type get_neuron_id() const {
        RelearnException::check(neuron_id.is_initialized(), "RankNeuronId::get_neuron_id: neuron_id is not initialized");
        return neuron_id;
    }

    /**
     * @brief Compares two objects by rank and id
     * @param other The other RankNeuronId
     * @return True iff both ranks and both neuron ids are equal
     */
    [[nodiscard]] bool operator==(const RankNeuronId& other) const noexcept {
        return (this->rank == other.rank && this->neuron_id == other.neuron_id);
    }

    /**
     * @brief Compares two objects by rank and id
     * @param other The other RankNeuronId
     * @return False iff both ranks and both neuron ids are equal
     */
    [[nodiscard]] bool operator!=(const RankNeuronId& other) const noexcept {
        return !(*this == other);
    }

    /**
     * @brief Compares two objects first by rank and then by id
     * @param other The other RankNeuronId
     * @return True iff this' rank is smaller than the other's or if the ranks are equal and this' neuron_id is smaller
     */
    [[nodiscard]] bool operator<(const RankNeuronId& other) const noexcept {
        return (this->rank < other.rank) || (this->rank == other.rank && this->neuron_id < other.neuron_id);
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

    template <std::size_t Index>
    [[nodiscard]] auto& get() & {
        if constexpr (Index == 0) {
            return rank;
        }
        if constexpr (Index == 1) {
            return neuron_id;
        }
    }

    template <std::size_t Index>
    [[nodiscard]] auto const& get() const& {
        if constexpr (Index == 0) {
            return rank;
        }
        if constexpr (Index == 1) {
            return neuron_id;
        }
    }

    template <std::size_t Index>
    [[nodiscard]] auto&& get() && {
        if constexpr (Index == 0) {
            return rank;
        }
        if constexpr (Index == 1) {
            return neuron_id;
        }
    }
};

namespace std {
template <>
struct tuple_size<::RankNeuronId> {
    static constexpr size_t value = 2;
};

template <>
struct tuple_element<0, ::RankNeuronId> {
    using type = RankNeuronId::rank_type;
};

template <>
struct tuple_element<1, ::RankNeuronId> {
    using type = RankNeuronId::neuron_id_type;
};

} // namespace std
