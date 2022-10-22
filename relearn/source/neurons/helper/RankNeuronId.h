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

#include "Config.h"
#include "mpi/MPIWrapper.h"
#include "util/RelearnException.h"
#include "util/TaggedID.h"

#include <compare>
#include <ostream>
#include <utility>

/**
 * Identifies a neuron by the MPI rank of its owner
 * and its neuron id on the owner, i.e., the pair <rank, neuron_id>
 */
class RankNeuronId {
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
    RankNeuronId(const int rank, const NeuronID neuron_id) noexcept
        : rank(rank)
        , neuron_id(neuron_id) {
    }

    /**
     * @brief Returns the associated MPI rank
     * @return The MPI rank
     * @exception Throws a RelearnException if the rank is negative
     */
    [[nodiscard]] int get_rank() const {
        RelearnException::check(rank >= 0, "RankNeuronId::get_rank: It was negative: {}", rank);
        return rank;
    }

    /**
     * @brief Returns the associated neuron id
     * @return The neuron id
     * @exception Throws a RelearnException if the id is not smaller than Constants::uninitialized
     */
    [[nodiscard]] NeuronID get_neuron_id() const {
        RelearnException::check(neuron_id.is_initialized(), "RankNeuronId::get_neuron_id: neuron_id is not initialized");
        return neuron_id;
    }

    [[nodiscard]] friend constexpr std::strong_ordering operator<=>(const RankNeuronId& first, const RankNeuronId& second) noexcept = default;

    /**
     * @brief Prints the object's rank and id; inserts \n
     * @param os The out-stream in which the object is printed
     * @return The argument os to allow chaining
     */
    friend std::ostream& operator<<(std::ostream& os, const RankNeuronId& rni) {
        os << "Rank: " << rni.get_rank() << "\t id: " << rni.get_neuron_id() << '\n';
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

private:
    int rank{ -1 }; // MPI rank of the owner
    NeuronID neuron_id{ NeuronID::uninitialized_id() }; // Neuron id on the owner
};

namespace std {
template <>
struct tuple_size<::RankNeuronId> {
    static constexpr size_t value = 2;
};

template <>
struct tuple_element<0, ::RankNeuronId> {
    using type = int;
};

template <>
struct tuple_element<1, ::RankNeuronId> {
    using type = NeuronID;
};

} // namespace std
