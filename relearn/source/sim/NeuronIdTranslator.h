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

#include "../neurons/helper/RankNeuronId.h"
#include "../util/TaggedID.h"

#include <map>
#include <memory>
#include <vector>

/**
 * This class offers a way to translate neuron ids:
 * Local id to global id and vice versa, also to and from RankNeuronId
 */
class NeuronIdTranslator {
public:
    /**
     * @brief Creates a new translator
    */
    NeuronIdTranslator() = default;
    virtual ~NeuronIdTranslator() = default;

    /**
     * @brief Checks if a global_id belongs to the current MPI rank
     * @param global_id The global neuron id 
     * @return True iff the global neuron id belongs to the current MPI rank
     */
    [[nodiscard]] virtual bool is_neuron_local(NeuronID global_id) const = 0;

    /**
     * @brief Translated the global neuron id to the local neuron id
     * @param global_id The global neuron id
     * @return The local neuron id
     */
    [[nodiscard]] virtual NeuronID get_local_id(NeuronID global_id) const = 0;

    /**
     * @brief Translated the local neuron id to the global neuron id
     * @param global_id The local neuron id
     * @return The global neuron id
     */
    [[nodiscard]] virtual NeuronID get_global_id(NeuronID local_id) const = 0;

    /**
     * @brief Translated a bunch of global neuron ids to RankNeuronIds
     * @param global_ids The global neuron ids
     * @return A translation map from global neuron id to RankNeuronId
     */
    [[nodiscard]] virtual std::map<NeuronID, RankNeuronId> translate_global_ids(const std::vector<NeuronID>& global_ids) = 0;

    /**
     * @brief Translated a RankNeuronId to a global neuron id 
     * @param rni The rank neuron id
     * @return The global neuron id of that neuron
     */
    [[nodiscard]] virtual NeuronID translate_rank_neuron_id(const RankNeuronId& rni) = 0;

    /**
     * @brief Translated a bunch of RankNeuronIds to global neuron ids
     * @param ids The RankNeuronIds
     * @return A translation map from RankNeuronId to global neuron id
     */
    [[nodiscard]] virtual std::map<RankNeuronId, NeuronID> translate_rank_neuron_ids(const std::vector<RankNeuronId>& ids) = 0;
};

