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

#include "mpi/CommunicationMap.h"
#include "neurons/enums/ElementType.h"
#include "neurons/enums/SignalType.h"
#include "neurons/helper/RankNeuronId.h"
#include "neurons/helper/SynapseDeletionRequests.h"
#include "util/TaggedID.h"

#include <memory>
#include <vector>

class NetworkGraph;
class NeuronsExtraInfo;
class SynapticElements;

class SynapseDeletionFinder {
public:
    void set_network_graph(std::shared_ptr<NetworkGraph> ng) noexcept {
        network_graph = std::move(ng);
    }

    void set_axons(std::shared_ptr<SynapticElements> se) noexcept {
        axons = std::move(se);
    }

    void set_dendrites_ex(std::shared_ptr<SynapticElements> se) noexcept {
        excitatory_dendrites = std::move(se);
    }

    void set_dendrites_in(std::shared_ptr<SynapticElements> se) noexcept {
        inhibitory_dendrites = std::move(se);
    }

    void set_extra_infos(std::shared_ptr<NeuronsExtraInfo> new_extra_info) {
        extra_info = std::move(new_extra_info);
    }

    [[nodiscard]] std::pair<std::uint64_t, std::uint64_t> delete_synapses();

protected:
    [[nodiscard]] CommunicationMap<SynapseDeletionRequest> find_synapses_to_delete(const std::shared_ptr<SynapticElements>& synaptic_elements, const std::pair<unsigned int, std::vector<unsigned int>>& to_delete);

    [[nodiscard]] std::uint64_t commit_deletions(const CommunicationMap<SynapseDeletionRequest>& list, MPIRank my_rank);

    [[nodiscard]] virtual std::vector<RankNeuronId> find_synapses_on_neuron(NeuronID neuron_id, ElementType element_type, SignalType signal_type, unsigned int num_synapses_to_delete) = 0;

    std::shared_ptr<SynapticElements> axons{};
    std::shared_ptr<SynapticElements> excitatory_dendrites{};
    std::shared_ptr<SynapticElements> inhibitory_dendrites{};

    std::shared_ptr<NetworkGraph> network_graph{};
    std::shared_ptr<NeuronsExtraInfo> extra_info{};
};

class RandomSynapseDeletionFinder : public SynapseDeletionFinder {
public:
    [[nodiscard]] std::vector<RankNeuronId> find_synapses_on_neuron(NeuronID neuron_id, ElementType element_type, SignalType signal_type, unsigned int num_synapses_to_delete) override;
};
