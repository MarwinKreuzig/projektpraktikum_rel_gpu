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

#include "algorithm/Internal/ExchangingAlgorithm.h"

#include "algorithm/FMMInternal/FastMultipoleMethodCell.h"
#include "mpi/CommunicationMap.h"
#include "enums/UpdateStatus.h"
#include "neurons/helper/SynapseCreationRequests.h"

#include <memory>
#include <utility>
#include <vector>

class NeuronsExtraInfo;
template <typename T>
class OctreeImplementation;

/**
 * This class represents the implementation and adaptation of fast multipole method. The parameters can be set on the fly.
 * It is strongly tied to Octree, and might perform MPI communication via NodeCache::download_children()
 */
class FastMultipoleMethod : public ForwardAlgorithm<SynapseCreationRequest, SynapseCreationResponse, FastMultipoleMethodCell> {
    friend class FMMTest;

public:
    using AdditionalCellAttributes = FastMultipoleMethodCell;
    using number_neurons_type = RelearnTypes::number_neurons_type;

    /**
     * @brief Constructs a new instance with the given octree
     * @param octree The octree on which the algorithm is to be performed, not null
     * @exception Throws a RelearnException if octree is nullptr
     */
    explicit FastMultipoleMethod(const std::shared_ptr<OctreeImplementation<FastMultipoleMethodCell>>& octree)
        : ForwardAlgorithm(octree) { }

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    void record_memory_footprint(const std::unique_ptr<MemoryFootprint>& footprint) override {
        const auto my_footprint = sizeof(*this) - sizeof(ForwardAlgorithm<SynapseCreationRequest, SynapseCreationResponse, FastMultipoleMethodCell>);
        footprint->emplace("FastMultipoleMethod", my_footprint);

        ForwardAlgorithm<SynapseCreationRequest, SynapseCreationResponse, FastMultipoleMethodCell>::record_memory_footprint(footprint);
    }

protected:
    /**
     * @brief Returns a collection of proposed synapse creations for each neuron with vacant axons.
     * @param number_neurons The number of local neurons
     * @exception Can throw a RelearnException
     * @return Returns a map, indicating for every MPI rank all requests that are made from this rank.
     */
    CommunicationMap<SynapseCreationRequest> find_target_neurons(number_neurons_type number_neurons) override;

    /**
     * @brief Processes all incoming requests from the MPI ranks locally, and prepares the responses
     * @param creation_requests The requests from all MPI ranks
     * @exception Can throw a RelearnException
     * @return A pair of (1) The responses to each request and (2) another pair of (a) all local synapses and (b) all distant synapses to the local rank
     */
    [[nodiscard]] std::pair<CommunicationMap<SynapseCreationResponse>, std::pair<PlasticLocalSynapses, PlasticDistantInSynapses>>
    process_requests(const CommunicationMap<SynapseCreationRequest>& creation_requests) override;

    /**
     * @brief Processes all incoming responses from the MPI ranks locally
     * @param creation_requests The requests from this MPI rank
     * @param creation_responses The responses from the other MPI ranks
     * @exception Can throw a RelearnException
     * @return All synapses from this MPI rank to other MPI ranks
     */
    [[nodiscard]] PlasticDistantOutSynapses process_responses(const CommunicationMap<SynapseCreationRequest>& creation_requests,
        const CommunicationMap<SynapseCreationResponse>& creation_responses) override;
};
