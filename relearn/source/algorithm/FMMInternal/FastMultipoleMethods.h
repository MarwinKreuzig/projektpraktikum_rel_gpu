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

#include "algorithm/FMMInternal/FastMultipoleMethodsCell.h"
#include "mpi/CommunicationMap.h"
#include "neurons/UpdateStatus.h"
#include "neurons/helper/SynapseCreationRequests.h"

#include <memory>
#include <utility>
#include <vector>

class NeuronsExtraInfo;
template <typename T>
class OctreeImplementation;

/**
 * This class represents the implementation and adaptation of fast multipole methods. The parameters can be set on the fly.
 * It is strongly tied to Octree, and might perform MPI communication via NodeCache::download_children()
 */
class FastMultipoleMethods : public ForwardAlgorithm<SynapseCreationRequest, SynapseCreationResponse, FastMultipoleMethodsCell> {
    friend class FMMTest;

public:
    using AdditionalCellAttributes = FastMultipoleMethodsCell;
    using number_neurons_type = RelearnTypes::number_neurons_type;

    /**
     * @brief Constructs a new instance with the given octree
     * @param octree The octree on which the algorithm is to be performed, not null
     * @exception Throws a RelearnException if octree is nullptr
     */
    explicit FastMultipoleMethods(const std::shared_ptr<OctreeImplementation<FastMultipoleMethodsCell>>& octree)
        : ForwardAlgorithm(octree) { }

protected:
    /**
     * @brief Returns a collection of proposed synapse creations for each neuron with vacant axons.
     * @param number_neurons The number of local neurons
     * @param disable_flags Flags that indicate if a local neuron is disabled. If so, the neuron is ignored
     * @param extra_infos Used to access the positions of the local neurons
     * @exception Can throw a RelearnException
     * @return Returns a map, indicating for every MPI rank all requests that are made from this rank.
     */
    CommunicationMap<SynapseCreationRequest> find_target_neurons(number_neurons_type number_neurons, const std::vector<UpdateStatus>& disable_flags,
        const std::unique_ptr<NeuronsExtraInfo>& extra_infos) override;

    /**
     * @brief Processes all incoming requests from the MPI ranks locally, and prepares the responses
     * @param creation_requests The requests from all MPI ranks
     * @exception Can throw a RelearnException
     * @return A pair of (1) The responses to each request and (2) another pair of (a) all local synapses and (b) all distant synapses to the local rank
     */
    [[nodiscard]] std::pair<CommunicationMap<SynapseCreationResponse>, std::pair<LocalSynapses, DistantInSynapses>>
    process_requests(const CommunicationMap<SynapseCreationRequest>& creation_requests) override;

    /**
     * @brief Processes all incoming responses from the MPI ranks locally
     * @param creation_requests The requests from this MPI rank
     * @param creation_responses The responses from the other MPI ranks
     * @exception Can throw a RelearnException
     * @return All synapses from this MPI rank to other MPI ranks
     */
    [[nodiscard]] DistantOutSynapses process_responses(const CommunicationMap<SynapseCreationRequest>& creation_requests,
        const CommunicationMap<SynapseCreationResponse>& creation_responses) override;
};
