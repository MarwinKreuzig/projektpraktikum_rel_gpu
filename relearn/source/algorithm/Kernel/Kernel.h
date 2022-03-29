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
#include "neurons/ElementType.h"
#include "neurons/SignalType.h"
#include "structure/OctreeNode.h"
#include "util/RelearnException.h"
#include "util/TaggedID.h"
#include "util/Vec3.h"

#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

template <typename AdditionalCellAttributes, typename KernelType>
class Kernel {
public:
    using counter_type = RelearnTypes::counter_type;
    using position_type = RelearnTypes::position_type;

    /**
     * @brief Calculates the probability for the source neuron to connect to each of the OctreeNodes in the vector,
     *      searching the specified element_type and signal_type
     * @param source_neuron_id The id of the source neuron, is used to prevent autapses
     * @param source_position The position of the source neuron
     * @param nodes All nodes from which the source neuron can pick
     * @param element_type The element type the source neuron searches
     * @param signal_type The signal type the source neuron searches
     * @param sigma The probability parameter for the calculation
     * @exception Throws a RelearnException if one of the pointer in nodes is a nullptr, or if KernelType::calculate_attractiveness_to_connect throws
     * @return A pair of (a) the total probability of all targets and (b) the respective probability of each target
     */
    [[nodiscard]] static std::pair<double, std::vector<double>> create_probability_interval(const NeuronID& source_neuron_id, const position_type& source_position,
        const std::vector<OctreeNode<AdditionalCellAttributes>*>& nodes, const ElementType element_type, const SignalType signal_type, const double sigma) {

        if (nodes.empty()) {
            return { 0.0, {} };
        }

        double sum = 0.0;

        std::vector<double> probabilities{};
        probabilities.reserve(nodes.size());

        std::transform(nodes.begin(), nodes.cend(), std::back_inserter(probabilities), [&](const OctreeNode<AdditionalCellAttributes>* target_node) {
            RelearnException::check(target_node != nullptr, "BarnesHut::update_leaf_nodes: target_node was nullptr");
            const auto prob = KernelType::calculate_attractiveness_to_connect(source_neuron_id, source_position, target_node, element_type, signal_type, sigma);
            sum += prob;
            return prob;
        });

        // Short-cut an empty vector here for later uses
        if (sum == 0.0) {
            return { 0.0, {} };
        }

        return { sum, std::move(probabilities) };
    }
};