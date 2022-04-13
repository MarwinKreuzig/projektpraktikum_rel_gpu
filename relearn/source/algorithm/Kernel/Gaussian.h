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
#include "Types.h"
#include "neurons/ElementType.h"
#include "neurons/SignalType.h"
#include "structure/OctreeNode.h"
#include "util/RelearnException.h"
#include "util/TaggedID.h"
#include "util/Vec3.h"

#include <numeric>

class GaussianKernel {
public:
    using counter_type = RelearnTypes::counter_type;
    using position_type = RelearnTypes::position_type;

    /**
     * @brief Sets probability parameter used to determine the probability for a cell of being selected
     * @param sigma The probability parameter, > 0.0
     * @exception Throws a RelearnExeption if sigma <= 0.0
     */
    static void set_probability_parameter(const double sigma) {
        RelearnException::check(sigma > 0.0, "In GaussianKernel::set_probability_parameter, sigma was not greater than 0.0");
        GaussianKernel::sigma = sigma;
    }

    /**
     * @brief Returns the currently used probability parameter
     * @return The currently used probability parameter
     */
    [[nodiscard]] static double get_probabilty_parameter() noexcept {
        return sigma;
    }

    /**
     * @brief Calculates the attractiveness to connect on the basis of
     *      k * exp(-||s - t||_2^2 / sigma^2)
     * @param source_position The source position s
     * @param target_position The target position t
     * @param number_free_elements The linear scaling factor k
     * @return The calculated attractiveness
     */
    [[nodiscard]] static double calculate_attractiveness_to_connect(const position_type& source_position, const position_type& target_position,
        const counter_type& number_free_elements) noexcept {
        if (number_free_elements == 0) {
            return 0.0;
        }

        const auto position_diff = target_position - source_position;
        const auto numerator = position_diff.calculate_squared_2_norm();

        const auto squared_sigma = sigma * sigma;
        const auto exponent = -numerator / squared_sigma;

        // Criterion from Markus' paper with doi: 10.3389/fnsyn.2014.00007
        const auto exp_val = std::exp(exponent);
        const auto ret_val = number_free_elements * exp_val;

        return ret_val;
    }

    /**
     * @brief Calculates the attractiveness to connect on the basis of
     *      k * exp(-||s - t||_2^2 / sigma^2)
     *      where t is the position of the target_node for the element_type and signal_type
     *      and k is the number of free elements of the target_node for the element_type and signal_type.
     *      It also prevents autapses by comparing the ids (returns 0.0 if they are equal) and checks for algorithmic errors
     * @param source_neuron_id The source neuron id
     * @param source_position The source position s
     * @param target_node The target node
     * @param element_type The element type
     * @param signal_type The signal type
     * @tparam AdditionalCellAttributes The additional cell attributes, doesn't affect the functionality of this method
     * @exception Throws a RelearnException if the position for (element_type, signal_type) from target_node is empty or not supported
     * @return The calculated attractiveness, might be 0.0 to avoid autapses
     */
    template <typename AdditionalCellAttributes>
    [[nodiscard]] static double calculate_attractiveness_to_connect(const NeuronID& source_neuron_id, const position_type& source_position,
        const OctreeNode<AdditionalCellAttributes>* target_node, const ElementType element_type, const SignalType signal_type) {
        // A neuron must not form an autapse, i.e., a synapse to itself
        if (target_node->is_child() && source_neuron_id == target_node->get_cell_neuron_id()) {
            return 0.0;
        }

        const auto& cell = target_node->get_cell();
        const auto& target_position = cell.get_position_for(element_type, signal_type);
        const auto& number_elements = cell.get_number_elements_for(element_type, signal_type);

        RelearnException::check(target_position.has_value(), "GaussianKernel::calculate_attractiveness_to_connect: target_position is bad");

        return calculate_attractiveness_to_connect(source_position, target_position.value(), number_elements);
    }

private:
    static inline double sigma{ Constants::default_sigma };
};