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
#include "algorithm/Kernel/Gamma.h"
#include "algorithm/Kernel/Gaussian.h"
#include "algorithm/Kernel/Linear.h"
#include "algorithm/Kernel/Weibull.h"
#include "neurons/ElementType.h"
#include "neurons/SignalType.h"
#include "structure/OctreeNode.h"
#include "util/Random.h"
#include "util/RelearnException.h"
#include "util/TaggedID.h"
#include "util/Vec3.h"

#include "fmt/ostream.h"

#include <algorithm>
#include <numeric>
#include <ostream>
#include <utility>
#include <vector>

/**
 * This enum reflects the different probabilty kernels, it must
 * be kept in sync with the classes to allow a seemless integration
 */
enum class KernelType {
    Gaussian,
    Linear,
    Gamma,
    Weibull
};

/**
 * @brief Pretty-prints the kernel type to the chosen stream
 * @param out The stream to which to print the kernel type
 * @param kernel_type The kernel type to print
 * @return The argument out, now altered with the kernel type
 */
inline std::ostream& operator<<(std::ostream& out, const KernelType& kernel_type) {
    switch (kernel_type) {
    case KernelType::Gamma:
        return out << "Gamma";
    case KernelType::Gaussian:
        return out << "Gaussian";
    case KernelType::Linear:
        return out << "Linear";
    case KernelType::Weibull:
        return out << "Weibull";
    }

    return out << "UNKNOWN";
}

template <>
struct fmt::formatter<KernelType> : ostream_formatter { };

/**
 * This class encapsulates the necessary probability kernels that determine how
 * likely it is that two neurons form a synapse.
 * It handles different use cases, depending on where in the pipeline it is inserted.
 * @tparam AdditionalCellAttributes The AdditionalCellAttributes for the OctreeNode.
 */
template <typename AdditionalCellAttributes>
class Kernel {
public:
    using counter_type = RelearnTypes::counter_type;
    using position_type = RelearnTypes::position_type;

    /**
     * @brief Sets the type of kernel that shall be used for calculations
     * @param kernel_type The kernel type that from now on shall be used
     */
    static void set_kernel_type(const KernelType kernel_type) noexcept {
        currently_used_kernel = kernel_type;
    }

    /**
     * @brief Returns the currently used kernel type
     * @return The currently used kernel type
     */
    [[nodiscard]] KernelType get_kernel_type() noexcept {
        return currently_used_kernel;
    }

    /**
     * @brief Calculates the attractiveness to connect on the basis of the set kernel type.
     *      Performs all necessary checks and passes the values to the actual kernel.
     * @param source_neuron_id The source neuron id
     * @param source_position The source position s
     * @param target_node The target node
     * @param element_type The element type
     * @param signal_type The signal type
     * @exception Throws a RelearnException if the position for (element_type, signal_type) from target_node is empty or not supported
     * @return The calculated attractiveness, might be 0.0 to avoid autapses
     */
    [[nodiscard]] static double calculate_attractiveness_to_connect(const NeuronID& source_neuron_id, const position_type& source_position,
        const OctreeNode<AdditionalCellAttributes>* target_node, const ElementType element_type, const SignalType signal_type) {
        // A neuron must not form an autapse, i.e., a synapse to itself
        if (target_node->is_leaf() && source_neuron_id == target_node->get_cell_neuron_id()) {
            return 0.0;
        }

        const auto& cell = target_node->get_cell();
        const auto& target_position = cell.get_position_for(element_type, signal_type);
        const auto& number_elements = cell.get_number_elements_for(element_type, signal_type);

        RelearnException::check(target_position.has_value(), "Kernel::calculate_attractiveness_to_connect: target_position is bad");

        switch (currently_used_kernel) {
        case KernelType::Gamma:
            return GammaDistributionKernel::calculate_attractiveness_to_connect(source_position, target_position.value(), number_elements);
        case KernelType::Gaussian:
            return GaussianDistributionKernel::calculate_attractiveness_to_connect(source_position, target_position.value(), number_elements);
        case KernelType::Linear:
            return LinearDistributionKernel::calculate_attractiveness_to_connect(source_position, target_position.value(), number_elements);
        case KernelType::Weibull:
            return WeibullDistributionKernel::calculate_attractiveness_to_connect(source_position, target_position.value(), number_elements);
        }

        RelearnException::fail("Kernel::calculate_attractiveness_to_connect: {} is an unknown kernel type!", currently_used_kernel);

        return 0.0;
    }

    /**
     * @brief Calculates the probability for the source neuron to connect to each of the OctreeNodes in the vector,
     *      searching the specified element_type and signal_type
     * @param source_neuron_id The id of the source neuron, is used to prevent autapses
     * @param source_position The position of the source neuron
     * @param nodes All nodes from which the source neuron can pick
     * @param element_type The element type the source neuron searches
     * @param signal_type The signal type the source neuron searches
     * @exception Throws a RelearnException if one of the pointer in nodes is a nullptr, or if KernelType::calculate_attractiveness_to_connect throws
     * @return A pair of (a) the total probability of all targets and (b) the respective probability of each target
     */
    [[nodiscard]] static std::pair<double, std::vector<double>> create_probability_interval(const NeuronID& source_neuron_id, const position_type& source_position,
        const std::vector<OctreeNode<AdditionalCellAttributes>*>& nodes, const ElementType element_type, const SignalType signal_type) {

        if (nodes.empty()) {
            return { 0.0, {} };
        }

        double sum = 0.0;

        std::vector<double> probabilities{};
        probabilities.reserve(nodes.size());

        std::transform(nodes.begin(), nodes.cend(), std::back_inserter(probabilities), [&](const OctreeNode<AdditionalCellAttributes>* target_node) {
            RelearnException::check(target_node != nullptr, "Kernel::create_probability_interval: target_node was nullptr");
            const auto prob = calculate_attractiveness_to_connect(source_neuron_id, source_position, target_node, element_type, signal_type);
            sum += prob;
            return prob;
        });

        // Short-cut an empty vector here for later uses
        if (sum == 0.0) {
            return { 0.0, {} };
        }

        return { sum, std::move(probabilities) };
    }

    /**
     * @brief Picks a target based on the supplied probabilities
     * @param nodes The target nodes, must not be empty
     * @param probabilities The associated probabilities to the target nodes, must be as large as nodes
     * @param random_number The random number that determines which target node to pick, must be >= 0.0
     * @exception Throws a RelearnException if there are no nodes, if the sizes of the vectors don't match,
     *      if random_number is < 0.0, or if the selected node is nullptr
     * @return The selected target node
     */
    [[nodiscard]] static OctreeNode<AdditionalCellAttributes>* pick_target(const std::vector<OctreeNode<AdditionalCellAttributes>*>& nodes,
        const std::vector<double>& probabilities, const double random_number) {
        RelearnException::check(nodes.size() == probabilities.size(), "Kernel::pick_target: Had a different number of probabilities than nodes: {} vs {}", nodes.size(), probabilities.size());
        RelearnException::check(!nodes.empty(), "Kernel::pick_target: There were no nodes to pick from");
        RelearnException::check(random_number >= 0.0, "Kernel::pick_target: random_number was smaller than 0.0");

        if (!(0.0 < random_number)) {
            auto* node_selected = nodes[0];
            RelearnException::check(node_selected != nullptr, "Kernel::pick_target: node_selected was nullptr");
            return node_selected;
        }

        auto counter = 0;
        for (auto sum_probabilities = 0.0; counter < probabilities.size() && sum_probabilities < random_number; counter++) {
            sum_probabilities += probabilities[counter];
        }

        auto* node_selected = nodes[counter - 1ULL];

        RelearnException::check(node_selected != nullptr, "Kernel::pick_target: node_selected was nullptr");

        return node_selected;
    }

    /**
     * @brief Picks a target based on the the KernelType
     * @param source_neuron_id The id of the source neuron, is used to prevent autapses
     * @param source_position The position of the source neuron
     * @param nodes The target nodes, must not be empty
     * @param element_type The element type the source neuron searches
     * @param signal_type The signal type the source neuron searches
     * @exception Throws a RelearnException if one of the pointer in nodes is a nullptr, or if KernelType::calculate_attractiveness_to_connect throws
     * @return The selected target node, is nullptr if nodes.empty()
     */
    [[nodiscard]] static OctreeNode<AdditionalCellAttributes>* pick_target(const NeuronID& source_neuron_id, const position_type& source_position,
        const std::vector<OctreeNode<AdditionalCellAttributes>*>& nodes, const ElementType element_type, const SignalType signal_type) {
        if (nodes.empty()) {
            return nullptr;
        }

        /**
         * Assign a probability to each node in the vector.
         * The probability for connecting to the same neuron (i.e., the axon's neuron) is set 0.
         */
        const auto& [total_probability, all_probabilities]
            = create_probability_interval(source_neuron_id, source_position, nodes, element_type, signal_type);

        // Short cut to avoid exceptions later on
        if (total_probability == 0.0) {
            return nullptr;
        }

        const auto random_number = RandomHolder::get_random_uniform_double(RandomHolderKey::Algorithm, 0.0, 
            std::nextafter(total_probability, total_probability + Constants::eps));

        auto* node_selected = pick_target(nodes, all_probabilities, random_number);
        return node_selected;
    }

private:
    static inline KernelType currently_used_kernel{ KernelType::Gaussian };
};