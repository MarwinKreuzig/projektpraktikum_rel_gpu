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
#include "algorithm/FMMInternal/FastMultipoleMethodsCell.h"
#include "algorithm/Kernel/Gaussian.h"
#include "neurons/helper/SynapseCreationRequests.h"
#include "neurons/enums/ElementType.h"
#include "neurons/enums/SignalType.h"
#include "structure/NodeCache.h"
#include "structure/OctreeNode.h"
#include "util/Random.h"
#include "util/RelearnException.h"
#include "util/Stack.h"
#include "util/Timers.h"
#include "util/Utility.h"
#include "util/Vec3.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <ostream>
#include <vector>

enum class CalculationType { Direct,
    Hermite,
    Taylor,
};

inline std::ostream& operator<<(std::ostream& out, const CalculationType& calc_type) {
    if (calc_type == CalculationType::Direct) {
        return out << "Direct";
    }

    if (calc_type == CalculationType::Hermite) {
        return out << "Hermite";
    }

    return out << "Taylor";
}

/**
 * This class represents a mathematical three-dimensional multi-index, which is required for the
 * series expansions and coefficient calculations.
 */
class Multiindex {
    friend class FMMTest;

public:
    /**
     * @brief Returns the number of all three-dimensional indices that the multi-index has. This depends on the selected p.
     * @return Returns the number of all indices.
     */
    static constexpr unsigned int get_number_of_indices() noexcept {
        return Constants::p3;
    }

    /**
     * @brief Returns the multi-index as a matrix with the dimensions (p^3, 3).
     * @return Returns a array of arrays wich represents the corresponding multi-index.
     */
    static constexpr std::array<Vec3u, Constants::p3> get_indices() noexcept {
        std::array<Vec3u, Constants::p3> result{};

        auto index = 0U;
        for (auto i = 0U; i < Constants::p; i++) {
            for (auto j = 0U; j < Constants::p; j++) {
                for (auto k = 0U; k < Constants::p; k++) {
                    // NOLINTNEXTLINE
                    result[index] = { i, j, k };
                    index++;
                }
            }
        }

        return result;
    }
};

/**
 * This class provides all computational elements of the Fast-Multipole-Method algorithm.
 * It purely calculates things, but does not change any state.
 * @tparam AdditionalCellAttributes The cell attributes that are
 */
class FastMultipoleMethodsBase {
public:
    using AdditionalCellAttributes = FastMultipoleMethodsCell;
    using interaction_list_type = std::vector<OctreeNode<AdditionalCellAttributes>*>;
    using position_type = typename Cell<AdditionalCellAttributes>::position_type;
    using counter_type = typename Cell<AdditionalCellAttributes>::counter_type;
    using node_pair = std::array<OctreeNode<AdditionalCellAttributes>*, 2>;

    struct stack_entry {
        OctreeNode<AdditionalCellAttributes>* source;
        OctreeNode<AdditionalCellAttributes>* target;
        bool unpacked = { false };
    };

    /**
     * @brief Calculates the n-th Hermite function at the point t
     * @param n Order of the Hermite function.
     * @param t Point of evaluation.
     * @return Value of the Hermite function of the n-th order at the point t.
     */
    static double h(unsigned int n, double t) {
        const auto t_squared = t * t;

        const auto fac_1 = std::exp(-t_squared);
        const auto fac_2 = std::hermite(n, t);

        const auto product = fac_1 * fac_2;

        return product;
    }

    /**
     * @brief Calculates the Hermite function for a multi index and a 3D vector.
     * @param multi_index A tuple of three natural numbers.
     * @param vector A 3D vector.
     * @return Value of the Hermite function.
     */
    static double h_multiindex(const Vec3u& multi_index, const Vec3d& vector) {
        const auto h1 = h(multi_index.get_x(), vector.get_x());
        const auto h2 = h(multi_index.get_y(), vector.get_y());
        const auto h3 = h(multi_index.get_z(), vector.get_z());

        const auto h_total = h1 * h2 * h3;

        return h_total;
    }

    /**
     * @brief The Kernel from Butz&Ooyen "A Simple Rule for Dendritic Spine and Axonal Bouton Formation Can Account for Cortical Reorganization afterFocal Retinal Lesions"
     *       Calculates the attraction between two neurons, where a and b represent the position in three-dimensional space
     * @param a 3D position of the first neuron.
     * @param b 3D position of the second neuron.
     * @param sigma scaling parameter.
     * @return Returns the attraction between the two neurons.
     */
    static double kernel(const Vec3d& a, const Vec3d& b, const double sigma) {
        const auto diff = a - b;
        const auto squared_norm = diff.calculate_squared_2_norm();

        return std::exp(-squared_norm / (sigma * sigma));
    }

    /**
     * @brief Randomly selects one of the different target nodes, to which the source node should connect.
     * @param attractiveness Vector in which the attraction forces for different nodes are entered.
     * @return Returns the index of the choosen node.
     */
    static unsigned int choose_interval(const std::vector<double>& attractiveness) {
        const auto random_number = RandomHolder::get_random_uniform_double(RandomHolderKey::Algorithm, 0.0, std::nextafter(1.0, Constants::eps));
        const auto vec_len = attractiveness.size();

        std::vector<double> intervals(vec_len + 1);
        intervals[0] = 0;

        double sum = 0;
        for (int i = 0; i < vec_len; i++) {
            sum = sum + attractiveness[i];
        }

        for (auto i = 1; i < vec_len + 1; i++) {
            intervals[i] = intervals[i - 1ULL] + (attractiveness[i - 1ULL] / sum);
        }

        int i = 0;
        while (random_number > intervals[i + 1ULL] && i <= vec_len) {
            i++;
        }

        if (i >= vec_len + 1) {
            return 0;
        }

        return i;
    }

    /**
     * @brief Counts the elements in an interaction list that are not nullptr.
     * @param arr Interaction list containing OctreeNodes.
     * @return Number of elements unequal to nullptr.
     */
    static unsigned int count_non_zero_elements(const interaction_list_type& arr) noexcept {
        auto non_zero_counter = 0U;
        for (auto i = 0U; i < arr.size(); i++) {
            if (arr[i] != nullptr) {
                non_zero_counter++;
            }
        }
        return non_zero_counter;
    }

    /**
     * @brief Returns the OctreeNode at the given index, nullptr elements are not counted.
     * @param arr Interaction list containing OctreeNodes.
     * @param index Index of the desired node.
     * @return The specified element, can be nullptr if not enough non-nullptr elements are present
     */
    static OctreeNode<AdditionalCellAttributes>* extract_element(const interaction_list_type& arr, const unsigned int index) noexcept {
        auto non_zero_counter = 0U;
        for (auto i = 0U; i < arr.size(); i++) {
            if (arr[i] != nullptr) {
                if (index == non_zero_counter) {
                    return arr[i];
                }
                non_zero_counter++;
            }
        }
        return nullptr;
    }

    /**
     * @brief Checks whether a node is already in the cache and reloads the child nodes if necessary. Sets a children to nullptr, when it has no vacant dendrites.
     * @param node Node which is checked.
     * @exception Throws a RelearnException if node == nullptr or if node->is_parent()
     * @return Interaction list with all children.
     */
    static std::array<OctreeNode<AdditionalCellAttributes>*, Constants::number_oct> get_children_to_array(OctreeNode<AdditionalCellAttributes>* node) {
        RelearnException::check(node != nullptr, "FastMultipoleMethodsBase::get_children_to_array: Node was a nullptr.");
        RelearnException::check(node->is_parent(), "FastMultipoleMethodsBase::get_children_to_array: Node has no children.");

        const auto is_local = node->is_local();
        const auto result = is_local ? node->get_children() : NodeCache<AdditionalCellAttributes>::download_children(node);

        return result;
    }

    /**
     * @brief Returns a vector of all positions of the selected type that have a free port of the requested SignalType.
     * @param node OctreeNode from which the elements are to be counted.
     * @param type Type of synaptic elements (axon or dendrite).
     * @param needed The requested SignalType.
     * @exception Throws a RelearnException if node == nullptr
     * @return A vector of all actual positions.
     */
    static std::vector<std::pair<position_type, counter_type>> get_all_positions_for(OctreeNode<AdditionalCellAttributes>* node, const ElementType element_type, const SignalType signal_type) {
        RelearnException::check(node != nullptr, "FastMultipoleMethodsBase::get_all_positions_for: node is nullptr");

        std::vector<std::pair<position_type, counter_type>> result{};
        result.reserve(30);

        Stack<OctreeNode<AdditionalCellAttributes>*> stack{ 30 };
        stack.emplace_back(node);

        while (!stack.empty()) {
            auto* current_node = stack.pop_back();
            if (current_node == nullptr) {
                continue;
            }

            if (current_node->is_leaf()) {
                // Get number and position, depending on which types were chosen.
                const auto& cell = current_node->get_cell();
                const auto& opt_position = cell.get_position_for(element_type, signal_type);
                RelearnException::check(opt_position.has_value(), "FastMultipoleMethodsBase::get_all_positions_for: opt_position has no value.");

                const auto number_elements = cell.get_number_elements_for(element_type, signal_type);
                result.emplace_back(opt_position.value(), number_elements);
                continue;
            }

            const auto& children = get_children_to_array(current_node);
            for (auto* child : children) {
                if (child == nullptr) {
                    continue;
                }

                if (const auto number_elements = child->get_cell().get_number_elements_for(element_type, signal_type); number_elements == 0) {
                    continue;
                }

                // push children to stack that have relevant elements
                stack.emplace_back(child);
            }
        }

        return result;
    }

    /**
     * @brief Checks which calculation type is suitable for a given source and target node
     * @param source Node with vacant searching elements
     * @param target Node with vacant searched elements
     * @param element_type Specifies which type of synaptic element initiates the search
     * @param signal_type Specifies for which type of neurons the calculation is to be executed (inhibitory or excitatory)
     * @exception Throws a RelearnException if source or target are nullptr
     * @return The calculation type that is appropriate for the forces
     */
    static CalculationType check_calculation_requirements(const OctreeNode<AdditionalCellAttributes>* source, const OctreeNode<AdditionalCellAttributes>* target,
        const ElementType element_type, const SignalType signal_type) {
        RelearnException::check(source != nullptr, "FastMultipoleMethodsBase::check_calculation_requirements: source is nullptr");
        RelearnException::check(target != nullptr, "FastMultipoleMethodsBase::check_calculation_requirements: target is nullptr");

        if (source->is_leaf() || target->is_leaf()) {
            return CalculationType::Direct;
        }

        const auto& source_cell = source->get_cell();
        const auto& target_cell = target->get_cell();

        const auto other_element_type = get_other_element_type(element_type);
        if (target_cell.get_number_elements_for(other_element_type, signal_type) <= Constants::max_neurons_in_target) {
            return CalculationType::Direct;
        }

        if (source_cell.get_number_elements_for(element_type, signal_type) > Constants::max_neurons_in_source) {
            return CalculationType::Hermite;
        }

        return CalculationType::Taylor;
    }

    /**
     * @brief Calculates the force of attraction between all neurons from the subtrees
     * @param source The subtree that has all the sources in it
     * @param targets The subtree that has all the targets in it
     * @param element_type The element type that is looking for a connection
     * @param signal_type_needed Specifies for which type of neurons the calculation is to be executed (inhibitory or excitatory).
     * @exception Throws a RelearnException if source or target are nullptr
     * @return Returns the total attraction of the neurons.
     */
    static double calc_direct_gauss(OctreeNode<AdditionalCellAttributes>* source, OctreeNode<AdditionalCellAttributes>* target,
        const ElementType element_type, const SignalType signal_type_needed) {
        RelearnException::check(source != nullptr, "FastMultipoleMethodsBase::calc_direct_gauss: source is nullptr");
        RelearnException::check(target != nullptr, "FastMultipoleMethodsBase::calc_direct_gauss: target is nullptr");

        const auto sigma = GaussianDistributionKernel::get_sigma();
        const auto other_element_type = get_other_element_type(element_type);

        const auto& sources = get_all_positions_for(source, element_type, signal_type_needed);
        const auto& targets = get_all_positions_for(target, other_element_type, signal_type_needed);

        auto result = 0.0;

        for (const auto& [target_position, number_targets] : targets) {
            for (const auto& [source_position, number_sources] : sources) {
                const auto kernel_value = kernel(target_position, source_position, sigma);
                result += kernel_value * number_sources * number_targets;
            }
        }

        return result;
    }

    /**
     * @brief Calculates the hermite coefficients for a source node. The calculation of coefficients and series
     *      expansion is executed separately, because the coefficients can be reused.
     * @param source Node with vacant elements
     * @param element_type The type of synaptic elements that searches for partners
     * @param signal_type_needed Specifies for which type of neurons the calculation is to be executed (inhibitory or excitatory).
     * @exception Throws a RelearnException if source is nullptr, has no children, has no valid position for the specified combination of element type and signal type, or
     *      the children have no valid position for it
     * @returns Returns the hermite coefficients.
     */
    static std::vector<double> calc_hermite_coefficients(const OctreeNode<AdditionalCellAttributes>* source, const ElementType element_type, const SignalType signal_type_needed) {
        RelearnException::check(source != nullptr, "FastMultipoleMethodsBase::calc_hermite_coefficients: source is nullptr");
        RelearnException::check(source->is_parent(), "FastMultipoleMethodsBase::calc_hermite_coefficients: source node was a leaf node");

        Timers::start(TimerRegion::CALC_HERMITE_COEFFICIENTS);

        const auto sigma = GaussianDistributionKernel::get_sigma();
        const auto& indices = Multiindex::get_indices();

        std::vector<double> hermite_coefficients{};
        hermite_coefficients.reserve(Constants::p3);

        const auto& source_cell = source->get_cell();
        const auto& source_position_opt = source_cell.get_position_for(element_type, signal_type_needed);
        RelearnException::check(source_position_opt.has_value(), "FastMultipoleMethodsBase::calc_hermite_coefficients: source has no valid position.");

        const auto& source_position = source_position_opt.value();

        for (auto a = 0U; a < Constants::p3; a++) {
            auto child_attraction = 0.0;

            const auto& children = source->get_children();
            for (auto* child : children) {
                if (child == nullptr) {
                    continue;
                }

                const auto& cell = child->get_cell();
                const auto child_number_axons = cell.get_number_elements_for(element_type, signal_type_needed);
                if (child_number_axons == 0) {
                    continue;
                }

                const auto& child_pos = cell.get_position_for(element_type, signal_type_needed);
                RelearnException::check(child_pos.has_value(), "FastMultipoleMethodsBase::calc_hermite_coefficients: source child has no valid position.");

                const auto& temp_vec = (child_pos.value() - source_position) / sigma;
                child_attraction += child_number_axons * (temp_vec.get_componentwise_power(indices[a]));
            }

            const auto hermite_coefficient = child_attraction / indices[a].get_componentwise_factorial();
            hermite_coefficients[a] = hermite_coefficient;
        }

        Timers::stop_and_add(TimerRegion::CALC_HERMITE_COEFFICIENTS);

        return hermite_coefficients;
    }

    /**
     * @brief Calculates the taylor coefficients for a pair of nodes. The calculation of coefficients and series
     *      expansion is executed separately.
     * @param source Node with vacant elements
     * @param target_center Position of the target node.
     * @param element_type The type of synaptic elements that searches for partners
     * @param signal_type_needed Specifies for which type of neurons the calculation is to be executed (inhibitory or excitatory).
     * @return Returns the taylor coefficients.
     */
    static std::vector<double> calc_taylor_coefficients(const OctreeNode<AdditionalCellAttributes>* source, const position_type& target_center,
        const ElementType element_type, const SignalType signal_type_needed) {
        RelearnException::check(source != nullptr, "FastMultipoleMethodsBase::calc_taylor_coefficients: source is nullptr");
        RelearnException::check(source->is_parent(), "FastMultipoleMethodsBase::calc_taylor_coefficients: source node was a leaf node");

        Timers::start(TimerRegion::CALC_TAYLOR_COEFFICIENTS);

        const auto sigma = GaussianDistributionKernel::get_sigma();
        const auto& indices = Multiindex::get_indices();

        std::vector<double> taylor_coefficients{ 0.0 };
        taylor_coefficients.reserve(Constants::p3);

        const auto& children = source->get_children();

        for (auto index = 0; index < Constants::p3; index++) {
            // NOLINTNEXTLINE
            const auto& current_index = indices[index];

            auto child_attraction = 0.0;
            for (const auto* source_child : children) {
                if (source_child == nullptr) {
                    continue;
                }

                const auto& cell = source_child->get_cell();
                const auto number_elements = cell.get_number_elements_for(element_type, signal_type_needed);
                if (number_elements == 0) {
                    continue;
                }

                const auto& child_pos = cell.get_position_for(element_type, signal_type_needed);
                RelearnException::check(child_pos.has_value(), "FastMultipoleMethodsBase::calc_taylor_coefficients: source child has no position.");

                const auto& temp_vec = (child_pos.value() - target_center) / sigma;
                child_attraction += number_elements * h_multiindex(current_index, temp_vec);
            }

            const auto coefficient = child_attraction / current_index.get_componentwise_factorial();
            const auto absolute_multiindex = current_index.calculate_1_norm();

            if (absolute_multiindex % 2 == 0) {
                // NOLINTNEXTLINE
                taylor_coefficients[index] = coefficient;
            } else {
                // NOLINTNEXTLINE
                taylor_coefficients[index] = -coefficient;
            }
        }

        Timers::stop_and_add(TimerRegion::CALC_TAYLOR_COEFFICIENTS);

        return taylor_coefficients;
    }

    /**
     * @brief Calculates the force of attraction between two nodes of the octree using a Hermite series expansion.
     * @param source Node with vacant searching elements.
     * @param target Node with vacant searched elements.
     * @param coefficients_buffer Memory location where the coefficients are stored.
     * @param element_type The element type that searches
     * @param signal_type_needed Specifies for which type of neurons the calculation is to be executed (inhibitory or excitatory).
     * @exception Can throw a RelearnException.
     * @return Returns the attraction force.
     */
    static double calc_hermite(const OctreeNode<AdditionalCellAttributes>* source, OctreeNode<AdditionalCellAttributes>* target,
        const std::vector<double>& coefficients_buffer, const ElementType element_type, const SignalType signal_type_needed) {
        RelearnException::check(source != nullptr, "FastMultipoleMethodsBase::calc_hermite::calc_direct_gauss: source is nullptr");
        RelearnException::check(target != nullptr, "FastMultipoleMethodsBase::calc_hermite::calc_direct_gauss: target is nullptr");

        const auto sigma = GaussianDistributionKernel::get_sigma();
        const auto other_element_type = get_other_element_type(element_type);

        RelearnException::check(target->is_parent(), "FastMultipoleMethodsBase::calc_hermite: target node was a leaf node");

        const auto& opt_source_center = source->get_cell().get_position_for(element_type, signal_type_needed);
        RelearnException::check(opt_source_center.has_value(), "FastMultipoleMethodsBase::calc_hermite: source node has no axon position.");

        const auto& source_center = opt_source_center.value();

        constexpr const auto indices = Multiindex::get_indices();
        constexpr const auto number_coefficients = Multiindex::get_number_of_indices();

        double total_attraction = 0.0;

        const auto& interaction_list = get_children_to_array(target);
        for (const auto* child_target : interaction_list) {
            if (child_target == nullptr) {
                continue;
            }

            const auto& cell = child_target->get_cell();
            const auto number_searched_elements = cell.get_number_elements_for(other_element_type, signal_type_needed);
            if (number_searched_elements == 0) {
                continue;
            }

            const auto& child_pos = cell.get_position_for(other_element_type, signal_type_needed);
            RelearnException::check(child_pos.has_value(), "FastMultipoleMethodsBase::calc_hermite: target child node has no axon position.");

            const auto& temp_vec = (child_pos.value() - source_center) / sigma;

            double child_attraction = 0.0;
            for (auto a = 0; a < number_coefficients; a++) {
                // NOLINTNEXTLINE
                child_attraction += coefficients_buffer[a] * h_multiindex(indices[a], temp_vec);
            }

            total_attraction += number_searched_elements * child_attraction;
        }

        return total_attraction;
    }

    /**
     * @brief Calculates the force of attraction between two nodes of the octree using a Taylor series expansion.
     * @param source Node with vacant searching elements.
     * @param target Node with vacant searched elements.
     * @param element_type The element type that searches
     * @param signal_type_needed Specifies for which type of neurons the calculation is to be executed (inhibitory or excitatory).
     * @exception Can throw a RelearnException.
     * @return Returns the attraction force.
     */
    static double calc_taylor(const OctreeNode<AdditionalCellAttributes>* source, OctreeNode<AdditionalCellAttributes>* target,
        const ElementType element_type, const SignalType signal_type_needed) {
        RelearnException::check(source != nullptr, "FastMultipoleMethodsBase::calc_taylor: source is nullptr");
        RelearnException::check(target != nullptr, "FastMultipoleMethodsBase::calc_taylor: target is nullptr");

        const auto sigma = GaussianDistributionKernel::get_sigma();
        const auto other_element_type = get_other_element_type(element_type);

        const auto& opt_target_center = target->get_cell().get_position_for(other_element_type, signal_type_needed);
        RelearnException::check(opt_target_center.has_value(), "FastMultipoleMethodsBase::calc_taylor: target node has no position.");

        const auto& target_center = opt_target_center.value();
        const auto& taylor_coefficients = calc_taylor_coefficients(source, target_center, other_element_type, signal_type_needed);

        const auto& indices = Multiindex::get_indices();
        const auto& target_children = get_children_to_array(target);

        auto result = 0.0;
        for (const auto* target_child : target_children) {
            if (target_child == nullptr) {
                continue;
            }

            const auto& cell = target_child->get_cell();
            const auto number_searched_elements = cell.get_number_elements_for(other_element_type, signal_type_needed);
            if (number_searched_elements == 0) {
                continue;
            }

            const auto& child_pos = cell.get_position_for(other_element_type, signal_type_needed);
            RelearnException::check(child_pos.has_value(), "FastMultipoleMethodsBase::calc_taylor: target child has no position.");

            const auto& temp_vec = (child_pos.value() - target_center) / sigma;

            double child_attraction = 0.0;
            for (auto b = 0; b < Constants::p3; b++) {
                // NOLINTNEXTLINE
                child_attraction += taylor_coefficients[b] * (temp_vec.get_componentwise_power(indices[b]));
            }
            result += number_searched_elements * child_attraction;
        }

        return result;
    }

    /**
     * @brief Calculates the attraction between a single source neuron and all target neurons in the interaction list.
     * @param source Node with vacant searching elements
     * @param interaction_list List of Nodes with vacant searched elements
     * @param signal_type_needed Specifies for which type of neurons the calculation is to be executed (inhibitory or excitatory).
     * @param element_type The searching element type
     * @exception Can throw a RelearnException
     * @return Returns a vector with the calculated forces of attraction. This contains as many elements as the interaction list.
     */
    static std::vector<double> calc_attractiveness_to_connect(OctreeNode<AdditionalCellAttributes>* source, const interaction_list_type& interaction_list,
        const ElementType element_type, const SignalType signal_type_needed) {
        RelearnException::check(source != nullptr, "FastMultipoleMethodsBase::calc_attractiveness_to_connect: Source was a nullptr.");

        std::vector<double> result{};
        result.reserve(interaction_list.size());

        const auto other_element_type = get_other_element_type(element_type);

        std::vector<double> hermite_coefficients{ 0.0 };
        hermite_coefficients.reserve(Constants::p3);
        auto hermite_coefficients_init = false;

        // For every target calculate the attractiveness
        for (auto* current_target : interaction_list) {
            if (current_target == nullptr) {
                continue;
            }

            const auto calculation_type = check_calculation_requirements(source, current_target, other_element_type, signal_type_needed);

            if (calculation_type == CalculationType::Direct) {
                const auto direct_attraction = calc_direct_gauss(source, current_target, other_element_type, signal_type_needed);
                result.emplace_back(direct_attraction);
                continue;
            }

            if (calculation_type == CalculationType::Taylor) {
                const auto taylor_attraction = calc_taylor(source, current_target, element_type, signal_type_needed);
                result.emplace_back(taylor_attraction);
                continue;
            }

            if (!hermite_coefficients_init) {
                // When the Calculation Type is Hermite, initialize the coefficients once.
                hermite_coefficients = calc_hermite_coefficients(source, other_element_type, signal_type_needed);
                hermite_coefficients_init = true;
            }

            const auto hermite_attraction = calc_hermite(source, current_target, hermite_coefficients, element_type, signal_type_needed);
            result.emplace_back(hermite_attraction);
        }

        return result;
    }

    /**
     * @brief Aligns the level of source and target node and thereby creates the associated interaction list.
     *      This ist due to the reason that only the target parent is pushed to the stack to reduce the size.
     * @param source_node Node with vacant searching elements and the desired level.
     * @param target_parent Node with vacant searched elements and smaller level.
     * @param element_type Specified for which type of synaptic element is searching
     * @param signal_type Specifies for which type of neurons the calculation is to be executed (inhibitory or excitatory).
     * @return The corresponding interaction list suitable for one source node.
     */
    static interaction_list_type align_interaction_list(OctreeNode<AdditionalCellAttributes>* source_node, OctreeNode<AdditionalCellAttributes>* target_parent,
        const ElementType element_type, const SignalType signal_type) {
        RelearnException::check(source_node != nullptr, "FastMultipoleMethodsBase::align_interaction_list: source_node was null!");
        RelearnException::check(target_parent != nullptr, "FastMultipoleMethodsBase::align_interaction_list: target_parent was null!");

        interaction_list_type result{};
        const auto expected_number_nodes = static_cast<size_t>(std::pow(Constants::number_oct, Constants::unpacking + 1));
        result.reserve(expected_number_nodes);
        if (target_parent->is_leaf()) {
            result.emplace_back(target_parent);
            return result;
        }

        Stack<OctreeNode<AdditionalCellAttributes>*> stack{ 100 };
        stack.emplace_back(target_parent);

        const auto other_element_type = get_other_element_type(element_type);
        const auto desired_level = source_node->get_level();
        while (!stack.empty()) {
            auto* current_node = stack.pop_back();
            if (current_node->get_level() == desired_level || current_node->is_leaf()) {
                result.emplace_back(current_node);
                continue;
            }

            const auto& children = get_children_to_array(current_node);
            for (auto* child : children) {
                if (child == nullptr) {
                    continue;
                }

                if (child->get_cell().get_number_elements_for(other_element_type, signal_type) == 0) {
                    continue;
                }

                stack.emplace_back(child);
            }
        }

        return result;
    }

    /**
     * @brief Creates a list of possible targets for a source node, which is a leaf,
     *      such that the number of axons in source is at least as large as the number of all dendrites in the targets.
     * @param source Node with vacant axons. Must be a leaf node.
     * @param interaction_list List of all possible targets.
     * @param element_type Specified for which type of synaptic element is searching
     * @param signal_type_needed Specifies for which type of neurons the calculation is to be executed (inhibitory or excitatory).
     * @return Returns selected targets, which were chosen according to probability and together have more dendrites than there are axons.
     */
    static std::vector<OctreeNode<AdditionalCellAttributes>*> make_target_list(OctreeNode<AdditionalCellAttributes>* source_node, const interaction_list_type& interaction_list,
        const ElementType element_type, const SignalType signal_type_needed) {
        RelearnException::check(source_node != nullptr, "FastMultipoleMethodsBase::make_target_list: source_node is nullptr");
        RelearnException::check(source_node->is_leaf(), "FastMultipoleMethodsBase::make_target_list: source_node is not a leaf");

        const auto other_element_type = get_other_element_type(element_type);

        // How many searching elements we still have left
        auto source_number = source_node->get_cell().get_number_elements_for(element_type, signal_type_needed);

        // How many target children we still have left
        auto number_target_children = count_non_zero_elements(interaction_list);
        std::vector<OctreeNode<AdditionalCellAttributes>*> target_list{};
        target_list.reserve(number_target_children);

        auto connection_probabilities = calc_attractiveness_to_connect(source_node, interaction_list, element_type, signal_type_needed);

        while (source_number > 0 && number_target_children > 0) {
            const auto chosen_index = choose_interval(connection_probabilities);
            const auto target_node = extract_element(interaction_list, chosen_index);

            source_number -= target_node->get_cell().get_number_elements_for(other_element_type, signal_type_needed);
            number_target_children--;

            connection_probabilities[chosen_index] = 0;
            target_list.emplace_back(target_node);
        }

        return target_list;
    }

    /**
     * @brief If a target is a leaf node but the source is not, a pair of a selected source child and the target must be pushed back on the stack.
     * How many pairs are made depends on how many dendrites the target has and how many axons the individual sources children have.
     *
     * @param target_node Node with vacant dendrites. Must be a leaf node.
     * @param element_type Specified for which type of synaptic element is searching
     * @param signal_type_needed Specifies for which type of neurons the calculation is to be executed (inhibitory or excitatory).
     * @param stack Reference to the stack on which the pairs must be pushed back.
     * @param source_children Refernce on the children of the source node.
     */
    static void make_stack_entries_for_leaf(OctreeNode<AdditionalCellAttributes>* target_node, const SignalType signal_type_needed, const ElementType element_type,
        Stack<stack_entry>& stack, const std::array<OctreeNode<AdditionalCellAttributes>*, Constants::number_oct>& source_children) {
        RelearnException::check(target_node != nullptr, "FastMultipoleMethodsBase::make_stack_entries_for_leaf: target_node is nullptr");
        RelearnException::check(target_node->is_leaf(), "FastMultipoleMethodsBase::make_stack_entries_for_leaf: target_node wasn't a leaf");

        const auto other_element_type = get_other_element_type(element_type);
        const auto fixed_interaction_list = interaction_list_type{ target_node };

        auto number_searched_elements = target_node->get_cell().get_number_elements_for(other_element_type, signal_type_needed);

        std::vector<double> attractiveness(Constants::number_oct, 0.0);
        unsigned int number_source_children = 0;

        for (auto i = 0U; i < Constants::number_oct; i++) {
            auto* const source_child = source_children[i];
            if (source_child == nullptr) {
                continue;
            }

            if (source_child->get_cell().get_number_elements_for(element_type, signal_type_needed) == 0) {
                continue;
            }

            number_source_children++;
            const auto& connection_probabilities = calc_attractiveness_to_connect(source_child, fixed_interaction_list, element_type, signal_type_needed);
            RelearnException::check(connection_probabilities.size() == 1, "FastMultipoleMethodsBase::make_stack_entries_for_leaf: The number of connection probabilities wasn't 1");

            attractiveness[i] = connection_probabilities[0];
        }

        while (number_searched_elements > 0 && number_source_children > 0) {
            const auto chosen_index = choose_interval(attractiveness);

            number_searched_elements -= source_children[chosen_index]->get_cell().get_number_elements_for(element_type, signal_type_needed);
            attractiveness[chosen_index] = 0;

            stack.emplace_back(source_children[chosen_index], target_node, false);
        }
    }

    /**
     * @brief Creates an initialized stack for the make_creation_request_for method. Source nodes and target nodes are paired based on their level in the octree.
     *      It also depends on the level_offset specified in the config file.
     * @param root The root of the octree
     * @param local_roots The local roots of the octree (those of the branch nodes that are local)
     * @param branch_level The level in the octree on which the branch nodes are
     * @param element_type The searching element type
     * @param signal_type_needed Specifies for which type of neurons the calculation is to be executed (inhibitory or excitatory).
     * @return Returns the initalised stack.
     */
    static Stack<stack_entry> init_stack(OctreeNode<AdditionalCellAttributes>* root, const std::vector<OctreeNode<AdditionalCellAttributes>*>& local_roots,
        const std::uint16_t branch_level, const ElementType element_type, const SignalType signal_type_needed) {

        Stack<stack_entry> stack{ 200 };
        if (local_roots.empty()) {
            return stack;
        }

        if (branch_level == 0) {
            const auto& root_children = FastMultipoleMethodsBase::get_children_to_array(root);
            for (auto* child : root_children) {
                if (child == nullptr) {
                    continue;
                }

                if (child->get_cell().get_number_elements_for(element_type, signal_type_needed) == 0) {
                    continue;
                }

                stack.emplace_back(child, root, false);
            }

            return stack;
        }

        const auto other_element_type = get_other_element_type(element_type);

        for (auto* node : local_roots) {
            if (node == nullptr) {
                continue;
            }

            if (node->get_cell().get_number_elements_for(other_element_type, signal_type_needed) == 0) {
                continue;
            }

            const auto& children = node->get_children();
            for (auto* child : children) {
                if (child == nullptr) {
                    continue;
                }

                if (child->get_cell().get_number_elements_for(element_type, signal_type_needed) == 0) {
                    continue;
                }

                stack.emplace_back(child, node, false);
            }
        }

        return stack;
    }

    /**
     * @brief Takes the top node pair from the stack and unpacks them as many times as specified in the config file.
     *      This serves to give the neurons more freedom of choice. After that, the resulting pairs are put back on the stack.
     *      When Constants::unpacking == 0 the stack is not changed.
     * @param stack Stack on which node pairs are located and on which is worked on.
     */
    static void unpack_node_pair(Stack<stack_entry>& stack) {
        if (Constants::unpacking == 0) {
            return;
        }

        const auto& initial_pair = stack.pop_back();
        const auto& [source_node, target_node, unpacked] = initial_pair;

        RelearnException::check(source_node != nullptr, "FastMultipoleMethodsBase::unpack_node_pair: Source node was null!");
        if (source_node->is_leaf() || unpacked == true) {
            stack.emplace_back(source_node, target_node, true);
            return;
        }

        const auto target_level = source_node->get_level() + Constants::unpacking;
        Stack<stack_entry> unpacking_stack{ 30 };
        unpacking_stack.emplace_back(initial_pair);

        while (!unpacking_stack.empty()) {
            const auto& new_pair = unpacking_stack.pop_back();
            const auto& [new_source, new_target, _] = new_pair;

            if (new_source == nullptr) {
                continue;
            }

            if (new_source->get_level() == target_level || new_source->is_leaf()) {
                stack.emplace_back(new_pair);
                continue;
            }

            const auto& source_children = new_source->get_children();
            for (auto* child : source_children) {
                unpacking_stack.emplace_back(child, new_target, true);
            }
        }
    }

    /**
     * @brief Appends pairs of neurons to a SynapseCreationRequest which are suitable for a synapse formation.
     * @param root The root of the octree
     * @param local_roots The local roots of the octree (those of the branch nodes that are local)
     * @param branch_level The level in the octree on which the branch nodes are
     * @param element_type The searching element type
     * @param signal_type_needed Specifies for which type of neurons the calculation is to be executed (inhibitory or excitatory)
     * @param request SynapseCreationRequest which should be extended. This must be created before the method is called.
     * @exception Can throw a RelearnException.
     */
    static void make_creation_request_for(OctreeNode<AdditionalCellAttributes>* root, const std::vector<OctreeNode<AdditionalCellAttributes>*>& local_roots,
        const std::uint16_t branch_level, const ElementType element_type, const SignalType signal_type_needed, CommunicationMap<SynapseCreationRequest>& request) {
        RelearnException::check(root != nullptr, "FastMultipoleMethodsBase::make_creation_request_for: root is nullptr");

        const auto other_element_type = get_other_element_type(element_type);
        auto stack = init_stack(root, local_roots, branch_level, element_type, signal_type_needed);

        while (!stack.empty()) {
            unpack_node_pair(stack);
            // get node and interaction list from stack
            const auto& [source_node, target_parent, _] = stack.pop_back();

            if (source_node == nullptr) {
                continue;
            }

            if (source_node->get_cell().get_number_elements_for(element_type, signal_type_needed) == 0) {
                continue;
            }

            if (target_parent->get_cell().get_number_elements_for(other_element_type, signal_type_needed) == 0) {
                continue;
            }

            // extract target children to interaction_list if possible
            interaction_list_type interaction_list = align_interaction_list(source_node, target_parent, element_type, signal_type_needed);

            // current source node is a leaf node
            if (!source_node->is_parent()) {
                auto const target_list = make_target_list(source_node, interaction_list, element_type, signal_type_needed);

                for (auto* target : target_list) {
                    if (target->is_parent()) {
                        stack.emplace_back(source_node, target, false);
                        continue;
                    }

                    // current target is a leaf node
                    const auto target_id = target->get_cell().get_neuron_id();
                    const auto source_id = source_node->get_cell().get_neuron_id();

                    // No autapse
                    if (target_id != source_id) {
                        const auto target_rank = target->get_mpi_rank();
                        const SynapseCreationRequest creation_request(target_id, source_id, signal_type_needed);
                        request.append(target_rank, creation_request);
                    }
                }
                continue;
            }

            // source is an inner node
            const auto& connection_probabilities = calc_attractiveness_to_connect(source_node, interaction_list, element_type, signal_type_needed);
            const auto chosen_index = choose_interval(connection_probabilities);
            auto* target_node = extract_element(interaction_list, chosen_index);
            const auto& source_children = source_node->get_children();

            // target is leaf
            if (!target_node->is_parent()) {
                make_stack_entries_for_leaf(target_node, signal_type_needed, element_type, stack, source_children);
                continue;
            }

            // target is inner node
            for (const auto& source_child_node : source_children) {
                stack.emplace_back(source_child_node, target_node, false);
            }
        }
    }

    static void print_calculation(std::ostream& out_stream, OctreeNode<FastMultipoleMethodsCell>* source, OctreeNode<FastMultipoleMethodsCell>* target,
        const ElementType element_type, const SignalType needed) {

        const auto other_element_type = get_other_element_type(element_type);
        const auto calc_type = check_calculation_requirements(source, target, element_type, needed);

        const auto direct = calc_direct_gauss(source, target, element_type, needed);
        const auto taylor = calc_taylor(source, target, element_type, needed);
        const auto& coefficients = calc_hermite_coefficients(source, element_type, needed);
        const auto hermite = calc_hermite(source, target, coefficients, element_type, needed);

        out_stream << std::fixed;
        out_stream << direct << ",\t";
        out_stream << taylor << ",\t";
        out_stream << hermite << ",\t" << calc_type << '\n';
    }
};
