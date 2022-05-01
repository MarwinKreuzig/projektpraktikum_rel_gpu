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
#include "structure/NodeCache.h"
#include "structure/OctreeNode.h"
#include "util/Random.h"
#include "util/RelearnException.h"
#include "util/Stack.h"
#include "FastMultipoleMethodsCell.h"
#include "util/Utility.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <vector>

enum class CalculationType { Direct,
    Hermite,
    Taylor,
};

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
    static constexpr unsigned int
    get_number_of_indices() noexcept {
        return Constants::p3;
    }

    /**
     * @brief Returns the multi-index as a matrix with the dimensions (p^3, 3).
     * @return Returns a array of arrays wich represents the corresponding multi-index.
     */
    static constexpr std::array<std::array<unsigned int, 3>, Constants::p3>
    get_indices() noexcept {
        std::array<std::array<unsigned int, 3>, Constants::p3> result{};
        int index = 0;
        for (unsigned int i = 0; i < Constants::p; i++) {
            for (unsigned int j = 0; j < Constants::p; j++) {
                for (unsigned int k = 0; k < Constants::p; k++) {
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
template <typename AdditionalCellAttributes>
class FastMultipoleMethodsBase {
public:
    using interaction_list_type = std::array<OctreeNode<AdditionalCellAttributes>*, Constants::number_oct>;
    using node_pair = std::array<OctreeNode<AdditionalCellAttributes>*, 2>;
    using position_type = typename Cell<AdditionalCellAttributes>::position_type;
    using counter_type = typename Cell<AdditionalCellAttributes>::counter_type;

    /**
     * @brief Counts the elements in an interaction list that are not nullptr.
     * @param arr Interaction list containing OctreeNodes.
     * @return Number of elements unequal to nullptr.
     */
    static unsigned int count_non_zero_elements(const interaction_list_type& arr) {
        auto non_zero_counter = 0;
        for (auto i = 0; i < Constants::number_oct; i++) {
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
     * @return const OctreeNode<AdditionalCellAttributes>*
     */
    static OctreeNode<AdditionalCellAttributes>* extract_element(const interaction_list_type& arr, unsigned int index) {
        auto non_zero_counter = 0;
        for (auto i = 0; i < Constants::number_oct; i++) {
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
     * @return Interaction list with all children.
     */
    static interaction_list_type get_children_to_interaction_list(OctreeNode<AdditionalCellAttributes>* node) {
        RelearnException::check(node != nullptr, "FastMultipoleMethods::Utilities::get_children_to_interaction_list: Node was a nullptr.");
        RelearnException::check(node->is_parent(), "FastMultipoleMethods::Utilities::get_children_to_interaction_list: Node has no children.");
        interaction_list_type result{ nullptr };

        const auto is_local = node->is_local();
        const auto& children = is_local ? node->get_children() : NodeCache<FastMultipoleMethodsCell>::download_children(node);

        unsigned int i = 0;

        for (auto it = children.crbegin(); it != children.crend(); ++it) {
            if (*it != nullptr) {
                result[i] = (*it);
            }
            i++;
        }
        return result;
    }

    /**
     * @brief Returns a vector of all positions of the selected type that have a free port of the requested SignalType.
     * @param node OctreeNode from which the elements are to be counted.
     * @param type Type of synaptic elements (axon or dendrite).
     * @param needed The requested SignalType.
     * @return A vector of all actual positions.
     */
    static const std::vector<std::pair<position_type, counter_type>> get_all_positions_for(OctreeNode<AdditionalCellAttributes>* node, const ElementType type, const SignalType signal_type_needed) {
        std::vector<std::pair<position_type, counter_type>> result{};

        //for (auto* current_node : node->get_children()) {
        //    const auto& cell = current_node->get_cell();
        //    unsigned int num_of_ports = 0;
        //    std::optional<VirtualPlasticityElementManual::position_type> opt_position;

        //    if (type == ElementType::Dendrite) {
        //        num_of_ports = cell.get_number_dendrites_for(signal_type_needed);
        //        opt_position = cell.get_dendrites_position_for(signal_type_needed);
        //    } else {
        //        num_of_ports = cell.get_number_axons_for(signal_type_needed);
        //        opt_position = cell.get_axons_position_for(signal_type_needed);
        //    }

        //    RelearnException::check(opt_position.has_value(), "FastMultipoleMethods::Utilities::get_all_positions_for: opt_position has no value.");
        //    // push number and position of dendritic elements to result
        //    result.emplace_back(std::pair<position_type, counter_type>(opt_position.value(), num_of_ports));
        //}

        //return result;

        Stack<OctreeNode<FastMultipoleMethodsCell>*> stack{ 30 };
        stack.emplace_back(node);

        while (!stack.empty()) {
            auto* current_node = stack.pop_back();

            // node is leaf
            if (!current_node->is_parent()) {
                // Get number and position, depending on which types were chosen.
                const auto& cell = current_node->get_cell();
                unsigned int num_of_ports = 0;
                std::optional<VirtualPlasticityElementManual::position_type> opt_position;

                if (type == ElementType::Dendrite) {
                    num_of_ports = cell.get_number_dendrites_for(signal_type_needed);
                    opt_position = cell.get_dendrites_position();
                } else {
                    num_of_ports = cell.get_number_axons_for(signal_type_needed);
                    opt_position = cell.get_axons_position();
                }

                RelearnException::check(opt_position.has_value(), "FastMultipoleMethods::Utilities::get_all_positions_for: opt_position has no value.");
                // push number and position of dendritic elements to result
                result.emplace_back(std::pair<position_type, counter_type>(opt_position.value(), num_of_ports));
                continue;
            }

            // node is inner node
            const auto is_local = current_node->is_local();
            const auto& children = is_local ? current_node->get_children() : NodeCache<AdditionalCellAttributes>::download_children(current_node);

            // push children to stack
            for (auto it = children.crbegin(); it != children.crend(); ++it) {
                if (*it == nullptr) {
                    continue;
                }
                auto number_syn_elemts = type == ElementType::Dendrite ? (*it)->get_cell().get_number_dendrites_for(signal_type_needed) : (*it)->get_cell().get_number_axons_for(signal_type_needed);
                if (number_syn_elemts == 0) {
                    continue;
                }
                stack.emplace_back(*it);
            }
        }
        return result;
    }

    /**
     * @brief Calculates the n-th Hermite function at the point t, if t is one of the real numbers.
     * @param n Order of the Hermite function.
     * @param t Point of evaluation.
     * @return Value of the Hermite function of the n-th order at the point t.
     */
    static double
    h(unsigned int n, double t) {
        double t_squared = t * t;

        double fac_1 = exp(-t_squared);
        double fac_2 = std::hermite(n, t);

        double product = fac_1 * fac_2;

        return product;
    }

    /**
     * @brief Calculates the Hermite function for a multi index and a 3D vector.
     * @param multi_index A tuple of three natural numbers.
     * @param vector A 3D vector.
     * @return Value of the Hermite function.
     */
    static double
    h_multiindex(const std::array<unsigned int, 3>& multi_index, const Vec3d& vector) {
        double h1 = h(multi_index[0], vector.get_x());
        double h2 = h(multi_index[1], vector.get_y());
        double h3 = h(multi_index[2], vector.get_z());

        double h_total = h1 * h2 * h3;

        return h_total;
    }

    /**
     * @brief Calculates the factorial of a multi index.
     * @param x a tuple of three natural numbers
     * @return Returns the factorial of x.
     */
    static constexpr size_t
    fac_multiindex(const std::array<unsigned int, 3>& x) {
        const auto fac_1 = Util::factorial(x[0]);
        const auto fac_2 = Util::factorial(x[1]);
        const auto fac_3 = Util::factorial(x[2]);

        const auto product = fac_1 * fac_2 * fac_3;

        return product;
    }

    /**
     * @brief Calculates base_vector^exponent.
     * @param base_vector A 3D vector.
     * @param exponent A 3D multi index.
     * @return The result of base_vector^exponent.
     */
    static double
    pow_multiindex(const Vec3d& base_vector, const std::array<unsigned int, 3>& exponent) {
        const auto fac_1 = pow(std::abs(base_vector.get_x()), exponent[0]);
        const auto fac_2 = pow(std::abs(base_vector.get_y()), exponent[1]);
        const auto fac_3 = pow(std::abs(base_vector.get_z()), exponent[2]);

        const auto product = fac_1 * fac_2 * fac_3;

        return product;
    }

    /**
     * @brief Calculates the absolute value of a 3D index.
     * @param x tuple of three natural numbers.
     * @return Returns the absolute value of x.
     */
    static constexpr size_t
    abs_multiindex(const std::array<unsigned int, 3>& x) {
        const auto sum = x[0] + x[1] + x[2];
        return sum;
    }

    /**
     * @brief The Kernel from Butz&Ooyen "A Simple Rule for Dendritic Spine and Axonal Bouton Formation Can Account for Cortical Reorganization afterFocal Retinal Lesions"
     *       Calculates the attraction between two neurons, where a and b represent the position in three-dimensional space
     * @param a 3D position of the first neuron.
     * @param b 3D position of the second neuron.
     * @param sigma scaling parameter.
     * @return Returns the attraction between the two neurons.
     */
    static double
    kernel(const Vec3d& a, const Vec3d& b, const double sigma) {
        const auto diff = a - b;
        const auto squared_norm = diff.calculate_squared_2_norm();

        return exp(-squared_norm / (sigma * sigma));
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
};