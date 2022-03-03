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

#include "FastMultipoleMethodsCell.h"
#include "algorithm/Algorithm.h"
#include "structure/OctreeNode.h"
#include "util/Random.h"
#include "util/RelearnException.h"
#include "util/Utility.h"

#include <array>
#include <memory>
#include <vector>
#include <cmath>

enum class CalculationType { DIRECT,
    HERMITE,
    TAYLOR,
};

template <typename T>
class OctreeImplementation;

/**
 * This class represents the implementation and adaptation of fast multipole methods. The parameters can be set on the fly.
 * It is strongly tied to Octree, which might perform MPI communication via NodeCache::download_children()
 */
class FastMultipoleMethods : public Algorithm {
    friend class FMMPrivateFunctionTest;

    std::shared_ptr<OctreeImplementation<FastMultipoleMethods>> global_tree{};

public:
    using AdditionalCellAttributes = FastMultipoleMethodsCell;
    using interaction_list_type = std::array<const OctreeNode<FastMultipoleMethodsCell>*, Constants::number_oct>;
    using position_type = typename Cell<AdditionalCellAttributes>::position_type;
    using counter_type = typename Cell<AdditionalCellAttributes>::counter_type;

    /**
     * @brief Constructs a new instance with the given octree
     * @param octree The octree on which the algorithm is to be performed, not null
     * @exception Throws a RelearnException if octree is nullptr
     */
    explicit FastMultipoleMethods(const std::shared_ptr<OctreeImplementation<FastMultipoleMethods>>& octree)
        : global_tree(octree) {
        RelearnException::check(octree != nullptr, "FastMultipoleMethods::FastMultipoleMethods: octree was null");
    }

    /**
     * @brief Updates all leaf nodes in the octree by the algorithm
     * @param disable_flags Flags that indicate if a neuron id disabled (0) or enabled (otherwise)
     * @param axons The model for the axons
     * @param excitatory_dendrites The model for the excitatory dendrites
     * @param inhibitory_dendrites The model for the inhibitory dendrites
     * @exception Throws a RelearnException if the vectors have different sizes or the leaf nodes are not in order of their neuron id
     */
    void update_leaf_nodes(const std::vector<UpdateStatus>& disable_flags) override;

    /**
     * @brief Updates the passed node with the values of its children according to the algorithm
     * @param node The node to update, must not be nullptr
     * @exception Throws a RelearnException if node is nullptr
     */
    static void
    update_functor(OctreeNode<FastMultipoleMethodsCell>* node) {
        RelearnException::check(node != nullptr, "FastMultipoleMethods::update_functor: node is nullptr");

        // NOLINTNEXTLINE
        if (!node->is_parent()) {
            return;
        }

        using position_type = FastMultipoleMethodsCell::position_type;
        using counter_type = FastMultipoleMethodsCell::counter_type;

        // I'm inner node, i.e., I have a super neuron
        position_type my_position_dendrites_excitatory = { 0., 0., 0. };
        position_type my_position_dendrites_inhibitory = { 0., 0., 0. };

        position_type my_position_axons_excitatory = { 0., 0., 0. };
        position_type my_position_axons_inhibitory = { 0., 0., 0. };

        // Sum of number of dendrites of all my children
        counter_type my_number_dendrites_excitatory = 0;
        counter_type my_number_dendrites_inhibitory = 0;

        counter_type my_number_axons_excitatory = 0;
        counter_type my_number_axons_inhibitory = 0;

        // For all my children
        for (const auto& child : node->get_children()) {
            if (child == nullptr) {
                continue;
            }

            // Sum up number of dendrites
            const auto child_number_dendrites_excitatory = child->get_cell().get_number_excitatory_dendrites();
            const auto child_number_dendrites_inhibitory = child->get_cell().get_number_inhibitory_dendrites();

            const auto child_number_axons_excitatory = child->get_cell().get_number_excitatory_axons();
            const auto child_number_axons_inhibitory = child->get_cell().get_number_inhibitory_axons();

            my_number_dendrites_excitatory += child_number_dendrites_excitatory;
            my_number_dendrites_inhibitory += child_number_dendrites_inhibitory;

            my_number_axons_excitatory += child_number_axons_excitatory;
            my_number_axons_inhibitory += child_number_axons_inhibitory;

            // Average the position by using the number of dendrites as weights
            std::optional<position_type> child_position_dendrites_excitatory = child->get_cell().get_excitatory_dendrites_position();
            std::optional<position_type> child_position_dendrites_inhibitory = child->get_cell().get_inhibitory_dendrites_position();

            std::optional<position_type> child_position_axons_excitatory = child->get_cell().get_excitatory_axons_position();
            std::optional<position_type> child_position_axons_inhibitory = child->get_cell().get_inhibitory_axons_position();

            /**
             * We can use position if it's valid or if corresponding num of dendrites is 0
             */
            RelearnException::check(child_position_dendrites_excitatory.has_value() || (0 == child_number_dendrites_excitatory), "FastMultipoleMethods::update_functor: The child had excitatory dendrites, but no position. ID: {}", child->get_cell_neuron_id());
            RelearnException::check(child_position_dendrites_inhibitory.has_value() || (0 == child_number_dendrites_inhibitory), "FastMultipoleMethods::update_functor: The child had inhibitory dendrites, but no position. ID: {}", child->get_cell_neuron_id());

            RelearnException::check(child_position_axons_excitatory.has_value() || (0 == child_number_axons_excitatory), "FastMultipoleMethods::update_functor: The child had excitatory axons, but no position. ID: {}", child->get_cell_neuron_id());
            RelearnException::check(child_position_axons_inhibitory.has_value() || (0 == child_number_axons_inhibitory), "FastMultipoleMethods::update_functor: The child had inhibitory axons, but no position. ID: {}", child->get_cell_neuron_id());

            if (child_position_dendrites_excitatory.has_value()) {
                const auto scaled_position = child_position_dendrites_excitatory.value() * static_cast<double>(child_number_dendrites_excitatory);
                my_position_dendrites_excitatory += scaled_position;
            }

            if (child_position_dendrites_inhibitory.has_value()) {
                const auto scaled_position = child_position_dendrites_inhibitory.value() * static_cast<double>(child_number_dendrites_inhibitory);
                my_position_dendrites_inhibitory += scaled_position;
            }

            if (child_position_axons_excitatory.has_value()) {
                const auto scaled_position = child_position_axons_excitatory.value() * static_cast<double>(child_number_axons_excitatory);
                my_position_axons_excitatory += scaled_position;
            }

            if (child_position_axons_inhibitory.has_value()) {
                const auto scaled_position = child_position_axons_inhibitory.value() * static_cast<double>(child_number_axons_inhibitory);
                my_position_axons_inhibitory += scaled_position;
            }
        }

        node->set_cell_number_dendrites(my_number_dendrites_excitatory, my_number_dendrites_inhibitory);
        node->set_cell_number_axons(my_number_axons_excitatory, my_number_axons_inhibitory);

        /**
         * For calculating the new weighted position, make sure that we don't
         * divide by 0. This happens if the my number of dendrites is 0.
         */
        if (0 == my_number_dendrites_excitatory) {
            node->set_cell_excitatory_dendrites_position({});
        } else {
            const auto scaled_position = my_position_dendrites_excitatory / my_number_dendrites_excitatory;
            node->set_cell_excitatory_dendrites_position(std::optional<position_type>{ scaled_position });
        }

        if (0 == my_number_dendrites_inhibitory) {
            node->set_cell_inhibitory_dendrites_position({});
        } else {
            const auto scaled_position = my_position_dendrites_inhibitory / my_number_dendrites_inhibitory;
            node->set_cell_inhibitory_dendrites_position(std::optional<position_type>{ scaled_position });
        }

        if (0 == my_number_axons_excitatory) {
            node->set_cell_excitatory_axons_position({});
        } else {
            const auto scaled_position = my_position_axons_excitatory / my_number_axons_excitatory;
            node->set_cell_excitatory_axons_position(std::optional<position_type>{ scaled_position });
        }

        if (0 == my_number_axons_inhibitory) {
            node->set_cell_inhibitory_axons_position({});
        } else {
            const auto scaled_position = my_position_axons_inhibitory / my_number_axons_inhibitory;
            node->set_cell_inhibitory_axons_position(std::optional<position_type>{ scaled_position });
        }
    }

    /**
     * @brief Returns a collection of proposed synapse creations for each neuron with vacant axons.
     * @param number_neurons The number of local neurons
     * @param disable_flags Flags that indicate if a local neuron is disabled. If so (== 0), the neuron is ignored
     * @param extra_infos Used to access the positions of the local neurons
     * @param axons The axon model that is used
     * @exception Can throw a RelearnException
     * @return Returns a map, indicating for every MPI rank all requests that are made from this rank. 
     */
    CommunicationMap<SynapseCreationRequest> find_target_neurons(size_t number_neurons, const std::vector<UpdateStatus>& disable_flags,
        const std::unique_ptr<NeuronsExtraInfo>& extra_infos) override;

private:
    /**
     * @brief Appends pairs of neurons to a SynapseCreationRequest which are suitable for a synapse formation.
     * @param signal_type_needed Specifies for which type of neurons the calculation is to be executed (inhibitory or excitatory)
     * @param request SynapseCreationRequest which should be extended. This must be created before the method is called.
     * @exception Can throw a RelearnException.
     */
    void make_creation_request_for(SignalType signal_type_needed, CommunicationMap<SynapseCreationRequest>& request);

    /**
     * @brief Calculates the attraction between a single source neuron and all target neurons in the interaction list.
     * @param source Node with vacant axons.
     * @param interaction_list List of Nodes with vacant dendrites.
     * @param signal_type_needed Specifies for which type of neurons the calculation is to be executed (inhibitory or excitatory).
     * @exception Can throw a RelearnException 
     * @return Returns a vector with the calculated forces of attraction. This contains as many elements as the interaction list.
     */
    std::vector<double> calc_attractiveness_to_connect(const OctreeNode<FastMultipoleMethodsCell>* source, const interaction_list_type& interaction_list, SignalType signal_type_needed);

    /**
     * @brief Checks which calculation type is suitable for a given source and target node.
     * @param source Node with vacant axons.
     * @param target Node with vacant dendrites.
     * @param signal_type_needed Specifies for which type of neurons the calculation is to be executed (inhibitory or excitatory)
     * @return CalculationType
     */
    static CalculationType check_calculation_requirements(const OctreeNode<FastMultipoleMethodsCell>* source, const OctreeNode<FastMultipoleMethodsCell>* target, SignalType signal_type_needed);

    /**
     * @brief Calculates the force of attraction between two nodes of the octree using a Taylor series expansion.
     * @param source Node with vacant axons.
     * @param target Node with vacant dendrites.
     * @param sigma Scaling constant.
     * @param signal_type_needed Specifies for which type of neurons the calculation is to be executed (inhibitory or excitatory).
     * @exception Can throw a RelearnException.
     * @return Returns the attraction force.
     */
    static double calc_taylor(const OctreeNode<FastMultipoleMethodsCell>* source, const OctreeNode<FastMultipoleMethodsCell>* target, double sigma, SignalType signal_type_needed);

    /**
     * @brief Calculates the force of attraction between two sets of neurons by using the kernel
     * presented by Butz and van Oooyen.
     * @param sources Vector of pairs with 3D position and number of vacant axons.
     * @param targets Vector of pairs with 3D position and number of vacant dendrites.
     * @param sigma Scaling constant.
     * @param signal_type_needed Specifies for which type of neurons the calculation is to be executed (inhibitory or excitatory).
     * @return Returns the total attraction of the neurons.
     */
    static double
    calc_direct_gauss(const OctreeNode<FastMultipoleMethodsCell>* source, const OctreeNode<FastMultipoleMethodsCell>* target, double sigma, SignalType signal_type_needed) {

        const std::vector<std::pair<Vec3d, unsigned int>>& sources = Utilities::get_all_positions_for(source, ElementType::AXON, signal_type_needed);
        const std::vector<std::pair<Vec3d, unsigned int>>& targets = Utilities::get_all_positions_for(target, ElementType::DENDRITE, signal_type_needed);

        auto result = 0.0;

        for (const auto& target : targets) {
            for (const auto& source : sources) {
                const auto kernel_value = Utilities::kernel(target.first, source.first, sigma);
                result += kernel_value * source.second * target.second;
            }
        }

        return result;
    }

    /**
     * @brief Calculates the hermite coefficients for a source node. The calculation of coefficients and series
     * expansion is executed separately, since the coefficients can be reused.
     * @param source Node with vacant axons.
     * @param sigma Scaling constant.
     * @param signal_type_needed Specifies for which type of neurons the calculation is to be executed (inhibitory or excitatory).
     * @exception Can throw a RelearnException.
     * @returns Returns an array of the hermite coefficients.
     */
    static std::array<double, Constants::p3> calc_hermite_coefficients(const OctreeNode<FastMultipoleMethodsCell>* source, double sigma, SignalType signal_type_needed);

    /**
     * @brief Calculates the force of attraction between two nodes of the octree using a Hermite series expansion.
     * @param source Node with vacant axons.
     * @param target Node with vacant dendrites.
     * @param coefficients_buffer Memory location where the coefficients are stored.
     * @param sigma Scaling constant.
     * @param signal_type_needed Specifies for which type of neurons the calculation is to be executed (inhibitory or excitatory).
     * @exception Can throw a RelearnException.
     * @return Retunrs the attraction force.
     */
    static double calc_hermite(const OctreeNode<FastMultipoleMethodsCell>* source, const OctreeNode<FastMultipoleMethodsCell>* target, const std::array<double, Constants::p3>& coefficients_buffer, double sigma, SignalType signal_type_needed);

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
     * This class contains service functions that are independent of class attributes, but are required for the FMM class.
     */
    class Utilities {
        friend class FMMPrivateFunctionTest;

    public:
        /**
         * @brief Counts the elements in an interaction list that are not nullptr.
         * @param arr Interaction list containing OctreeNodes.
         * @return Number of elements unequal to nullptr.
         */
        static unsigned int count_non_zero_elements(const interaction_list_type& arr);

        /**
         * @brief Returns the OctreeNode at the given index, nullptr elements are not counted.
         * @param arr Interaction list containing OctreeNodes.
         * @param index Index of the desired node.
         * @return const OctreeNode<AdditionalCellAttributes>* 
         */
        static const OctreeNode<AdditionalCellAttributes>* extract_element(const interaction_list_type& arr, unsigned int index);

        /**
         * @brief Checks whether a node is already in the cache and reloads the child nodes if necessary. Sets a children to nullptr, when it has no vacant dendrites.
         * @param node Node which is checked.
         * @return Interaction list with all children.
         */
        static interaction_list_type get_children_to_interaction_list(const OctreeNode<AdditionalCellAttributes>* node);

        /**
         * @brief Returns a vector of all positions of the selected type that have a free port of the requested SignalType. 
         * @param node OctreeNode from which the elements are to be counted.
         * @param type Type of synaptic elements (axon or dendrite).
         * @param needed The requested SignalType.
         * @return A vector of all actual positions.
         */
        static const std::vector<std::pair<position_type, counter_type>> get_all_positions_for(const OctreeNode<AdditionalCellAttributes>* node, const ElementType type, const SignalType signal_type_needed);

        /**
         * @brief Calculates the coefficients which are needed for the derivatives of e^(-t^2).
         * @param derivative_order Order of the needed deriative (>0).
         * @return Retruns a vector with the coefficients.
         */
        static std::vector<int64_t>
        calculate_coefficients_for_deriative(unsigned int derivative_order) {
            static std::vector<std::vector<int64_t>> sequences{};

            if (sequences.empty()) {
                std::vector<int64_t> initial_sequence(2);
                std::fill(std::begin(initial_sequence), std::end(initial_sequence), 0);
                initial_sequence[0] = 1;

                sequences.emplace_back(std::move(initial_sequence));
            }

            const auto old_size = sequences.size();

            if (old_size > derivative_order) {
                return sequences[derivative_order];
            }

            sequences.resize(derivative_order + 1ULL);

            for (auto i = old_size; i <= derivative_order; i++) {
                std::vector<int64_t> current_sequence(i + 2);
                std::fill(std::begin(current_sequence), std::end(current_sequence), 0);

            for (auto j = 0; j <= i; j++) {
                if (j != i) {
                    current_sequence[j] = sequences[i - 1][j + 1ULL] * (j + 1ULL);
                }

                    if (j > 0) {
                        current_sequence[j] += sequences[i - 1][j - 1ULL] * (-2);
                    }
                }

                sequences[i] = std::move(current_sequence);
            }

            return sequences[derivative_order];
        }

    /**
     * @brief Calculates the value of a certain derivative of e^(-t^2) at a desired point.
     * @param t Point for which the calculation is made.
     * @param derivative_order Order of the deriative.
     * @return Returns the value of the deriative.
     */
    static double
    function_derivative(double t, unsigned int derivative_order) noexcept {
        const auto& coefficients = calculate_coefficients_for_deriative(derivative_order);

            auto result = 0.0;
            for (unsigned int monom_exponent = 0; monom_exponent <= derivative_order; monom_exponent++) {
                const auto current_coefficient = coefficients[monom_exponent];

                if (current_coefficient == 0) {
                    continue;
                }

                const auto powered = pow(t, monom_exponent);
                const auto term = powered * current_coefficient;
                result += term;
            }

            const auto factor = exp(-(t * t));
            result *= factor;

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
        const auto t_squared = t * t;

            const auto fac_1 = exp(-t_squared);
            const auto fac_2 = exp(t_squared);
            const auto fac_3 = function_derivative(t, n);

            const auto product = fac_1 * fac_2 * fac_3;

            if (n % 2 == 0) {
                return product;
            }

            return -product;
        }

    /**
     * @brief Calculates the Hermite function for a multi index and a 3D vector.
     * @param multi_index A tuple of three natural numbers.
     * @param vector A 3D vector.
     * @return Value of the Hermite function.
     */
    static double
    h_multiindex(const std::array<unsigned int, 3>& multi_index, const Vec3d& vector) {
        const auto h1 = h(multi_index[0], vector.get_x());
        const auto h2 = h(multi_index[1], vector.get_y());
        const auto h3 = h(multi_index[2], vector.get_z());

            const auto h_total = h1 * h2 * h3;

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
        const auto fac_1 = pow(base_vector.get_x(), exponent[0]);
        const auto fac_2 = pow(base_vector.get_y(), exponent[1]);
        const auto fac_3 = pow(base_vector.get_z(), exponent[2]);

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
    };

    /**
     * This class represents a mathematical three-dimensional multi-index, which is required for the
     * series expansions and coefficient calculations.
     */
    class Multiindex {
        friend class FMMPrivateFunctionTest;

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
};
