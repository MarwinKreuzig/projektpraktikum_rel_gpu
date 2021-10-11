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

#include "../structure/OctreeNode.h"
#include "../util/RelearnException.h"
#include "../util/Random.h"
#include "Algorithm.h"
#include "FastMultipoleMethodsCell.h"

#include <array>
#include <memory>
#include <vector>
#include <cmath>

template <typename T>
class OctreeImplementation;

/**
 * This class represents the implementation and adaptation of fast multipole methods. The parameters can be set on the fly.
 * It is strongly tied to Octree, which might perform MPI communication via NodeCache::download_children()
 */
class FastMultipoleMethods : public Algorithm {
public:
    using AdditionalCellAttributes = FastMultipoleMethodsCell;

    /**
     * @brief Constructs a new instance with the given octree
     * @param octree The octree on which the algorithm is to be performed, not null
     * @exception Throws a RelearnException if octree is nullptr
     */
    FastMultipoleMethods(const std::shared_ptr<OctreeImplementation<FastMultipoleMethods>>&octree)
    : global_tree(octree) {
        RelearnException::check(octree != nullptr, "FastMultipoleMethods::FastMultipoleMethods: octree was null");
    }

    /**
     * @brief Returns a collection of proposed synapse creations for each neuron with vacant axons. TODO: Does not work with MPI
     * @param num_neurons The number of local neurons
     * @param disable_flags Flags that indicate if a local neuron is disabled. If so (== 0), the neuron is ignored
     * @param extra_infos Used to access the positions of the local neurons
     * @param axons The axon model that is used
     * @exception Can throw a RelearnException
     * @return Returns a map, indicating for every MPI rank all requests that are made from this rank. 
     */
    MapSynapseCreationRequests find_target_neurons(const size_t num_neurons, const std::vector<char>& disable_flags,
            const std::unique_ptr<NeuronsExtraInfo>& extra_infos, const std::unique_ptr<SynapticElements>& axons) override;

    /**
     * @brief Updates all leaf nodes in the octree by the algorithm
     * @param disable_flags Flags that indicate if a neuron id disabled (0) or enabled (otherwise)
     * @param axons The model for the axons
     * @param excitatory_dendrites The model for the excitatory dendrites
     * @param inhibitory_dendrites The model for the inhibitory dendrites
     * @exception Throws a RelearnException if the vectors have different sizes or the leaf nodes are not in order of their neuron id
     */
    void update_leaf_nodes(const std::vector<char>& disable_flags, const std::unique_ptr<SynapticElements>& axons,
            const std::unique_ptr<SynapticElements>& excitatory_dendrites, const std::unique_ptr<SynapticElements>& inhibitory_dendrites) override;

    /**
     * @brief Updates the passed node with the values of its children according to the algorithm
     * @param node The node to update, must not be nullptr
     * @exception Throws a RelearnException if node is nullptr
     */
    static void update_functor(OctreeNode<FastMultipoleMethodsCell>* node) {
        RelearnException::check(node != nullptr, "FastMultipoleMethods::update_functor: node is nullptr");

        // NOLINTNEXTLINE
        if (!node->is_parent()) {
            return;
        }

        // I'm inner node, i.e., I have a super neuron
        Vec3d my_position_dendrites_excitatory = {0., 0., 0.};
        Vec3d my_position_dendrites_inhibitory = {0., 0., 0.};

        Vec3d my_position_axons_excitatory = {0., 0., 0.};
        Vec3d my_position_axons_inhibitory = {0., 0., 0.};

        // Sum of number of dendrites of all my children
        auto my_number_dendrites_excitatory = 0;
        auto my_number_dendrites_inhibitory = 0;

        auto my_number_axons_excitatory = 0;
        auto my_number_axons_inhibitory = 0;

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
            std::optional<Vec3d> child_position_dendrites_excitatory = child->get_cell().get_excitatory_dendrites_position();
            std::optional<Vec3d> child_position_dendrites_inhibitory = child->get_cell().get_inhibitory_dendrites_position();

            std::optional<Vec3d> child_position_axons_excitatory = child->get_cell().get_excitatory_axons_position();
            std::optional<Vec3d> child_position_axons_inhibitory = child->get_cell().get_inhibitory_axons_position();

            /**
             * We can use position if it's valid or if corresponding num of dendrites is 0 
             */
            RelearnException::check(child_position_dendrites_excitatory.has_value() || (0 == child_number_dendrites_excitatory), "FastMultipoleMethods::update_functor: The child had excitatory dendrites, but no position. ID: {}", child->get_cell_neuron_id());
            RelearnException::check(child_position_dendrites_inhibitory.has_value() || (0 == child_number_dendrites_inhibitory), "FastMultipoleMethods::update_functor: The child had inhibitory dendrites, but no position. ID: {}", child->get_cell_neuron_id());

            RelearnException::check(child_position_axons_excitatory.has_value() || (0 == child_number_axons_excitatory), "FastMultipoleMethods::update_functor: The child had excitatory axons, but no position. ID: {}", child->get_cell_neuron_id());
            RelearnException::check(child_position_axons_inhibitory.has_value() || (0 == child_number_axons_inhibitory), "FastMultipoleMethods::update_functor: The child had inhibitory axons, but no position. ID: {}", child->get_cell_neuron_id());

            if (child_position_dendrites_excitatory.has_value()) {
                const auto scaled_position = child_position_dendrites_excitatory.value() * static_cast<double> (child_number_dendrites_excitatory);
                my_position_dendrites_excitatory += scaled_position;
            }

            if (child_position_dendrites_inhibitory.has_value()) {
                const auto scaled_position = child_position_dendrites_inhibitory.value() * static_cast<double> (child_number_dendrites_inhibitory);
                my_position_dendrites_inhibitory += scaled_position;
            }

            if (child_position_axons_excitatory.has_value()) {
                const auto scaled_position = child_position_axons_excitatory.value() * static_cast<double> (child_number_axons_excitatory);
                my_position_axons_excitatory += scaled_position;
            }

            if (child_position_axons_inhibitory.has_value()) {
                const auto scaled_position = child_position_axons_inhibitory.value() * static_cast<double> (child_number_axons_inhibitory);
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
            node->set_cell_excitatory_dendrites_position(std::optional<Vec3d>{scaled_position});
        }

        if (0 == my_number_dendrites_inhibitory) {
            node->set_cell_inhibitory_dendrites_position({});
        } else {
            const auto scaled_position = my_position_dendrites_inhibitory / my_number_dendrites_inhibitory;
            node->set_cell_inhibitory_dendrites_position(std::optional<Vec3d>{scaled_position});
        }

        const auto& indices = Multiindex::get_indices();
        const auto num_coef = Multiindex::get_number_of_indices();

        if (0 == my_number_axons_excitatory) {
            node->set_cell_excitatory_axons_position({});
        } else {
            const auto scaled_position = my_position_axons_excitatory / my_number_axons_excitatory;
            node->set_cell_excitatory_axons_position(std::optional<Vec3d>{scaled_position});

            if (my_number_axons_excitatory > Constants::max_neurons_in_source) {
                for (auto a = 0; a < Constants::p3; a++) {
                    auto temp = 0.0;
                    for (auto i = 0; i < Constants::number_oct; i++) {
                        const auto* child = node->get_child(i);
                        if (child == nullptr) {
                            continue;
                        }

                        const auto child_number_excitatory_axons = child->get_cell().get_number_excitatory_axons();
                        if (child_number_excitatory_axons == 0) {
                            continue;
                        }

                        const auto& child_pos = child->get_cell().get_excitatory_axons_position();
                        const auto& temp_vec = (child_pos.value() - scaled_position) / default_sigma; // TODO: Change default_sigma to sigma
                        temp += child_number_excitatory_axons * pow_multiindex(temp_vec, indices[a]);
                    }

                    const auto hermite_coefficient = 1.0 * temp / fac_multiindex(indices[a]);
                    node->set_cell_excitatory_hermite_coefficient(a, hermite_coefficient);
                }
            }
        }

        if (0 == my_number_axons_inhibitory) {
            node->set_cell_inhibitory_axons_position({});
        } else {
            const auto scaled_position = my_position_axons_inhibitory / my_number_axons_inhibitory;
            node->set_cell_inhibitory_axons_position(std::optional<Vec3d>{scaled_position});

            if (my_number_axons_inhibitory > Constants::max_neurons_in_source) {
                for (auto a = 0; a < num_coef; a++) {
                    auto temp = 0.0;
                    // NOLINTNEXTLINE
                    const auto& current_index = indices[a];

                    for (auto i = 0; i < Constants::number_oct; i++) {
                        const auto* child = node->get_child(i);
                        if (child == nullptr) {
                            continue;
                        }

                        const auto child_number_inhibitory_axons = child->get_cell().get_number_inhibitory_axons();
                        if (child_number_inhibitory_axons == 0) {
                            continue;
                        }

                        const auto& child_pos = child->get_cell().get_inhibitory_axons_position();
                        const auto& temp_vec = (child_pos.value() - scaled_position) / default_sigma; // TODO: Change default_sigma to sigma
                        temp += child_number_inhibitory_axons * pow_multiindex(temp_vec, current_index);
                    }

                    const auto hermite_coefficient = 1.0 * temp / fac_multiindex(current_index);
                    node->set_cell_inhibitory_hermite_coefficient(a, hermite_coefficient);
                }
            }
        }
    }

private:
    std::vector<double> calc_attractiveness_to_connect_FMM(const OctreeNode<FastMultipoleMethodsCell>* source, const std::array<const OctreeNode<FastMultipoleMethodsCell>*, 8>& interaction_list,
            const SignalType dendrite_type_needed) const;

    unsigned int do_random_experiment(const OctreeNode<FastMultipoleMethodsCell>* source, const std::vector<double>& attractiveness) const;

    std::vector<double> calc_attractiveness_to_connect_FMM(const OctreeNode<FastMultipoleMethodsCell>* source,
            const std::array<const OctreeNode<FastMultipoleMethodsCell>*, 8>& interaction_list, const SignalType dendrite_type_needed);

    void make_creation_request_for(const SignalType needed, MapSynapseCreationRequests& request,
            std::stack<std::pair<OctreeNode<FastMultipoleMethodsCell>*, std::array<const OctreeNode<FastMultipoleMethodsCell>*, 8 >> >& nodes_with_axons);

    /**
     * @brief Calculates the coefficients which are needed for the derivatives of e^(-t^2).
     * @param derivative_order Order of the needed deriative (>0).
     * @return Retruns a vector with the coefficients.
     */
    static std::vector<int64_t> calculate_coefficients_for_deriative(unsigned int derivative_order) {
        static std::vector<std::vector < int64_t >> sequences{};

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
    static double function_derivative(double t, unsigned int derivative_order) noexcept {
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

    template <typename T>
    /**
     * @brief Calculates the faculty.
     * @param value 
     * @return Returns the faculty of the paramter value.
     */
    static T factorial(T value) noexcept {
        if (value < 2) {
            return 1;
        }

        T result = 1;
        while (value > 1) {
            result *= value;
            value--;
        }

        return result;
    }

    /**
     * @brief Calculates the n-th Hermite function at the point t, if t is one of the real numbers.
     * @param n Order of the Hermite function.
     * @param t Point of evaluation.
     * @return Value of the Hermite function of the n-th order at the point t.
     */
    static double h(unsigned int n, double t) {
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
   static double h_multiindex(const std::array<unsigned int, 3>& multi_index, const Vec3d& vector) {
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
    static size_t fac_multiindex(const std::array<unsigned int, 3>& x) {
        const auto fac_1 = factorial(x[0]);
        const auto fac_2 = factorial(x[1]);
        const auto fac_3 = factorial(x[2]);

        const auto product = fac_1 * fac_2 * fac_3;

        return product;
    }

    /**
     * @brief Calculates base_vector^exponent. 
     * @param base_vector A 3D vector.
     * @param exponent A 3D multi index.
     * @return The result of base_vector^exponent.
     */
    static double pow_multiindex(const Vec3d& base_vector, const std::array<unsigned int, 3>& exponent) {
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
    static size_t abs_multiindex(const std::array<unsigned int, 3>& x) {
        const auto sum = x[0] + x[1] + x[2];
        return sum;
    }

    /**
     * @brief The Kernel from Butz&Ooyen "A Simple Rule for Dendritic Spine and Axonal Bouton Formation Can Account for Cortical Reorganization afterFocal Retinal Lesions"
     * Calculates the attraction between two neurons, where a and b represent the position in three-dimensional space
     * @param a 3D position of the first neuron. 
     * @param b 3D position of the second neuron. 
     * @param sigma scaling parameter.
     * @return Returns the attraction between the two neurons.
     */
    static double kernel(const Vec3d& a, const Vec3d& b, const double sigma) {
        const auto diff = a - b;
        const auto squared_norm = diff.calculate_squared_2_norm();

        return exp(-squared_norm / (sigma * sigma));
    }

    /**
     * @brief Calculates the force of attraction between two nodes of the octree using a Taylor series expansion.
     * @param source Node with vacant axons.
     * @param target Node with vacant dendrites.
     * @param sigma Scaling constant.
     * @param needed Specifies for which type of neurons the calculation is to be executed (inhibitory or excitatory)
     * @return Retunrs the attraction force.
     */
    static double calc_taylor_expansion(const OctreeNode<FastMultipoleMethodsCell>* source, const OctreeNode<FastMultipoleMethodsCell>* target, const double sigma, const SignalType needed);

    /**
     * @brief Calculates the force of attraction between two sets of neurons by using the kernel 
     * presented by Butz and van Oooyen.
     * @param sources Vector of 3D positions of neurons with vacant axons.
     * @param targets Vector of 3D positions of neurons with vacant dendrites.
     * @param sigma Scaling constant.
     * @return Returns the total attraction of the neurons.
     */
    static double calc_direct_gauss(const std::vector<Vec3d>& sources, const std::vector<Vec3d>& targets, const double sigma) {
        auto result = 0.0;

        for (const auto& target : targets) {
            for (const auto& source : sources) {
                const auto kernel_value = kernel(target, source, sigma);
                result += kernel_value;
            }
        }

        return result;
    }

    /**
     * @brief Calculates the force of attraction between two nodes of the octree using a Hermite series expansion.
     * @param source Node with vacant axons.
     * @param target Node with vacant dendrites.
     * @param sigma Scaling constant.
     * @param needed Specifies for which type of neurons the calculation is to be executed (inhibitory or excitatory)
     * @return Retunrs the attraction force.
     */
    static double calc_hermite(const OctreeNode<FastMultipoleMethodsCell>* source, const OctreeNode<FastMultipoleMethodsCell>* target, const double sigma, const SignalType needed);

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

        // RelearnException::check(temp,"The sum of all attractions was 0.");
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

    std::shared_ptr<OctreeImplementation<FastMultipoleMethods>> global_tree
    {
    };

    /**
     * This class represents a mathematical three-dimensional multi-index, which is required for the 
     * series expansions and coefficient calculations. 
     */
    class Multiindex {
    public:

        /**
         * @brief Returns the number of all three-dimensional indices that the multi-index has. This depends on the selected p.
         * @return Returns the number of all indices.
         */
        static size_t get_number_of_indices() noexcept {
            return Constants::p3;
        }

        /**
         * @brief Returns the multi-index as a matrix with the dimensions 3 x p^3.
         * @return Returns a array of arrays wich represents the corresponding multi-index.
         */
        static std::array<std::array<unsigned int, 3>, Constants::p3> get_indices() {
            std::array<std::array<unsigned int, 3>, Constants::p3> result{};
            int index = 0;
            for (unsigned int i = 0; i < Constants::p; i++) {
                for (unsigned int j = 0; j < Constants::p; j++) {
                    for (unsigned int k = 0; k < Constants::p; k++) {
                        // NOLINTNEXTLINE
                        result[index] = {i, j, k};
                        index++;
                    }
                }
            }

            return result;
        }
    };
};