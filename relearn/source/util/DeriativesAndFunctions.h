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

#include "Multiindex.h"
#include "Vec3.h"
#include "../algorithm/FastMultipoleMethodsCell.h"
#include "../structure/OctreeNode.h"
#include "../util/Random.h"

#include <math.h>

// TODO: Move the functions into FastMultipoleMethods when applicable

template <typename T>
T factorial(T value) noexcept {
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

namespace Deriatives {
inline std::vector<int64_t> calculate_coefficients(unsigned int derivative_order) {
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

    sequences.resize(derivative_order + 1);

    for (auto i = old_size; i <= derivative_order; i++) {
        std::vector<int64_t> current_sequence(i + 2);
        std::fill(std::begin(current_sequence), std::end(current_sequence), 0);

        for (auto j = 0; j <= i; j++) {
            if (j != i) {
                current_sequence[j] = sequences[i - 1][j + 1] * (j + 1);
            }

            if (j > 0) {
                current_sequence[j] += sequences[i - 1][j - 1] * (-2);
            }
        }

        sequences[i] = std::move(current_sequence);
    }

    return sequences[derivative_order];
}

inline double function_derivative(double t, unsigned int derivative_order) noexcept {
    const auto& coefficients = calculate_coefficients(derivative_order);

    auto result = 0.0;
    for (auto monom_exponent = 0; monom_exponent <= derivative_order; monom_exponent++) {
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
}

namespace Functions {
// Hermite functions, returns -1 when n is smaller than 1
inline double h(unsigned int n, double t) {
    const auto t_squared = t * t;

    const auto fac_1 = exp(-t_squared);
    const auto fac_2 = exp(t_squared);
    const auto fac_3 = Deriatives::function_derivative(t, n);

    const auto product = fac_1 * fac_2 * fac_3;

    if (n % 2 == 0) {
        return product;
    }

    return -product;
}

inline double h_multiindex(const std::array<unsigned int, 3>& multi_index, const Vec3d& vector) {
    const auto h1 = h(multi_index[0], vector.get_x());
    const auto h2 = h(multi_index[1], vector.get_y());
    const auto h3 = h(multi_index[2], vector.get_z());

    const auto h_total = h1 * h2 * h3;

    return h_total;
}

inline int fac_multiindex(const std::array<unsigned int, 3>& x) {
    const auto fac_1 = factorial(x[0]);
    const auto fac_2 = factorial(x[1]);
    const auto fac_3 = factorial(x[2]);

    const auto product = fac_1 * fac_2 * fac_3;

    return product;
}

inline double pow_multiindex(const Vec3d& base_vector, const std::array<unsigned int, 3>& exponent) {
    const auto fac_1 = pow(base_vector.get_x(), exponent[0]);
    const auto fac_2 = pow(base_vector.get_y(), exponent[1]);
    const auto fac_3 = pow(base_vector.get_z(), exponent[2]);

    const auto product = fac_1 * fac_2 * fac_3;

    return product;
}

inline unsigned int abs_multiindex(const std::array<unsigned int, 3>& x) {
    const auto sum = x[0] + x[1] + x[2];
    return sum;
}

// Kernel from Butz&Ooyen "A Simple Rule for Dendritic Spine and Axonal Bouton Formation Can Account for Cortical Reorganization afterFocal Retinal Lesions"
// Calculates the attraction between two neurons, where a and b represent the position in three-dimensional space
inline double kernel(const Vec3d& a, const Vec3d& b, const double sigma) {
    const auto diff = a - b;
    const auto squared_norm = diff.calculate_squared_2_norm();

    return exp(-squared_norm / (sigma * sigma));
}

inline double calc_taylor_expansion(const OctreeNode<FastMultipoleMethodsCell>* source, const OctreeNode<FastMultipoleMethodsCell>* target, const double sigma, const SignalType needed) {
    const auto& opt_target_center = target->get_cell().get_dendrites_position_for(needed);
    RelearnException::check(opt_target_center.has_value(), "Target node has no position for Taylor calculation");

    const auto& target_center = opt_target_center.value();

    const auto& indices = Multiindex::get_indices();
    const auto number_coefficients = Multiindex::get_number_of_indices();

    std::array<double, Constants::p3> taylor_coefficients{};

    for (auto index = 0; index < number_coefficients; index++) {
        double temp = 0;
        const auto& current_index = indices[index];

        for (auto j = 0; j < Constants::number_oct; j++) {
            const auto* source_child = source->get_child(j);
            if (source_child == nullptr) {
                continue;
            }

            const auto number_axons = source_child->get_cell().get_number_axons_for(needed);
            if (number_axons == 0) {
                continue;
            }

            const auto& child_pos = source_child->get_cell().get_axons_position_for(needed);
            const auto& temp_vec = (child_pos.value() - target_center) / sigma;
            temp += number_axons * h_multiindex(current_index, temp_vec);
        }

        const auto factorial_multiindex = Functions::fac_multiindex(current_index);
        const auto coefficient = temp / factorial_multiindex;

        const auto absolute_multiindex = abs_multiindex(current_index);

        if (absolute_multiindex % 2 == 0) {
            taylor_coefficients[index] = coefficient;
        } else {
            taylor_coefficients[index] = -coefficient;
        }
    }

    double result = 0.0;

    for (auto j = 0; j < Constants::number_oct; j++) {
        const auto* target_child = target->get_child(j);
        if (target_child == nullptr) {
            continue;
        }

        const auto number_dendrites = target_child->get_cell().get_number_dendrites_for(needed);
        if (number_dendrites == 0) {
            continue;
        }

        const auto& child_pos = target_child->get_cell().get_dendrites_position_for(needed);
        const auto& temp_vec = (child_pos.value() - target_center) / sigma;

        double temp = 0.0;
        for (auto b = 0; b < number_coefficients; b++) {
            temp += taylor_coefficients[b] * pow_multiindex(temp_vec, indices[b]);
        }

        result += number_dendrites * temp;
    }

    return result;
}

inline double calc_direct_gauss(const std::vector<Vec3d>& sources, const std::vector<Vec3d>& targets, const double sigma) {
    auto result = 0.0;

    for (auto t = 0; t < targets.size(); t++) {
        for (auto s = 0; s < sources.size(); s++) {
            const auto kernel_value = Functions::kernel(targets[t], sources[s], sigma);
            result += kernel_value;
        }
    }

    return result;
}

inline double calc_hermite(const OctreeNode<FastMultipoleMethodsCell>* source, const OctreeNode<FastMultipoleMethodsCell>* target, const double sigma, const SignalType needed) {
    const auto& opt_source_center = source->get_cell().get_axons_position_for(needed);
    RelearnException::check(opt_source_center.has_value(), "Source node has no axon position for Hermite calculation \n");
    const auto& source_center = opt_source_center.value();

    const auto& indices = Multiindex::get_indices();
    const auto number_coefficients = Multiindex::get_number_of_indices();

    double result = 0.0;

    for (auto j = 0; j < Constants::number_oct; j++) {
        const auto* child_target = target->get_child(j);
        if (child_target == nullptr) {
            continue;
        }

        double temp = 0.0;

        const auto number_dendrites = child_target->get_cell().get_number_dendrites_for(needed);
        if (number_dendrites == 0) {
            continue;
        }

        const auto& child_pos = child_target->get_cell().get_dendrites_position_for(needed);
        const auto& temp_vec = (child_pos.value() - source_center) / sigma;

        for (auto a = 0; a < number_coefficients; a++) {
            temp += source->get_cell_hermite_coefficient_for(a, needed) * h_multiindex(indices[a], temp_vec);
        }

        result += number_dendrites * temp;
    }

    return result;
}

inline int choose_interval(const std::vector<double>& attractiveness) {
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
        intervals[i] = intervals[i - 1] + (attractiveness[i - 1] / sum);
    }

    int i = 0;
    while (random_number > intervals[i + 1] && i <= vec_len) {
        i++;
    }

    if (i >= vec_len + 1) {
        return 0;
    }

    return i;
}
}
