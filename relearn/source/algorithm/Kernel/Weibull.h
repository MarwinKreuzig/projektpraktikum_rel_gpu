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
#include "util/RelearnException.h"
#include "util/Vec3.h"
#include "gpu/algorithm/kernel/KernelGPUInterface.h"
#include "gpu/utils/CudaHelper.h"

#include <cmath>
#include <numeric>

/**
 * Offers a static interface to calculate the attraction based on a weibull distribution, i.e.,
 * if x is the distance, the attraction is proportional to
 * b * k * x^(k-1) * exp(-b * x^k)
 */
class WeibullDistributionKernel {
public:
    using counter_type = RelearnTypes::counter_type;
    using position_type = RelearnTypes::position_type;

    static constexpr double default_k = 1.0;
    static constexpr double default_b = 1.0;

    /**
     * @brief Sets the shape parameter k, must be greater than 0.0
     * @param k The shape parameter, > 0.0
     * @exception Throws a RelearnException if k <= 0.0
     */
    static void set_k(const double k) {
        RelearnException::check(k > 0.0, "In WeibullDistributionKernel::set_k, k was not greater than 0.0");
        WeibullDistributionKernel::k = k;

        if (get_gpu_handle()) {
            get_gpu_handle()->set_k(k);
        }
    }

    /**
     * @brief Returns the currently used shape parameter
     * @return The currently used shape parameter
     */
    [[nodiscard]] static double get_k() noexcept {
        return k;
    }

    /**
     * @brief Sets the scale parameter b, must be greater than 0.0
     * @param b The scaling parameter, > 0.0
     * @exception Throws a RelearnException if b <= 0.0
     */
    static void set_b(const double b) {
        RelearnException::check(b > 0.0, "In WeibullDistributionKernel::set_b, b was not greater than 0.0");
        WeibullDistributionKernel::b = b;

        if (get_gpu_handle()) {
            get_gpu_handle()->set_b(b);
        }
    }

    /**
     * @brief Returns the currently used scale parameter
     * @return The currently used scale parameter
     */
    [[nodiscard]] static double get_b() noexcept {
        return b;
    }

    /**
     * @brief Get the handle to the GPU version of this class
     * @return The GPU Handle
     */
    [[nodiscard]] static const std::shared_ptr<gpu::kernel::WeibullDistributionKernelHandle>& get_gpu_handle() {
        RelearnException::check(CudaHelper::is_cuda_available(), "WeibullDistributionKernel::get_gpu_handle: GPU not supported");

        static std::shared_ptr<gpu::kernel::WeibullDistributionKernelHandle> gpu_handle{ gpu::kernel::create_weibull(default_k, default_b) };

        return gpu_handle;
    }

    /**
     * @brief Calculates the attractiveness to connect on the basis of the weibull distribution
     * @param source_position The source position s
     * @param target_position The target position t
     * @param number_free_elements The linear scaling factor l
     * @return The calculated attractiveness
     */
    [[nodiscard]] static double calculate_attractiveness_to_connect(const position_type& source_position, const position_type& target_position,
        const counter_type& number_free_elements) noexcept {
        if (number_free_elements == 0) {
            return 0.0;
        }

        const auto factor_1 = number_free_elements * b * k;

        const auto x = (source_position - target_position).calculate_2_norm();

        const auto factor_2 = std::pow(x, k - 1);
        const auto exponent = -b * factor_2 * x;
        const auto factor_3 = std::exp(exponent);

        const auto result = factor_1 * factor_2 * factor_3;

        return result;
    }

private:
    static inline double k{ default_k };
    static inline double b{ default_b };
};
