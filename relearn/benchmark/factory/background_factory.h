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

#include "neurons/models/BackgroundActivityCalculator.h"
#include "neurons/models/BackgroundActivityCalculators.h"

#include <memory>

class BackgroundFactory {
public:
    static std::unique_ptr<BackgroundActivityCalculator> construct_null_background() {
        return std::make_unique<NullBackgroundActivityCalculator>();
    }

    static std::unique_ptr<ConstantBackgroundActivityCalculator> construct_constant_background(const double background = 1.0) {
        return std::make_unique<ConstantBackgroundActivityCalculator>(background);
    }

    static std::unique_ptr<NormalBackgroundActivityCalculator> construct_normal_background(const double mean = 1.0, const double stddev = 1.0) {
        return std::make_unique<NormalBackgroundActivityCalculator>(mean, stddev);
    }
};
