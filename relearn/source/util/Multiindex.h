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

#include "../Config.h"

#include <array>

// TODO(hannah): Move into FastMultipoleMethods if applicable

class Multiindex {
public:
    static int get_number_of_indices() noexcept {
        return Constants::p3;
    }

    static std::array<std::array<unsigned int, 3>, Constants::p3> get_indices() {
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
