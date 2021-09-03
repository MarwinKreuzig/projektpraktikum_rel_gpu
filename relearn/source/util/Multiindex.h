#pragma once

#include "../Config.h"

#include <array>

class Multiindex {
public:
    static int get_number_of_indices() noexcept {
        return Constants::p3;
    }

    static std::array<std::array<unsigned int, 3>, Constants::p3> get_indices() {
        std::array<std::array<unsigned int, 3>, Constants::p3> result{};
        int x = 0;
        for (unsigned int i = 0; i < Constants::p; i++) {
            for (unsigned int j = 0; j < Constants::p; j++) {
                for (unsigned int k = 0; k < Constants::p; k++) {
                    result[x] = { i, j, k };
                    x++;
                }
            }
        }

        return result;
    }
};
