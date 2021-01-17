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

#include <string>
#include <utility>
#include <variant>

/**
 * Parameter of a model of type T
 */
template <typename T>
class Parameter {
public:
    using value_type = T;

    Parameter(std::string name, T& value, const T& min, const T& max)
        : name_ { std::move(name) }
        , value_ { value }
        , min_ { min }
        , max_ { max } { }

    [[nodiscard]] const std::string& name() const noexcept {
        return name_;
    }

    [[nodiscard]] value_type& value() noexcept {
        return value_;
    }

    [[nodiscard]] const value_type& value() const noexcept {
        return value_;
    }

    [[nodiscard]] const value_type& min() const noexcept {
        return min_;
    }

    [[nodiscard]] const value_type& max() const noexcept {
        return max_;
    }

private:
    const std::string name_ {}; // name of the parameter
    T& value_ {}; // value of the parameter
    const T min_ {}; // minimum value of the parameter
    const T max_ {}; // maximum value of the parameter
};

/**
 * Variant of every Parameter of type T
 */
using ModelParameter = std::variant<Parameter<unsigned int>, Parameter<double>, Parameter<size_t>>;
