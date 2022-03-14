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

#include "RelearnException.h"

#include <utility>
#include <vector>

template <typename T>
class Stack {
    std::vector<T> container{};

public:
    using size_type = std::vector<T>::size_type;

    Stack(size_type reserved_size = 0) {
        container.reserve(reserved_size);
    }

    template <class... _Valty>
    constexpr decltype(auto) emplace_back(_Valty&&... _Val) {
        container.emplace_back(std::forward<_Valty>(_Val)...);
    }

    constexpr T pop_back() {
        T result = container[container.size() - 1];
        container.pop_back();
        return result;
    }

    constexpr void reserve(size_type new_capacity) {
        container.reserve(new_capacity);
    }

    constexpr size_type size() const noexcept {
        return container.size();
    }

    constexpr size_type capacity() const noexcept {
        return container.capacity();
    }

    constexpr bool empty() const noexcept {
        return container.empty();
    }
};