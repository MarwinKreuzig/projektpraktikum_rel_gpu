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

#include "util/RelearnException.h"

#include <utility>
#include <vector>

#include "util/RelearnException.h"

/**
 * This class provides a stack-like interface, uses an std::vector as container, and allows to reserve space before.
 * @tparam T The type of elements on the stack
 */
template <typename T>
class Stack {
public:
    using size_type = typename std::vector<T>::size_type;

    /**
     * @brief Constructs a new Stack with the specified reserved capacity
     * @param reserved_size The reserved capacity
     * @exception Throws an exception if the memory allocation fails
     */
    explicit Stack(size_type reserved_size = 0) {
        container.reserve(reserved_size);
    }

    /**
     * @brief Emplaces a newly created element on the stack
     * @tparam ...ValueType The type for the constructor of the element
     * @param ...Val The values for the constructor of the element
     * @exception Throws an exception if the memory allocation fails or the constructor of the element throws
     * @return A reference to the newly created element
     */
    template <class... ValueType>
    constexpr decltype(auto) emplace_back(ValueType&&... Val) {
        return container.emplace_back(std::forward<ValueType>(Val)...);
    }

    /**
     * @brief Returns a mutable reference to the last element
     * @exception Throws a RelearnException if the stack was empty
     * @return A mutable reference to the last element
     */
    [[nodiscard]] constexpr T& top() {
        RelearnException::check(!empty(), "Stack::top(): The stack was empty!");
        return container.back();
    }

    /**
     * @brief Removes the last element
     * @exception Throws a RelearnException if the stack was empty
     */
    constexpr void pop() {
        RelearnException::check(!empty(), "Stack::pop(): The stack was empty!");
        container.pop_back();
    }

    /**
     * @brief Returns the latest element stored in the stack and pops it as well.
     * @exception Throws a RelearnException if the stack was empty
     * @return The latest element
     */
    [[nodiscard]] constexpr T pop_back() {
        RelearnException::check(!empty(), "Stack::pop_back(): The stack was empty!");

        auto result = std::move(container.back());
        container.pop_back();
        return result;
    }

    /**
     * @brief Reserves the specified capacity. Does nothing if the current capacity is larger than the specified one
     * @param new_capacity The to-be-reserved capacity
     * @exception Throws an exception if allocating the memory fails
     */
    constexpr void reserve(size_type new_capacity) {
        container.reserve(new_capacity);
    }

    /**
     * @brief Returns the number of currently stored elements
     * @return The number of currently stored elements
     */
    [[nodiscard]] constexpr size_type size() const noexcept {
        return container.size();
    }

    /**
     * @brief Returns the current capacity
     * @return The current capacity
     */
    [[nodiscard]] constexpr size_type capacity() const noexcept {
        return container.capacity();
    }

    /**
     * @brief Returns whether the stack is empty
     * @return True iff the stack is empty
     */
    [[nodiscard]] constexpr bool empty() const noexcept {
        return container.empty();
    }

private:
    std::vector<T> container{};
};
