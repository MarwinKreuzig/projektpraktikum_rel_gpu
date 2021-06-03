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

#include <exception>
#include <string>
#include <vector>

class RelearnException : std::exception {
private:
    std::string message;

    template <typename... Args>
    static std::string string_format(const char* format, Args... args) {
        // NOLINTNEXTLINE
        int size = snprintf(nullptr, 0, format, args...) + 1; // Extra space for '\0'
        if (size <= 0) {
            return std::string("");
        }

        std::vector<char> vec(size);

        // NOLINTNEXTLINE
        snprintf(vec.data(), size, format, args...);
        return std::string(vec.data(), size - 1); // We don't want the '\0' inside
    }

public:
    static inline bool hide_messages{ false };

    RelearnException() = default;

    explicit RelearnException(std::string&& mes)
        : message(mes) {
    }

    [[nodiscard]] const char* what() const noexcept override;

    /**
    * If condition is true, nothing happens
    * If condition is false, format will serve as the error message, with placeholders replaced by args
    */
    template <typename... Args>
    static void check(bool condition, const char* format, Args... args) {
        if (condition) {
            return;
        }

        fail(std::move(string_format(format, args...)));
    }

    static void fail(std::string&& message);
};
