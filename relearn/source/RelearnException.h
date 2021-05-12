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
#include <utility>

#include "spdlog/fmt/bundled/core.h"

class RelearnException : std::exception {
private:
    std::string message;

    static void log_message(const std::string& message);

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
    template <typename FormatString, typename... Args>
    static void check(bool condition, FormatString&& format, Args&&... args) {
        if (condition) {
            return;
        }

        fail(std::forward<FormatString>(format), std::forward<Args>(args)...);
    }

    template <typename FormatString, typename... Args>
    static void fail(FormatString&& format, Args&&... args) {
        if (hide_messages) {
            throw RelearnException{};
        }

        auto message = fmt::format(std::forward<FormatString>(format), std::forward<Args>(args)...);
        log_message(message);
        throw RelearnException{ std::move(message) };
    }
};
