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

#include "Config.h"
#include <cuda.h>
#include <iostream>

#include <exception>
#include <string>
#include <utility>
#include <memory>
#include <stdexcept>

/**
 * This class serves as a collective exception class that can check for conditions,
 * and in case of the condition evaluating to false, it logs the message and then fails.
 * Log messages can be disabled via RelearnException::hide_messages.
 * In case a condition evaluated to false and it logs the message, it calls MPIWrapper::get_num_ranks and MPIWrapper::get_my_rank.
 */
class RelearnGPUException : public std::exception {
public:
    /**
     * @brief Allows to hide the messages, i.e., not print the messages to std::
     */
    static inline bool hide_messages{ false };

    /**
     * @brief Returns the cause of the exception, i.e., the stored message
     * @return A constant char pointer to the content of the message
     */
    [[nodiscard]] const char* what() const noexcept override;

    /**
     * @brief Checks the condition and in case of false, logs the message and throws an RelearnException
     * @tparam FormatString A string-like type
     * @tparam ...Args Different types that can be substituted into the placeholders
     * @param condition The condition to evaluate
     * @param format The format string. Placeholders can used: "{}"
     * @param ...args The values that shall be substituted for the placeholders
     * @exception Throws an exception if the number of args does not match the number of placeholders in format
     *      Throws a RelearnException if the condition evaluates to false
     */
    template <typename... Args>
    static void check(bool condition, std::string&& format, Args&&... args) {
        if (condition) {
            return;
        }

        fail(std::move(format), std::forward<Args>(args)...);
    }

    template <typename... Args>
    static __device__ void device_check(bool condition, const char* format, Args&&... args) {
        if (condition) {
            return;
        }

        fail_device(std::move(format), std::forward<Args>(args)...);
    }

    /**
     * @brief Prints the log message and throws a RelearnException afterwards
     * @tparam FormatString A string-like type
     * @tparam ...Args Different types that can be substituted into the placeholders
     * @param format The format string. Placeholders can used: "{}"
     * @param ...args The values that shall be substituted for the placeholders
     * @exception Throws an exception if the number of args does not match the number of placeholders in format
     *      Throws a RelearnException
     */
    template <typename... Args>
    [[noreturn]] static void fail(const std::string&& format, Args&&... args) {
        if (hide_messages) {
            throw RelearnGPUException{};
        }

        const auto message = string_format(format, std::forward<Args>(args)...);

        log_message(message);
        throw RelearnGPUException{ message };
    }

    template <typename... Args>
    [[noreturn]] static __device__ void fail_device(const char* format, Args&&... args) {

        printf(format, args...);

        __trap();
    }

private:
    std::string message{};

    /**
     * @brief Default constructs an instance with empty message
     */
    RelearnGPUException() = default;

    /**
     * @brief Constructs an instance with the associated message
     * @param mes The message of the exception
     */
    explicit RelearnGPUException(std::string mes);

    static void log_message(const std::string& message);

    template <typename... Args>
    static std::string string_format(const std::string& format, Args... args) {
        int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) + 1; // Extra space for '\0'
        if (size_s <= 0) {
            throw std::runtime_error("Error during formatting.");
        }
        auto size = static_cast<size_t>(size_s);
        std::unique_ptr<char[]> buf(new char[size]);
        std::snprintf(buf.get(), size, format.c_str(), args...);
        return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
    }
};
