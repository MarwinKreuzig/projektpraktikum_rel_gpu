
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

#include <compare>
#include <concepts>
#include <cstdint>
#include <ostream>
#include <type_traits>

#include <spdlog/fmt/bundled/core.h>
#include <spdlog/fmt/bundled/ostream.h>

#include "RelearnException.h"

namespace detail {
template <std::integral T>
[[nodiscard]] inline constexpr T get_max_size(const std::size_t& bit_count) {
    std::size_t res{ 1 };

    for (std::size_t i = 0; i < bit_count - std::size_t{ 1 }; ++i) {
        res <<= 1U;
        ++res;
    }

    return static_cast<T>(res);
}

template <std::integral T, std::size_t num_bits>
struct TaggedIDNumericalLimitsSigned {
    using value_type = T;
    static constexpr value_type min = -get_max_size<value_type>(num_bits - 1) - 1;
    static constexpr value_type max = get_max_size<value_type>(num_bits - 1);
};

template <std::integral T, std::size_t num_bits>
struct TaggedIDNumericalLimitsUnsigned {
    using value_type = T;
    static constexpr value_type min = 0;
    static constexpr value_type max = get_max_size<value_type>(num_bits);
};

template <std::integral T, std::size_t num_bits>
struct TaggedIDNumericalLimits : public std::conditional_t<
                                     std::is_signed_v<T>,
                                     TaggedIDNumericalLimitsSigned<T, num_bits>,
                                     TaggedIDNumericalLimitsUnsigned<T, num_bits>> {
};
} // namespace detail

/**
 * @brief ID class to represent a tagged id value.
 *
 * Flag members include is_global, is_virtual and is_initialized.
 * The limits type can be used to query the range of id values the tagged id can represent.
 *
 * The flags is_virtual and is_global are false by default and can only be specified in the constructor.
 * The is_initialized flag is true when the id was explicitly initialized with an id value,
 * or the id object gets an id assigned.
 *
 * @tparam T value type of the underlying id
 */
template <std::integral T = std::uint64_t>
class TaggedID {
public:
    using value_type = T;
    static constexpr auto num_flags = 3;
    static constexpr auto id_bit_count = sizeof(value_type) * 8 - num_flags;
    using limits = detail::TaggedIDNumericalLimits<value_type, id_bit_count>;

    /**
     * @brief Get an uninitialized id
     *
     * @return constexpr TaggedID uninitialized id
     */
    static constexpr TaggedID uninitialized_id() noexcept { return TaggedID{}; }

    /**
     * @brief Get a virtual id (is initialized, but virtual)
     *
     * @return constexpr TaggedID
     */
    static constexpr TaggedID virtual_id() noexcept { return TaggedID{ false, true, 0 }; }

    /**
     * @brief Construct a new TaggedID object where the flag is_initialized is false
     *
     */
    TaggedID() = default;

    /**
     * @brief Construct a new initialized TaggedID object with the given id
     *
     * @param id the id value
     */
    constexpr explicit TaggedID(std::integral auto id) noexcept
        : is_initialized_{ true }
        , id_{ static_cast<value_type>(id) } { }

    /**
     * @brief Construct a new initialized TaggedID object with the given flags and id
     *
     * @param is_global flag if the id should be marked global
     * @param is_virtual flag if the id should be marked virtual
     * @param id the id value
     */
    constexpr explicit TaggedID(bool is_global, bool is_virtual, std::integral auto id) noexcept
        : is_initialized_{ true }
        , is_global_{ is_global }
        , is_virtual_{ is_virtual }
        , id_{ static_cast<value_type>(id) } { }

    TaggedID(const TaggedID&) noexcept = default;
    TaggedID& operator=(const TaggedID&) noexcept = default;

    TaggedID(TaggedID&&) noexcept = default;
    TaggedID& operator=(TaggedID&&) noexcept = default;

    ~TaggedID() = default;

    /**
     * @brief Assign the new id to this tagged id and set is_initialized to true
     *
     * @param id the new id
     * @return TaggedID& *this
     */
    constexpr TaggedID& operator=(const std::integral auto& id) noexcept {
        is_initialized_ = true;
        this->id_ = static_cast<value_type>(id);
        return *this;
    }

    [[nodiscard]] constexpr explicit operator value_type() const noexcept {
        return id();
    }

    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return is_initialized();
    }

    [[nodiscard]] constexpr value_type id() const { return id_; }

    [[nodiscard]] constexpr value_type get_global_id() const {
        RelearnException::check(is_global(), "TaggedID::get_global_id is not global {:s}", *this);
        return id();
    }

    [[nodiscard]] constexpr value_type get_local_id() const {
        RelearnException::check(is_local(), "TaggedID::get_local_id is not local {:s}", *this);
        return id();
    }

    [[nodiscard]] constexpr bool is_initialized() const noexcept { return is_initialized_; }

    [[nodiscard]] constexpr bool is_virtual() const noexcept { return is_virtual_; }

    [[nodiscard]] constexpr bool is_global() const noexcept { return is_global_; }

    [[nodiscard]] constexpr bool is_local() const noexcept { return !is_global_; }

    [[nodiscard]] constexpr TaggedID operator+(const TaggedID& v) const noexcept {
        auto res = *this;
        res.id_ += v.id();
        return res;
    }

    [[nodiscard]] constexpr TaggedID operator-(const TaggedID& v) const noexcept {
        auto res = *this;
        res.id_ -= v.id();
        return res;
    }

    [[nodiscard]] constexpr TaggedID operator+(const std::integral auto& v) const noexcept {
        auto res = *this;
        res.id_ += v;
        return res;
    }

    [[nodiscard]] constexpr TaggedID operator-(const std::integral auto& v) const noexcept {
        auto res = *this;
        res.id_ -= v;
        return res;
    }

    /**
     * @brief Compare two TaggedIDs
     *
     * Compares the members in order of declaration (defaulted <=> opterator)
     *
     * @return std::strong_ordering ordering
     */
    [[nodiscard]] friend constexpr std::strong_ordering operator<=>(const TaggedID&, const TaggedID&) noexcept = default;

private:
    // the ordering of members is important for the defaulted <=> comparison

    bool is_initialized_ : 1 = false;
    bool is_global_ : 1 = false;
    bool is_virtual_ : 1 = false;
    value_type id_ : id_bit_count = 0;
};

/**
 * @brief Formatter for TaggedID
 *
 * TaggedID is represented as follows:
 * is_initialized is_global is_virtual : id
 * printing the flags is optional
 *
 * Formatting options are:
 * - i (default): id only   -> 123456
 * - s: small               -> 000:123456
 * - m: medium              -> i0g0v0:123456
 * - l: large               -> initialized: bool, global: bool, virtual: bool:123456
 *
 * The id can be formatted with the appropriate
 * formatting for its type.
 * Requirement: TaggedID formatting has to be specified
 * before the formatting of the id.
 * Example: "{:s>20}"
 *
 * @tparam T value type of the TaggedID
 */
template <typename T>
class fmt::formatter<TaggedID<T>> : public fmt::formatter<typename TaggedID<T>::value_type> {
public:
    [[nodiscard]] constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
        const auto* it = ctx.begin();
        const auto* const end = ctx.end();
        if (it != end && (*it == 'i' || *it == 's' || *it == 'm' || *it == 'l')) {
            presentation = *it++; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
            ctx.advance_to(it);
        }

        return fmt::formatter<typename TaggedID<T>::value_type>::parse(ctx);
    }

    template <typename FormatContext>
    [[nodiscard]] auto format(const TaggedID<T>& id, FormatContext& ctx) -> decltype(ctx.out()) {
        switch (presentation) {
        case 'i':
            break;
        case 's':
            format_to(
                ctx.out(),
                "{:1b}{:1b}{:1b}:",
                id.is_initialized(), id.is_global(), id.is_virtual());
            break;
        case 'm':
            format_to(
                ctx.out(),
                "i{:1b}g{:1b}v{:1b}:",
                id.is_initialized(), id.is_global(), id.is_virtual());
            break;
        case 'l':
            format_to(
                ctx.out(),
                "initialized: {:5}, global: {:5}, virtual: {:5}, id: ",
                id.is_initialized(), id.is_global(), id.is_virtual());
            break;
        default:
            throw format_error("unrecognized format for TaggedID<T>");
        }

        return fmt::formatter<typename TaggedID<T>::value_type>::format(id.id(), ctx);
    }

private:
    char presentation = 'i';
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const TaggedID<T>& id) {
    os << fmt::format("{}", id);
    return os;
}

using NeuronID = TaggedID<std::uint64_t>;
