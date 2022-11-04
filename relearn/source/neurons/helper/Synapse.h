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

#include <compare>
#include <utility>

/**
 * This class encapsulates what a synapse is made of in this program,
 *  i.e., a target, a source, and a weight.
 * @tparam Target The type of the target
 * @tparam Source The type of the source
 * @tparam Weight The type of the weight
 */
template <typename Target, typename Source, typename Weight, typename ExtraInfo>
class Synapse {
public:
    /**
     * @brief Constructs a new synapse with the given parameter
     * @param target The target of the synapse
     * @param source The source of the synapse
     * @param weight The weight of the synapse
     */
    Synapse(const Target target, const Source source, const Weight weight, const ExtraInfo extra_info)
        : target(target)
        , source(source)
        , weight(weight)
        , extra_info(extra_info){ }

    /**
     * @brief Returns the target of the synapse
     * @return The target of the synapse
     */
    [[nodiscard]] const Target& get_target() const noexcept {
        return target;
    }

    /**
     * @brief Returns the source of the synapse
     * @return The source of the synapse
     */
    [[nodiscard]] const Source& get_source() const noexcept {
        return source;
    }

    /**
     * @brief Returns the weight of the synapse
     * @return The weight of the synapse
     */
    [[nodiscard]] const Weight& get_weight() const noexcept {
        return weight;
    }

    [[nodiscard]] const ExtraInfo& get_extra_info() const noexcept {
        return extra_info;
    }

    [[nodiscard]] friend constexpr std::strong_ordering operator<=>(const Synapse& first, const Synapse& second) noexcept = default;

    template <std::size_t Index>
    [[nodiscard]] auto& get() & {
        if constexpr (Index == 0) {
            return target;
        }
        if constexpr (Index == 1) {
            return source;
        }
        if constexpr (Index == 2) {
            return weight;
        }
        if constexpr (Index == 3) {
            return extra_info;
        }
    }

    template <std::size_t Index>
    [[nodiscard]] auto const& get() const& {
        if constexpr (Index == 0) {
            return target;
        }
        if constexpr (Index == 1) {
            return source;
        }
        if constexpr (Index == 2) {
            return weight;
        }
        if constexpr (Index == 3) {
            return extra_info;
        }
    }

    template <std::size_t Index>
    [[nodiscard]] auto&& get() && {
        if constexpr (Index == 0) {
            return std::move(target);
        }
        if constexpr (Index == 1) {
            return std::move(source);
        }
        if constexpr (Index == 2) {
            return std::move(weight);
        }
        if constexpr (Index == 3) {
            return std::move(extra_info);
        }
    }

private:
    Target target{};
    Source source{};
    Weight weight{};
    ExtraInfo extra_info{};
};

namespace std {
template <typename Target, typename Source, typename Weight, typename ExtraInfo>
struct tuple_size<::Synapse<Target, Source, Weight, ExtraInfo>> {
    static constexpr size_t value = 4;
};

template <typename Target, typename Source, typename Weight, typename ExtraInfo>
struct tuple_element<0, ::Synapse<Target, Source, Weight, ExtraInfo>> {
    using type = Target;
};

template <typename Target, typename Source, typename Weight, typename ExtraInfo>
struct tuple_element<1, ::Synapse<Target, Source, Weight, ExtraInfo>> {
    using type = Source;
};

template <typename Target, typename Source, typename Weight, typename ExtraInfo>
struct tuple_element<2, ::Synapse<Target, Source, Weight, ExtraInfo>> {
    using type = Weight;
};

template <typename Target, typename Source, typename Weight, typename ExtraInfo>
struct tuple_element<3, ::Synapse<Target, Source, Weight, ExtraInfo>> {
    using type = ExtraInfo;
};

} // namespace std
