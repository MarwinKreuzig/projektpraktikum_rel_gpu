#include "RelearnTest.hpp"
#include <bits/ranges_algo.h>
#include <cstdint>
#include <functional>
#include <gtest/gtest-typed-test.h>
#include <gtest/gtest.h>
#include <spdlog/fmt/bundled/core.h>
#include <type_traits>

using test_types = ::testing::Types<std::uint8_t, std::uint16_t, std::int16_t, typename NeuronID::value_type>;
TYPED_TEST_SUITE(TaggedIDTest, test_types);

template <typename T>
[[nodiscard]] static T foo() {
    return ~T{ 0 } >> T{ 3 };
}

TYPED_TEST(TaggedIDTest, testTaggedIDConstructorDefault) { // NOLINT
    TaggedID<TypeParam> id{};
    ASSERT_FALSE(id.is_initialized);
    ASSERT_FALSE(id.is_global);
    ASSERT_FALSE(id.is_virtual);
}

TYPED_TEST(TaggedIDTest, testTaggedIDConstructorOnlyID) { // NOLINT
    const auto id_val = get_random_integer<TypeParam>(TaggedID<TypeParam>::limits::min, TaggedID<TypeParam>::limits::max);

    const TaggedID<TypeParam> id{ id_val };

    ASSERT_EQ(id.id, id_val);
    ASSERT_EQ(static_cast<TypeParam>(id), id_val);
    ASSERT_TRUE(id.is_initialized);
    ASSERT_TRUE(static_cast<bool>(id));
    ASSERT_FALSE(id.is_global);
    ASSERT_FALSE(id.is_virtual);
}

TYPED_TEST(TaggedIDTest, testTaggedIDConstructorFlagsAndID) { // NOLINT
    const auto is_global = TaggedIDTest<TypeParam>::get_random_bool();
    const auto is_virtual = TaggedIDTest<TypeParam>::get_random_bool();
    const auto id_val = get_random_integer<TypeParam>(TaggedID<TypeParam>::limits::min, TaggedID<TypeParam>::limits::max);

    const TaggedID<TypeParam> id{ is_global, is_virtual, id_val };

    ASSERT_EQ(id.id, id_val);
    ASSERT_EQ(static_cast<TypeParam>(id), id_val);
    ASSERT_TRUE(id.is_initialized);
    ASSERT_TRUE(static_cast<bool>(id));
    ASSERT_EQ(id.is_global, is_global);
    ASSERT_EQ(id.is_virtual, is_virtual);
}

TYPED_TEST(TaggedIDTest, testTaggedIDUninitialized) { // NOLINT
    const auto id = TaggedID<TypeParam>::uninitialized_id();

    ASSERT_FALSE(id.is_initialized);
    ASSERT_FALSE(id.is_global);
    ASSERT_FALSE(id.is_virtual);
}

TYPED_TEST(TaggedIDTest, testTaggedIDVirtual) { // NOLINT
    const auto id = TaggedID<TypeParam>::virtual_id();

    ASSERT_TRUE(id.is_initialized);
    ASSERT_TRUE(static_cast<bool>(id));
    ASSERT_FALSE(id.is_global);
    ASSERT_TRUE(id.is_virtual);
}

TYPED_TEST(TaggedIDTest, testTaggedIDAssignToUninitializeForInitialization) { // NOLINT
    auto id = TaggedID<TypeParam>::uninitialized_id();
    const auto id_val = get_random_integer<TypeParam>(TaggedID<TypeParam>::limits::min, TaggedID<TypeParam>::limits::max);
    id = id_val;

    ASSERT_EQ(id.id, id_val);
    ASSERT_EQ(static_cast<TypeParam>(id), id_val);
    ASSERT_TRUE(id.is_initialized);
    ASSERT_TRUE(static_cast<bool>(id));
    ASSERT_FALSE(id.is_global);
    ASSERT_FALSE(id.is_virtual);
}

TYPED_TEST(TaggedIDTest, testTaggedIDIncrementDecrement) { // NOLINT
    constexpr auto edge_distance = TypeParam{ 3 };
    auto id_val = get_random_integer<TypeParam>(TaggedID<TypeParam>::limits::min + edge_distance, TaggedID<TypeParam>::limits::max - edge_distance);

    TaggedID<TypeParam> id{ id_val };
    const auto old = id;

    ASSERT_EQ(static_cast<TypeParam>(--id), --id_val);
    ASSERT_EQ(static_cast<TypeParam>(++id), ++id_val);
    ASSERT_EQ(static_cast<TypeParam>(id--), id_val--);
    ASSERT_EQ(static_cast<TypeParam>(id), id_val);
    ASSERT_EQ(static_cast<TypeParam>(id++), id_val++);
    ASSERT_EQ(static_cast<TypeParam>(id), id_val);
    ASSERT_EQ(id, old);
}

TYPED_TEST(TaggedIDTest, testTaggedIDArithmetic) { // NOLINT
    constexpr static auto min = TaggedID<TypeParam>::limits::min;
    constexpr static auto max = TaggedID<TypeParam>::limits::max;

    const auto get_rand = [this](const auto min2, const auto max2) {
        const auto rand = get_random_integer<TypeParam>(min2, max2);
        return rand;
    };

    const auto initial_val = get_rand(min, max);
    const auto id_val = initial_val;

    const TaggedID<TypeParam> id{ id_val };

    const auto id_distance_to_min = id_val - min;
    const auto id_distance_to_max = max - id_val;
    constexpr static auto is_unsigned = std::is_unsigned_v<TypeParam>;

    const auto verify =
        [id_distance_to_min,
            id_distance_to_max](const auto tagged_id,
            const auto id_val,
            const auto random_val,
            const std::string_view operation_name,
            const auto& operation) {
            ASSERT_EQ(
                static_cast<TypeParam>(
                    operation(tagged_id, random_val)),
                operation(id_val, random_val))
                << fmt::format("operation: {}\noriginal_val: {}\nrandom_val: {}\nmin: {}\nmax: {}\nid_distance_to_min: {}\nid_distance_to_max: {}\n",
                       operation_name,
                       id_val,
                       random_val,
                       min,
                       max,
                       id_distance_to_min,
                       id_distance_to_max);
        };

    // TaggedID& operator std::integral&
    // implemented in terms of:
    // TaggedID& operator compound-assign const std::integral&
    verify(
        id,
        id_val,
        get_rand(is_unsigned ? 0 : -id_distance_to_min, id_distance_to_max),
        "TaggedID& operator+ std::integral&",
        [](auto v, const auto& r) { return v + r; });

    verify(
        id,
        id_val,
        get_rand(is_unsigned ? 0 : -id_distance_to_max, id_distance_to_min),
        "TaggedID& operator- std::integral&",
        [](auto v, const auto& r) { return v - r; });

    verify(
        id,
        id_val,
        get_rand(1, max),
        "TaggedID& operator% std::integral&",
        [](auto v, const auto& r) { return v % r; });

    // TaggedID& operator TaggedID&
    // implemented in terms of:
    // TaggedID& operator compound-assign TaggedID&
    // implemented in terms of:
    // TaggedID& operator compound-assign std::integral&
    verify(
        id,
        id_val,
        TaggedID<TypeParam>{ get_rand(is_unsigned ? 0 : std::max<TypeParam>(-id_distance_to_min, min), std::min<TypeParam>(id_distance_to_max, max)) },
        "TaggedID& operator+ TaggedID&",
        [](auto v, const auto& r) {
            if constexpr (std::is_same_v<decltype(v), TaggedID<TypeParam>>) {
                return v + r;
            } else {
                return v + r.id;
            }
        });

    verify(
        id,
        id_val,
        TaggedID<TypeParam>{ get_rand(is_unsigned ? 0 : std::max<TypeParam>(-id_distance_to_max, min), std::min<TypeParam>(id_distance_to_min, max)) },
        "TaggedID& operator- TaggedID&",
        [](auto v, const auto& r) {
            if constexpr (std::is_same_v<decltype(v), TaggedID<TypeParam>>) {
                return v - r;
            } else {
                return v - r.id;
            }
        });

    verify(
        id,
        id_val,
        TaggedID<TypeParam>{ get_rand(1, max) },
        "TaggedID& operator% TaggedID&",
        [](auto v, const auto& r) {
            if constexpr (std::is_same_v<decltype(v), TaggedID<TypeParam>>) {
                return v % r;
            } else {
                return v % r.id;
            }
        });
}

TYPED_TEST(TaggedIDTest, testTaggedIDComparisons1) { // NOLINT
    constexpr static auto min = TaggedID<TypeParam>::limits::min;
    constexpr static auto max = TaggedID<TypeParam>::limits::max;

    const auto get_random_id = [this]() { return TaggedID<TypeParam>{ get_random_integer<TypeParam>(min, max) }; };

    const auto id1 = get_random_id();
    const auto id2 = get_random_id();

    EXPECT_EQ(id1 <=> id2, id1.id <=> id2.id);
    EXPECT_EQ(TaggedID<TypeParam>{}, TaggedID<TypeParam>{});
}

TYPED_TEST(TaggedIDTest, testTaggedIDComparisons2) { // NOLINT
    constexpr static auto min = TaggedID<TypeParam>::limits::min;
    constexpr static auto max = TaggedID<TypeParam>::limits::max;

    const auto get_random_id = [this]() {
        auto res = TaggedID<TypeParam>{
            this->get_random_bool(),
            this->get_random_bool(),
            get_random_integer<TypeParam>(min, max)
        };

        res.is_initialized = this->get_random_bool();
        return res;
    };

    const auto id1 = get_random_id();
    const auto id2 = get_random_id();

    // test for == and != behavior
    // only compare equal if all members compare equal, otherwise compare unequal
    EXPECT_EQ(
        id1 == id2,
        id1.is_initialized == id2.is_initialized
            && id1.is_global == id2.is_global
            && id1.is_virtual == id2.is_virtual
            && id1.id == id2.id);

    const auto comp = id1 <=> id2;
    const auto initialized_comparison = id1.is_initialized <=> id2.is_initialized;
    const auto global_comprison = id1.is_global <=> id2.is_global;
    const auto virtual_comparison = id1.is_virtual <=> id2.is_virtual;
    const auto id_comparison = id1.id <=> id2.id;

    // members are compared in order of declaration
    // -> if any compares not equal,
    // then the result of that first comparison that is not equal
    // is the result of the comparison

    if (initialized_comparison != 0) {
        EXPECT_EQ(comp, initialized_comparison);
        return;
    }

    if (global_comprison != 0) {
        EXPECT_EQ(comp, global_comprison);
        return;
    }

    if (virtual_comparison != 0) {
        EXPECT_EQ(comp, virtual_comparison);
        return;
    }

    if (id_comparison != 0) {
        EXPECT_EQ(comp, id_comparison);
        return;
    }
}
