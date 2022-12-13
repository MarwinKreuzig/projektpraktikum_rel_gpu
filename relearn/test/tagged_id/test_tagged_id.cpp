/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_tagged_id.h"

#include "RandomAdapter.h"

#include <cstdint>
#include <functional>
#include <type_traits>

using test_types = ::testing::Types<std::uint16_t, std::int16_t, std::uint32_t, std::int32_t, std::int64_t, std::uint64_t>;
TYPED_TEST_SUITE(TaggedIDTest, test_types);

TYPED_TEST(TaggedIDTest, testTaggedIDUninitialized) { // NOLINT
    const auto id = TaggedID<TypeParam>::uninitialized_id();

    ASSERT_FALSE(id.is_initialized());
    ASSERT_FALSE(static_cast<bool>(id));
    ASSERT_FALSE(id.is_virtual());
    ASSERT_FALSE(id.is_local());

    ASSERT_THROW(auto val = id.get_neuron_id(), RelearnException);
}

TYPED_TEST(TaggedIDTest, testTaggedIDVirtual) { // NOLINT
    const auto id = TaggedID<TypeParam>::virtual_id();

    ASSERT_TRUE(id.is_initialized());
    ASSERT_TRUE(static_cast<bool>(id));
    ASSERT_TRUE(id.is_virtual());
    ASSERT_FALSE(id.is_local());

    ASSERT_THROW(auto val = id.get_neuron_id(), RelearnException);
}

TYPED_TEST(TaggedIDTest, testTaggedIDConstructorDefault) { // NOLINT
    TaggedID<TypeParam> id{};

    ASSERT_FALSE(id.is_initialized());
    ASSERT_FALSE(static_cast<bool>(id));
    ASSERT_FALSE(id.is_virtual());
    ASSERT_FALSE(id.is_local());

    ASSERT_THROW(auto val = id.get_neuron_id(), RelearnException);
}

TYPED_TEST(TaggedIDTest, testTaggedIDConstructorOnlyID) { // NOLINT
    const auto id_val = RandomAdapter::template get_random_integer<TypeParam>(TaggedID<TypeParam>::limits::min, TaggedID<TypeParam>::limits::max, this->mt);

    const TaggedID<TypeParam> id{ id_val };

    ASSERT_TRUE(id.is_initialized());
    ASSERT_TRUE(static_cast<bool>(id));
    ASSERT_FALSE(id.is_virtual());
    ASSERT_TRUE(id.is_local());

    ASSERT_EQ(id.get_neuron_id(), id_val);
    ASSERT_EQ(static_cast<TypeParam>(id), id_val);
}

TYPED_TEST(TaggedIDTest, testTaggedIDConstructorLocal) {
    const auto id_val = RandomAdapter::template get_random_integer<TypeParam>(TaggedID<TypeParam>::limits::min, TaggedID<TypeParam>::limits::max, this->mt);

    const TaggedID<TypeParam> id{ false, id_val };

    ASSERT_TRUE(id.is_initialized());
    ASSERT_TRUE(static_cast<bool>(id));
    ASSERT_FALSE(id.is_virtual());
    ASSERT_TRUE(id.is_local());

    ASSERT_EQ(id.get_neuron_id(), id_val);
    ASSERT_EQ(static_cast<TypeParam>(id), id_val);
}

TYPED_TEST(TaggedIDTest, testTaggedIDConstructorVirtual) {
    const auto id_val = RandomAdapter::template get_random_integer<TypeParam>(TaggedID<TypeParam>::limits::min, TaggedID<TypeParam>::limits::max, this->mt);

    const TaggedID<TypeParam> id{ true, id_val };

    ASSERT_TRUE(id.is_initialized());
    ASSERT_TRUE(static_cast<bool>(id));
    ASSERT_TRUE(id.is_virtual());
    ASSERT_FALSE(id.is_local());

    ASSERT_THROW(auto val = id.get_neuron_id(), RelearnException);
}

TYPED_TEST(TaggedIDTest, testTaggedIDComparisons1) { // NOLINT
    constexpr static auto min = TaggedID<TypeParam>::limits::min;
    constexpr static auto max = TaggedID<TypeParam>::limits::max;

    const auto get_random_id = [this]() { return TaggedID<TypeParam>{ RandomAdapter::template get_random_integer<TypeParam>(min, max, this->mt) }; };

    const auto id1 = get_random_id();
    const auto id2 = get_random_id();

    ASSERT_EQ(id1 <=> id2, id1.get_neuron_id() <=> id2.get_neuron_id());
    ASSERT_EQ(TaggedID<TypeParam>{}, TaggedID<TypeParam>{});
}

TYPED_TEST(TaggedIDTest, testTaggedIDComparisons2) { // NOLINT
    constexpr static auto min = TaggedID<TypeParam>::limits::min;
    constexpr static auto max = TaggedID<TypeParam>::limits::max;

    const auto get_random_id = [this]() {
        auto res = TaggedID<TypeParam>{
            RandomAdapter::get_random_bool(this->mt),
            RandomAdapter::template get_random_integer<TypeParam>(min, max, this->mt)
        };

        // res.is_initialized() = this->RandomAdapter::get_random_bool(this->mt);
        return res;
    };

    const auto id1 = get_random_id();
    const auto id2 = get_random_id();

    // test for == and != behavior
    // only compare equal if all members compare equal, otherwise compare unequal
    ASSERT_EQ(
        id1 == id2,
        this->get_initialized(id1) == this->get_initialized(id2)
            && this->get_virtual(id1) == this->get_virtual(id2)
            && this->get_id(id1) == this->get_id(id2));

    std::stringstream ss{};

    ss << "ID 1: (" << this->get_id(id1) << ", " << id1.is_initialized() << ", " << id1.is_virtual() << ")\n";
    ss << "ID 2: (" << this->get_id(id2) << ", " << id2.is_initialized() << ", " << id2.is_virtual() << ")\n";

    const auto comp = id1 <=> id2;
    const auto initialized_comparison =  this->get_initialized(id1) <=> this->get_initialized(id2);
    const auto virtual_comparison = this->get_virtual(id1) <=> this->get_virtual(id2);
    const auto id_comparison = this->get_id(id1) <=> this->get_id(id2);

    // members are compared in order of declaration
    // -> if any compares not equal,
    // then the result of that first comparison that is not equal
    // is the result of the comparison

    if (initialized_comparison != 0) {
        EXPECT_EQ(comp, initialized_comparison) << ss.str();
        return;
    }

    if (virtual_comparison != 0) {
        EXPECT_EQ(comp, virtual_comparison) << ss.str();
        return;
    }

    if (id_comparison != 0) {
        EXPECT_EQ(comp, id_comparison) << ss.str();
        return;
    }
}
