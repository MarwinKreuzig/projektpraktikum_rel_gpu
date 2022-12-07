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

#include "RelearnTest.hpp"

template <typename T>
class TaggedIDTest : public RelearnTest {
protected:
    static void SetUpTestSuite() {
        SetUpTestCaseTemplate();
    }

    static bool get_initialized(const TaggedID<T>& id) {
        return id.is_initialized_;
    }

    static bool get_virtual(const TaggedID<T>& id) {
        return id.is_virtual_;
    }

    static bool get_global(const TaggedID<T>& id) {
        return id.is_global_;
    }

    static typename TaggedID<T>::value_type get_id(const TaggedID<T>& id) {
        return id.id_;
    }

    static_assert(sizeof(typename TaggedID<T>::value_type) == sizeof(T));
};
