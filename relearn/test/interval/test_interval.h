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

#include "util/Interval.h"

#include <climits>
#include <string>
#include <utility>

class IntervalTest : public RelearnTest {
protected:
    static void SetUpTestSuite() {
        SetUpTestCaseTemplate();
    }
};
