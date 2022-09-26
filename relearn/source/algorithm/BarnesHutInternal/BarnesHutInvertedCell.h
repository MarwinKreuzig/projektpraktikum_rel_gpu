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

#include "structure/BaseCell.h"

// For the inverted BH, we need the axons
using BarnesHutInvertedCell = BaseCell<false, false, true, true>;
