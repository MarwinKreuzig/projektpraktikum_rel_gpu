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

/**
  * An instance of this enum signals if a neuron should be updated or not.
  */
enum class UpdateStatus : char { DISABLED = 0,
    ENABLED = 1};
