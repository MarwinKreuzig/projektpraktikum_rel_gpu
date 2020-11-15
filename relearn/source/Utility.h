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

namespace Util {
	constexpr  int num_digits(int val) /*noexcept*/ {
		int num_digits = 0;

		do {
			++num_digits;
			val /= 10;
		} while (val);

		return num_digits;
	}
}
