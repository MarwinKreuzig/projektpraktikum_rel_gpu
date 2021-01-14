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
	template <typename T>
	constexpr unsigned int num_digits(T val) noexcept {
		unsigned int num_digits = 0;

		do {
			++num_digits;
			val /= 10;
		} while (val != 0);

		return num_digits;
	}
} // namespace Util
