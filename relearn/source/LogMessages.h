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

class LogMessages {
public:

	/**
	 * Static functions for printing a tagged log message to std::cout
	 */
	static void print_message(char const* string);

	// Print tagged message only at MPI rank "rank"
	static void print_message_rank(char const* string, int rank);

	static void print_error(char const* string);

	static void print_debug(char const* string);
};
