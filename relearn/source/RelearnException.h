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

#include <exception>
#include <iostream>
#include <string>

class RelearnException : std::exception {
private:	
	std::string message;

public:
	static bool hide_messages;

	RelearnException() = default;

	explicit RelearnException(std::string&& mes) : message(mes) {

	}

	[[nodiscard]] const char* what() const noexcept override;

	static void check(bool condition);

	static void fail();

	static void check(bool condition, std::string&& message);
	
	static void fail(std::string&& message);
};
