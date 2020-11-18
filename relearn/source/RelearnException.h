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
	std::string message;

public:
	RelearnException() noexcept {

	}

	explicit RelearnException(std::string&& mes) : message(mes) {

	}

	const char* what() const noexcept override {
		return message.c_str();
	}

	static void check(bool condition) {
		if (condition) {
			return;
		}

		std::cerr << "There was an error!" << std::endl;
		std::cerr << "But no error message" << std::endl;

		throw RelearnException{};
	}

	static void check(bool condition, std::string&& message) {
		if (condition) {
			return;
		}

		std::cerr << "There was an error!" << std::endl;
		std::cerr << message << std::endl;

		throw RelearnException{ std::move(message) };
	}
};
