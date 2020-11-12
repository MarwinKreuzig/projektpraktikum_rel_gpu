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
