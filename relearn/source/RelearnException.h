#pragma once

#include <exception>
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

	static void check(bool condition, const char* message) {
		if (condition) {
			return;
		}

		throw RelearnException{ message };
	}
};
