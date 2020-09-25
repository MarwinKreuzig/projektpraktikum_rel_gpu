#pragma once

#include <random>

namespace randomNumberSeeds {
	extern long int partition;
	extern long int octree;
}

template<typename T>
class RandomHolder {
private:
	static std::mt19937 random_generator;

public:
	static std::mt19937& get_random_generator() noexcept {
		return random_generator;
	}
};

template<typename T>
std::mt19937 RandomHolder<T>::random_generator{};
