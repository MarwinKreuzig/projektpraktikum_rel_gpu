#pragma once

#include <random>

//#define private public
//#define protected public

constexpr int iterations = 10;
constexpr double eps = 0.00001;
constexpr size_t num_neurons_test = 1000;


extern std::mt19937 mt;

void setup();
