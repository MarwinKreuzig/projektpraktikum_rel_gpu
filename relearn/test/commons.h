#pragma once

#include <random>

//#define private public
#define protected public


constexpr const int iterations = 10;
constexpr const double eps = 0.00001;


extern std::mt19937 mt;

void setup();
