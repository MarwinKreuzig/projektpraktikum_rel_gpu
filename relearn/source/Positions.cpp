/*
 * File:   Positions.cpp
 * Author: naveau
 *
 * Created on September 26, 2014, 1:28 PM
 */

#include <algorithm>
#include <iterator>
#include <cmath>
#include <sstream>

#include "Positions.h"
#include "LogMessages.h"

Positions::Positions(size_t s) :
	size(s) {
	x_dims = new double[size];
	y_dims = new double[size];
	z_dims = new double[size];
}

Positions::~Positions() {
	delete[] x_dims;
	delete[] y_dims;
	delete[] z_dims;
}
