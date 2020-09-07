/*
 * File:   SynapticElements.cpp
 * Author: naveau
 *
 * Created on September 26, 2014, 2:33 PM
 */

#include "SynapticElements.h"
#include <algorithm>
#include <cstdlib>


SynapticElements::SynapticElements(ElementType type, size_t s, double min_C_level_to_grow, double C_target, double nu, double vacant_retract_ratio) :
	type(type),
	size(s),
	min_C_level_to_grow(min_C_level_to_grow),
	C_target(C_target),
	nu(nu),
	vacant_retract_ratio(vacant_retract_ratio) {
	cnts = new double[size];
	connected_cnts = new double[size];
	delta_cnts = new double[size];
	signal_types = new SignalType[size];

	// Init num elements to 0
	for (size_t i = 0; i < size; i++) {
		cnts[i] = 2;
		connected_cnts[i] = 0;
		delta_cnts[i] = 0;
	}
}

SynapticElements::~SynapticElements() {
	delete[] cnts;
	delete[] connected_cnts;
	delete[] delta_cnts;
	delete[] signal_types;
}
