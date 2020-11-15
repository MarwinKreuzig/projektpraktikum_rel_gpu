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

#include <mpi.h>

#include <algorithm>

namespace MPIUserDefinedOperation {
	// This combination function assumes that it's called with the
	// correct MPI datatype
	void min_sum_max(const int* invec, int* inoutvec, const int* const len, MPI_Datatype* dtype) /*noexcept*/ {
		const double* in = (const double*)invec;
		double*inout = (double*)inoutvec;

		for (int i = 0; i < *len; i++) {
			inout[3 * i] = std::min(in[3 * i], inout[3 * i]);
			inout[3 * i + 1] += in[3 * i + 1];
			inout[3 * i + 2] = std::max(in[3 * i + 2], inout[3 * i + 2]);
		}
	}
}
