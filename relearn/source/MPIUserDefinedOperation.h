/*
 * File:   MPIUserDefinedOperation.h
 * Author: rinke
 *
 * Created on Mar 21, 2017
 */

#ifndef MPIUSERDEFINEDOPERATION_H
#define MPIUSERDEFINEDOPERATION_H

#include <algorithm>

namespace MPIUserDefinedOperation {

	// This combination function assumes that it's called with the
	// correct MPI datatype
	void min_sum_max(int* invec, int* inoutvec, int* len, MPI_Datatype* dtype) {
		double* in = (double*)invec, * inout = (double*)inoutvec;

		for (int i = 0; i < *len; i++) {
			inout[3 * i] = std::min(in[3 * i], inout[3 * i]);
			inout[3 * i + 1] += in[3 * i + 1];
			inout[3 * i + 2] = std::max(in[3 * i + 2], inout[3 * i + 2]);
		}
	}
}

#endif /* MPIUSERDEFINEDOPERATION */
