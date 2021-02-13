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

#if MPI_FOUND
#include <mpi.h>
#else
using MPI_Comm = int;
using MPI_Datatype = int;
using MPI_Op = int;
using MPI_Request = int;
using MPI_Win = int;

constexpr inline auto MPI_DOUBLE = 0; // TMP
constexpr inline auto MPI_CHAR = 'c'; // TMP

constexpr inline auto MPI_LOCK_EXCLUSIVE = 0;
constexpr inline auto MPI_LOCK_SHARED = 1;
#endif
