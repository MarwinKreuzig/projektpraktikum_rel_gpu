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

#ifdef __has_include

#if __has_include(<mpi.h>)
#include <mpi.h>
#else
using MPI_Comm = int;
using MPI_Datatype = int;
using MPI_Op = int;
using MPI_Request = int;
using MPI_Win = int;

#define MPI_LOCK_EXCLUSIVE 0
#define MPI_LOCK_SHARED 1
#endif

#else

#if __has_include(<mpi.h>)
#include <mpi.h>
#else
#error "The compiler is missing __has_include!"
#endif

#endif