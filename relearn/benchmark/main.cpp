/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "main.h"

#include "mpi/MPIWrapper.h"
#include "mpi/MPINoWrapper.h"
#include "algorithm/Cells.h"
#include "structure/OctreeNode.h"
#include "util/MemoryHolder.h"

std::vector<OctreeNode<BarnesHutCell>> holder_bh_cells{};

int main(int argc, char** argv) {
    MPIWrapper::init(argc, argv);

    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;


    holder_bh_cells.resize(1024 * 1024 * 4);
    MemoryHolder<BarnesHutCell>::init(holder_bh_cells);


    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}
