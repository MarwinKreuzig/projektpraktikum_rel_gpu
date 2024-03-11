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
//std::vector<OctreeNode<BarnesHutInvertedCell>> holder_bhi_cells{};
//std::vector<OctreeNode<FastMultipoleMethodCell>> holder_fmm_cells{};


int main(int argc, char** argv) {
    MPIWrapper::init(argc, argv);

    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;



//    const auto octree_node_size = sizeof(OctreeNode<AdditionalCellAttributes>);
//    const auto max_num_objects = Constants::mpi_alloc_mem / octree_node_size;
//
//    create_rma_window<OctreeNode<BarnesHutCell>>(MPIWindow::Window::Octree, max_num_objects, 1);
//    auto& data = MPIWindow::mpi_windows[MPIWindow::Window::Octree];
//    auto& base_ptr = std::any_cast<std::vector<OctreeNode<AdditionalCellAttributes>>&>(data);

//    std::span<OctreeNode<BarnesHutCell>> span{ base_ptr.data(), max_num_objects };
    holder_bh_cells.resize(1024 * 1024 * 4);
//    holder_bhi_cells.resize(1024 * 1024);
//    holder_fmm_cells.resize(1024 * 1024);

    MemoryHolder<BarnesHutCell>::init(holder_bh_cells);
//    MemoryHolder<BarnesHutInvertedCell>::init(holder_bhi_cells);
//    MemoryHolder<FastMultipoleMethodCell>::init(holder_fmm_cells);
//    MemoryHolder<BarnesHutCell>::init(span);


    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}
