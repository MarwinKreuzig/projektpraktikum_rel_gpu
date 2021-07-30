/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "MPINo_RMA_MemAllocator.h"

#if !RELEARN_MPI_FOUND

#include "../algorithm/BarnesHutCell.h"
#include "../algorithm/FastMultipoleMethodsCell.h"
#include "../io/LogFiles.h"
#include "../structure/OctreeNode.h"
#include "../util/RelearnException.h"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <sstream>

template class MPINo_RMA_MemAllocator<BarnesHutCell>;
template class HolderOctreeNode<BarnesHutCell>;

template class MPINo_RMA_MemAllocator<FastMultipoleMethodsCell>;
template class HolderOctreeNode<FastMultipoleMethodsCell>;

template <typename AdditionalCellAttributes>
inline void HolderOctreeNode<AdditionalCellAttributes>::make_available(OctreeNode<AdditionalCellAttributes>* ptr) {
    const size_t dist = std::distance(base_ptr, ptr);

    available.push(ptr);
    non_available[dist] = nullptr;

    ptr->reset();
}

template <typename AdditionalCellAttributes>
inline void HolderOctreeNode<AdditionalCellAttributes>::make_all_available() noexcept {
    for (auto& ptr : non_available) {
        if (ptr == nullptr) {
            continue;
        }

        available.push(ptr);

        ptr->reset();
        ptr = nullptr;
    }
}

template <typename AdditionalCellAttributes>
void MPINo_RMA_MemAllocator<AdditionalCellAttributes>::init(size_t size_requested) {
    MPINo_RMA_MemAllocator::size_requested = size_requested;
    max_num_objects = size_requested / sizeof(OctreeNode<AdditionalCellAttributes>);
    max_size = size_requested;

    base_ptr = new OctreeNode<AdditionalCellAttributes>[max_num_objects];

    // create_rma_window();
    base_pointers = reinterpret_cast<int64_t>(base_ptr);

    holder_base_ptr = HolderOctreeNode<AdditionalCellAttributes>(base_ptr, max_num_objects);

    LogFiles::print_message_rank(0, "MPI RMA MemAllocator: max_num_objects: {}  sizeof(OctreeNode): {}", max_num_objects, sizeof(OctreeNode<AdditionalCellAttributes>));
}

template <typename AdditionalCellAttributes>
void MPINo_RMA_MemAllocator<AdditionalCellAttributes>::finalize() {
    delete base_ptr;
}

#endif
