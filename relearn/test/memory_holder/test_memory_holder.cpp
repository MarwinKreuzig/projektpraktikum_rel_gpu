/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "test_memory_holder.h"

#include "algorithm/Cells.h"
#include "structure/OctreeNode.h"
#include "util/MemoryHolder.h"
#include "util/RelearnException.h"
#include "util/Vec3.h"

using test_types = ::testing::Types<BarnesHutCell, BarnesHutInvertedCell, FastMultipoleMethodsCell, NaiveCell>;
TYPED_TEST_SUITE(MemoryHolderTest, test_types);

TYPED_TEST(MemoryHolderTest, testEmpty) {
    //using AdditionalCellAttributes = TypeParam;
    //using MH = MemoryHolder<AdditionalCellAttributes>;

    //std::vector<OctreeNode<AdditionalCellAttributes>> memory(1024, OctreeNode<AdditionalCellAttributes>{});

    //MH::init(memory);
    //ASSERT_EQ(MH::get_size(), 1024);

    //MH::make_all_available();
    //ASSERT_EQ(MH::get_size(), 1024);

    //std::vector<OctreeNode<AdditionalCellAttributes>> memory2(1024 * 1024, OctreeNode<AdditionalCellAttributes>{});

    //MH::init(memory2);
    //ASSERT_EQ(MH::get_size(), 1024 * 1024);
}
