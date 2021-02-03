/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "OctreeNode.h"

#include "LogFiles.h"

#include <sstream>

void OctreeNode::print() const {
    std::stringstream ss;

    ss << "== OctreeNode (" << this << ") ==\n";

    ss << "  children[8]: ";
    for (const auto* const child : children) {
        ss << child << " ";
    }
    ss << "\n";

    ss << "  is_parent  : " << parent << "\n\n";
    ss << "  rank       : " << rank << "\n";
    ss << "  level      : " << level << "\n\n";

    cell.print();

    ss << "\n";

    LogFiles::write_to_file(LogFiles::EventType::Cout, ss.str(), true);
}
