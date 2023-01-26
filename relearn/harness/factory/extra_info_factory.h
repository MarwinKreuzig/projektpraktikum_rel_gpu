#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "neurons/NeuronsExtraInfo.h"
#include "util/TaggedID.h"

#include <memory>

class NeuronsExtraInfoFactory {
public:
    static std::shared_ptr<NeuronsExtraInfo> construct_extra_info() {
        return std::make_shared<NeuronsExtraInfo>();
    }

    static void enable_all(const std::shared_ptr<NeuronsExtraInfo>& extra_info) {
        extra_info->set_enabled_neurons(NeuronID::range(extra_info->get_size()));
    }

    static void disable_all(const std::shared_ptr<NeuronsExtraInfo>& extra_info) {
        extra_info->set_enabled_neurons(NeuronID::range(extra_info->get_size()));
    }
};
