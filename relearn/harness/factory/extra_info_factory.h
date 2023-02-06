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
        std::vector<NeuronID> ids_to_enable{};

        const auto disable_flags = extra_info->get_disable_flags();
        for (size_t i = 0; i < disable_flags.size(); i++) {
            if (disable_flags[i] == UpdateStatus::Disabled) {
                ids_to_enable.emplace_back(i);
            }
        }

        extra_info->set_enabled_neurons(std::move(ids_to_enable));
    }

    static void disable_all(const std::shared_ptr<NeuronsExtraInfo>& extra_info) {
        std::vector<NeuronID> ids_to_disable{};

        const auto disable_flags = extra_info->get_disable_flags();
        for (size_t i = 0; i < disable_flags.size(); i++) {
            if (disable_flags[i] == UpdateStatus::Enabled) {
                ids_to_disable.emplace_back(i);
            }
        }

        extra_info->set_disabled_neurons(std::move(ids_to_disable));
    }
};
