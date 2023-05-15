/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "FiredStatusApproximator.h"

#include "mpi/CommunicationMap.h"
#include "mpi/MPIWrapper.h"
#include "neurons/NetworkGraph.h"
#include "util/Random.h"

void FiredStatusApproximator::set_local_fired_status(const step_type step, const std::span<const FiredStatus> fired_status) {
    RelearnException::check(accumulated_fired.size() == fired_status.size(), "FiredStatusApproximator::set_local_fired_status: Mismatching sizes: {} vs. {}", accumulated_fired.size(), fired_status.size());

    for (auto i = size_t(0); i < fired_status.size(); i++) {
        if (fired_status[i] == FiredStatus::Fired) {
            accumulated_fired[i]++;
        }
    }
}

void FiredStatusApproximator::exchange_fired_status(const step_type step) {
    // This is a no-op
}

bool FiredStatusApproximator::contains(const MPIRank rank, const NeuronID neuron_id) const {
    const auto rank_it = rank.get_rank();
    RelearnException::check(rank_it < firing_rate_cache.size(), "FiredStatusApproximator::contains: Rank {} is too large for the sizes {}.", rank, firing_rate_cache.size());

    const auto& rank_cache = firing_rate_cache[rank_it];

    const auto pos = rank_cache.find(neuron_id);
    if (pos == rank_cache.end()) {
        return false;
    }

    const auto firing_rate = pos->second;

    const auto random_number = RandomHolder::get_random_uniform_double(RandomHolderKey::FiringStatusApproximator, 0.0, 1.0);
    return firing_rate >= random_number;
}

void FiredStatusApproximator::notify_of_plasticity_change(const step_type step) {
    RelearnException::check(last_synced <= step, "FiredStatusApproximator::notify_of_plasticity_change: step is smaller than last_synced: {} > {}", last_synced, step);
    const auto steps_since_last_sync = step - last_synced;

    if (steps_since_last_sync > 0) {
        const auto steps_since_last_sync_inv = 1.0 / steps_since_last_sync;
        for (auto i = size_t(0); i < accumulated_fired.size(); i++) {
            latest_firing_rate[i] = steps_since_last_sync_inv * static_cast<double>(accumulated_fired[i]);
        }

        std::ranges::fill(accumulated_fired, 0);
    }

    struct communication_type {
        NeuronID neuron_id;
        double firing_rate;
    };

    const auto number_local_neurons = get_number_local_neurons();

    const auto size_hint = std::min(get_number_ranks(), number_local_neurons);
    CommunicationMap<communication_type> outgoing_firing_rates{ get_number_ranks(), size_hint };

    auto add_to_communication_map = [&outgoing_firing_rates, number_local_neurons, this](const auto& synapses) {
        for (const auto neuron_id : NeuronID::range(number_local_neurons)) {
            const auto it = neuron_id.get_neuron_id();

            for (const auto& [target_id, weight] : synapses[it]) {
                const auto& [target_rank, target_neuron_id] = target_id;

                outgoing_firing_rates.append(target_rank, { neuron_id, latest_firing_rate[it] });
            }
        }
    };

    const auto& [outgoing_plastic_synapses, outgoing_static_synapses] = network_graph->get_all_distant_out_edges();
    add_to_communication_map(outgoing_plastic_synapses);
    add_to_communication_map(outgoing_static_synapses);

    const auto& incoming_firing_rates = MPIWrapper::exchange_requests(outgoing_firing_rates);

    for (auto& rank_cache : firing_rate_cache) {
        rank_cache.clear();
    }

    for (const auto& [rank, values] : incoming_firing_rates) {
        auto& cache = firing_rate_cache[rank.get_rank()];
        for (const auto& [neuron_id, firing_rate] : values) {
            cache[neuron_id] = firing_rate;
        }
    }

    last_synced = step;
}
