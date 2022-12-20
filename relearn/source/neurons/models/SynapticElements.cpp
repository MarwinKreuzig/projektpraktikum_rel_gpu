/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "SynapticElements.h"

unsigned int SynapticElements::update_number_elements(const NeuronID& neuron_id) {
    const auto local_neuron_id = neuron_id.get_neuron_id();

    RelearnException::check(local_neuron_id < size, "SynapticElements::update_number_elements: {} is too large! {}", neuron_id, size);

    const auto current_count = grown_elements[local_neuron_id];
    const auto current_connected_count_integral = connected_elements[local_neuron_id];
    const auto current_connected_count = static_cast<double>(current_connected_count_integral);
    const auto current_vacant = current_count - current_connected_count;
    const auto current_delta = deltas_since_last_update[local_neuron_id];

    RelearnException::check(current_count >= 0.0, "SynapticElements::update_number_elements: {}", current_count);
    RelearnException::check(current_connected_count >= 0.0, "SynapticElements::update_number_elements: {}", current_connected_count);
    RelearnException::check(current_vacant >= 0.0, "SynapticElements::update_number_elements: {}", current_count - current_connected_count);

    // The vacant portion after caring for the delta
    // No deletion of bound synaptic elements required, connected_elements stays the same
    if (const auto new_vacant = current_vacant + current_delta; new_vacant >= 0.0) {
        const auto new_count = (1 - vacant_retract_ratio) * new_vacant + current_connected_count;
        RelearnException::check(new_count >= current_connected_count, "SynapticElements::update_number_elements: new count is smaller than connected count");

        grown_elements[local_neuron_id] = new_count;
        deltas_since_last_update[local_neuron_id] = 0.0;
        return 0;
    }

    /**
     * More bound elements should be deleted than are available.
     * Now, neither vacant (see if branch above) nor bound elements are left.
     */
    if (current_count + current_delta < 0.0) {
        connected_elements[local_neuron_id] = 0;
        grown_elements[local_neuron_id] = 0.0;
        deltas_since_last_update[local_neuron_id] = 0.0;

        return current_connected_count_integral;
    }

    const auto new_count = current_count + current_delta;
    const auto new_connected_count = floor(new_count);
    const auto num_vacant = new_count - new_connected_count;

    const auto retracted_new_count = (1 - vacant_retract_ratio) * num_vacant + new_connected_count;

    RelearnException::check(num_vacant >= 0, "SynapticElements::update_number_elements: num vacant is neg");

    connected_elements[local_neuron_id] = static_cast<unsigned int>(new_connected_count);
    grown_elements[local_neuron_id] = retracted_new_count;
    deltas_since_last_update[local_neuron_id] = 0.0;

    const auto deleted_counts = current_connected_count - new_connected_count;

    RelearnException::check(deleted_counts >= 0.0, "SynapticElements::update_number_elements:  deleted was negative");
    const auto num_delete_connected = static_cast<unsigned int>(deleted_counts);

    return num_delete_connected;
}
