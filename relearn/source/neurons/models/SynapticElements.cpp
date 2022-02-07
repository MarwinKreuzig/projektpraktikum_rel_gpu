#include "SynapticElements.h"

/**
 * Updates the number of synaptic elements for neuron "neuron_id"
 * Returns the number of synapses to be deleted as a consequence of deleting synaptic elements
 *
 * Synaptic elements are deleted based on "deltas_since_last_update" in the following way:
 * 1. Delete vacant elements
 * 2. Delete bound elements
 */

unsigned int SynapticElements::update_number_elements(const NeuronID& neuron_id) {
    RelearnException::check(neuron_id.id() < size, "SynapticElements::update_number_elements: {} is too large! {}", neuron_id, size);

    const auto current_count = grown_elements[neuron_id.id()];
    const auto current_connected_count_integral = connected_elements[neuron_id.id()];
    const auto current_connected_count = static_cast<double>(current_connected_count_integral);
    const auto current_vacant = current_count - current_connected_count;
    const auto current_delta = deltas_since_last_update[neuron_id.id()];

    RelearnException::check(current_count >= 0.0, "SynapticElements::update_number_elements: {}", current_count);
    RelearnException::check(current_connected_count >= 0.0, "SynapticElements::update_number_elements: {}", current_connected_count);
    RelearnException::check(current_vacant >= 0.0, "SynapticElements::update_number_elements: {}", current_count - current_connected_count);

    // The vacant portion after caring for the delta
    // No deletion of bound synaptic elements required, connected_elements stays the same
    if (const auto new_vacant = current_vacant + current_delta; new_vacant >= 0.0) {
        const auto new_count = (1 - vacant_retract_ratio) * new_vacant + current_connected_count;
        RelearnException::check(new_count >= current_connected_count, "SynapticElements::update_number_elements: new count is smaller than connected count");

        grown_elements[neuron_id.id()] = new_count;
        deltas_since_last_update[neuron_id.id()] = 0.0;
        return 0;
    }

    /**
     * More bound elements should be deleted than are available.
     * Now, neither vacant (see if branch above) nor bound elements are left.
     */
    if (current_count + current_delta < 0.0) {
        connected_elements[neuron_id.id()] = 0;
        grown_elements[neuron_id.id()] = 0.0;
        deltas_since_last_update[neuron_id.id()] = 0.0;

        return current_connected_count_integral;
    }

    const auto new_count = current_count + current_delta;
    const auto new_connected_count = floor(new_count);
    const auto num_vacant = new_count - new_connected_count;

    const auto retracted_new_count = (1 - vacant_retract_ratio) * num_vacant + new_connected_count;

    RelearnException::check(num_vacant >= 0, "SynapticElements::update_number_elements: num vacant is neg");

    connected_elements[neuron_id.id()] = static_cast<unsigned int>(new_connected_count);
    grown_elements[neuron_id.id()] = retracted_new_count;
    deltas_since_last_update[neuron_id.id()] = 0.0;

    const auto deleted_cnts = current_connected_count - new_connected_count;

    RelearnException::check(deleted_cnts >= 0.0, "SynapticElements::update_number_elements:  deleted was negative");
    const auto num_delete_connected = static_cast<unsigned int>(deleted_cnts);

    return num_delete_connected;
}
