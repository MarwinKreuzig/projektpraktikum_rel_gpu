#include "SynapticElements.h"

/**
* Updates the number of synaptic elements for neuron "neuron_id"
* Returns the number of synapses to be deleted as a consequence of deleting synaptic elements
*
* Synaptic elements are deleted based on "delta_cnts" in the following way:
* 1. Delete vacant elements
* 2. Delete bound elements
*/

unsigned int SynapticElements::update_number_elements(size_t neuron_id) {
    RelearnException::check(neuron_id < size, "In update number elements: %u is too large! %u", neuron_id, size);

    const double current_count = cnts[neuron_id];
    const auto current_connected_count_integral = connected_cnts[neuron_id];
    const double current_connected_count = static_cast<double>(current_connected_count_integral);
    const double current_vacant = current_count - current_connected_count;
    const double current_delta = delta_cnts[neuron_id];

    RelearnException::check(current_count >= 0.0, "f", current_count);
    RelearnException::check(current_connected_count >= 0.0, "f", current_connected_count);
    RelearnException::check(current_vacant >= 0.0, "f", current_count - current_connected_count);

    // The vacant portion after caring for the delta
    const double new_vacant = current_vacant + current_delta;

    // No deletion of bound synaptic elements required, connected_cnts stays the same
    if (new_vacant >= 0.0) {
        const double new_count = (1 - vacant_retract_ratio) * new_vacant + current_connected_count;
        RelearnException::check(new_count >= current_connected_count, "new count is smaller than connected count");

        cnts[neuron_id] = new_count;
        delta_cnts[neuron_id] = 0.0;
        return 0;
    }

    /**
	* More bound elements should be deleted than are available.
	* Now, neither vacant (see if branch above) nor bound elements are left.
	*/
    if (current_count + current_delta < 0.0) {
        connected_cnts[neuron_id] = 0;
        cnts[neuron_id] = 0.0;
        delta_cnts[neuron_id] = 0.0;

        const auto num_delete_connected = static_cast<unsigned int>(current_connected_count);
        return num_delete_connected;
    }

    const double new_cnts = current_count + current_delta;
    const double new_connected_cnt = floor(new_cnts);
    const auto num_vacant = new_cnts - new_connected_cnt;

    RelearnException::check(num_vacant >= 0, "num vacant is neg");

    connected_cnts[neuron_id] = static_cast<unsigned int>(new_connected_cnt);
    cnts[neuron_id] = new_cnts;
    delta_cnts[neuron_id] = 0.0;

    const double deleted_cnts = current_connected_count - new_connected_cnt;

    RelearnException::check(deleted_cnts >= 0.0, "In synaptic elements, deleted was negative");
    const auto num_delete_connected = static_cast<unsigned int>(deleted_cnts);

    return num_delete_connected;
}
