#pragma once

__global__ void cuda_izhekevich_kernel() {
    const auto number_local_neurons = get_number_neurons();
    const auto disable_flags = get_extra_infos()->get_disable_flags();

    const auto h = get_h();
    const auto scale = 1.0 / h;

#pragma omp parallel for shared(disable_flags, number_local_neurons, h, scale) default(none)
    for (NeuronID::value_type neuron_id = 0U; neuron_id < number_local_neurons; ++neuron_id) {
        if (disable_flags[neuron_id] == UpdateStatus::Disabled) {
            continue;
        }

        NeuronID converted_id{ neuron_id };

        const auto synaptic_input = get_synaptic_input(converted_id);
        const auto background = get_background_activity(converted_id);
        const auto stimulus = get_stimulus(converted_id);
        const auto input = synaptic_input + background + stimulus;

        auto x_val = get_x(converted_id);
        auto u_val = u[neuron_id];

        auto has_spiked = FiredStatus::Inactive;

        for (unsigned int integration_steps = 0; integration_steps < h; ++integration_steps) {
            const auto x_increase = k1 * x_val * x_val + k2 * x_val + k3 - u_val + input;
            const auto u_increase = a * (b * x_val - u_val);

            x_val += x_increase * scale;
            u_val += u_increase * scale;

            const auto spiked = x_val >= V_spike;

            if (spiked) {
                x_val = c;
                u_val += d;
                has_spiked = FiredStatus::Fired;
                break;
            }
        }

        set_fired(converted_id, has_spiked);
        set_x(converted_id, x_val);
        u[neuron_id] = u_val;
    }
}

void cuda_update_izhekevich_activity() {

}