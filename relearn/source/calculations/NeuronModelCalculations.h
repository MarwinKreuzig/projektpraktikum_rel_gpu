#pragma once

#include "neurons/enums/FiredStatus.h"
#include "gpu/CudaHelper.h"

#include <tuple>

namespace Calculations {

    GPU_AND_HOST inline std::tuple<double,FiredStatus,unsigned int> poisson(double x_val, double synaptic_input, double background, double stimulus, unsigned int refractory_time, double random_value, double x_0, double refractory_period, unsigned int h,  double scale, double tau_x_inverse) {

        const auto input = synaptic_input + background + stimulus;


        for (unsigned int integration_steps = 0; integration_steps < h; integration_steps++) {
            x_val += ((x_0 - x_val) * tau_x_inverse + input) * scale;
        }

        FiredStatus fired;

        if (refractory_time == 0) {
            const auto threshold = random_value;
            const auto f = x_val >= threshold;
            if (f) {
                fired = FiredStatus::Fired;
                refractory_time = refractory_period;
            } else {
                fired =  FiredStatus::Inactive;
            }
        } else {
            fired = FiredStatus::Inactive;
            --refractory_time;
        }

        return std::make_tuple(x_val, fired, refractory_time);
    }

};