#pragma once

#include "neurons/enums/FiredStatus.h"

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

    GPU_AND_HOST inline std::tuple<double, FiredStatus, double> izhikevich(double x_val, double synaptic_input, double background, double stimulus, double u_val, double h, double scale, double V_spike, double a, double b, double c, double d, double k1, double k2, double k3) {
        const auto input = synaptic_input + background + stimulus;

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

        return std::make_tuple(x_val, has_spiked, u_val);
    }

    GPU_AND_HOST inline std::tuple<double, FiredStatus, double> aeif(double x_val, double synaptic_input, double background, double stimulus, double w_val, double h, double scale, double V_spike,double g_L, double E_L, double V_T, double d_T, double d_T_inverse, double a, double b, double C_inverse, double tau_w_inverse) {
        auto has_spiked = FiredStatus::Inactive;
        const auto input = synaptic_input + background + stimulus;

        for (unsigned int integration_steps = 0; integration_steps < h; ++integration_steps) {
            const auto linear_part = -g_L * (x_val - E_L);
            const auto exp_part = g_L * d_T * std::exp((x_val - V_T) * d_T_inverse);
            const auto x_increase = (linear_part + exp_part - w_val + input) * C_inverse;
            const auto w_increase = (a * (x_val - E_L) - w_val) * tau_w_inverse;

            x_val += x_increase * scale;
            w_val += w_increase * scale;

            if (x_val >= V_spike) {
                x_val = E_L;
                w_val += b;
                has_spiked = FiredStatus::Fired;
                break;
            }
        }
        return std::make_tuple(x_val, has_spiked, w_val);
    }

    GPU_AND_HOST inline std::tuple<double, FiredStatus, double> fitz_hugh_nagumo(double x_val, double synaptic_input, double background, double stimulus, double w_val, double h, double scale, double phi, double a, double b) {
        const auto input = synaptic_input + background + stimulus;

        for (unsigned int integration_steps = 0; integration_steps < h; ++integration_steps) {
            const auto x_increase = x_val - x_val * x_val * x_val * (1.0 / 3.0) - w_val + input;
            const auto w_increase = phi * (x_val + a - b * w_val);

            x_val += x_increase * scale;
            w_val += w_increase * scale;
        }

        const auto spiked = w_val > x_val - x_val * x_val * x_val * (1.0 / 3.0) && x_val > 1.0;

        return std::make_tuple(x_val, spiked ? FiredStatus::Fired : FiredStatus::Inactive, w_val);
    }

};