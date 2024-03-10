#include "PoissonModel.cuh"

#include "../../utils/Interface.h"

#include "../NeuronsExtraInfos.cuh"
#include "../../Commons.cuh"
#include "NeuronModel.cuh"
#include "../../utils/Random.cuh"

#include "../../../shared/calculations/NeuronModelCalculations.h"

#include "../../../shared/enums/FiredStatus.h"

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

namespace gpu::models {
    __device__ PoissonModel::PoissonModel(const unsigned int _h, gpu::background::BackgroundActivity* bgc, const double _x_0,
                            const double _tau_x,
                            const unsigned int _refractory_period)
            : NeuronModel(_h, bgc)
            , x_0(_x_0)
            , tau_x(_tau_x)
            , refractory_period(_refractory_period) {
    }

    __device__ void PoissonModel::init(const RelearnGPUTypes::number_neurons_type number_neurons) {
        NeuronModel::init(number_neurons);

        refractory_time.resize(number_neurons);
    }

    __device__ void PoissonModel::create_neurons(RelearnGPUTypes::number_neurons_type creation_count) {
        NeuronModel::create_neurons(creation_count);
        const auto new_size = extra_infos->get_number_local_neurons();
        refractory_time.resize(new_size);
    }

    __device__ void PoissonModel::update_activity(RelearnGPUTypes::step_type step) {

        const auto tau_x_inverse = 1.0 / tau_x;

        const auto neuron_id = block_thread_to_neuron_id(blockIdx.x, threadIdx.x, blockDim.x);

        if (neuron_id >= extra_infos->get_number_local_neurons()) {
            return;
        }

        // Init
        auto curand_state = gpu::RandomHolder::init(step, extra_infos->get_number_local_neurons(), gpu::RandomHolder::POISSON, neuron_id);
        const auto random_value = gpu::RandomHolder::get_percentage(&curand_state);

        if (extra_infos->disable_flags[neuron_id] == UpdateStatus::Disabled) {
            return;
        }
        const auto synaptic_input = get_synaptic_input(step, neuron_id);
        const auto background_activity = get_background_activity(step, neuron_id);
        const auto stimulus = get_stimulus(step, neuron_id);

        const auto x_ = x[neuron_id];

        const auto& [x_val, this_fired, this_refractory_time] = Calculations::poisson(x_, synaptic_input, background_activity, stimulus, refractory_time[neuron_id], random_value, x_0, refractory_period, gpu::models::NeuronModel::h, gpu::models::NeuronModel::scale, tau_x_inverse);

        refractory_time[neuron_id] = this_refractory_time;
        set_x(neuron_id, x_val);
        set_fired(neuron_id, this_fired);
    }

    namespace poisson {

        std::shared_ptr<NeuronModelHandle> construct_gpu(std::shared_ptr<gpu::background::BackgroundHandle> background_handle, const unsigned int _h, double x_0, double _tau_x, const unsigned int _refractory_period) {
            return gpu::models::construct<gpu::models::PoissonModel>(background_handle, _h, x_0, _tau_x, _refractory_period);
        }
    };
};