#include "NeuronModel.cuh"
#include "../../Commons.cuh"
#include <numeric>

namespace gpu::models {
    __device__ NeuronModel::NeuronModel(const unsigned int _h, void* gpu_background_calculator) {
        h = _h;
        const auto _scale = 1.0 / _h;
        scale = _scale;
        background_calculator = (gpu::background::BackgroundActivity*)gpu_background_calculator;
    }

    /**
     * @brief Initializes the model to include number_neurons many local neurons.
     *      Sets the initial membrane potential and initial synaptic inputs to 0.0 and fired to false
     * @param number_neurons The number of local neurons to store in this class
     */
    __device__ void NeuronModel::init(const RelearnGPUTypes::number_neurons_type number_neurons) {
        x.resize(number_neurons);
    }

    /**
     * @brief Creates new neurons and adds those to the local portion.
     * @param creation_count The number of local neurons that should be added
     */
    __device__ void NeuronModel::create_neurons(RelearnGPUTypes::number_neurons_type creation_count) {
    }

    /**
     * @brief Provides a hook to initialize all neurons with local id in [start_id, end_id)
     *      This method exists because of the order of operations when creating neurons
     * @param start_id The first local neuron id to initialize
     * @param end_id The next to last local neuron id to initialize
     */
    __device__ void NeuronModel::init_neurons(const RelearnGPUTypes::number_neurons_type start_id, const RelearnGPUTypes::number_neurons_type end_id) { }

    /**
     * Sets the NeuronsExtraInfos
     * @param _extra_infos Pointer to the NeuronsExtraInfos instance on the gpu
     */
    __device__ void NeuronModel::set_extra_infos(neurons::NeuronsExtraInfos* _extra_infos) {
        extra_infos = _extra_infos;
    }

/**
 * The kernel to update the activity of the neuron model
 * @param Pointer to the neuron model on the gpu
 * @param step The current step
 */
    __global__ void update_activity_kernel(NeuronModel* neuron_model, RelearnGPUTypes::step_type step) {
        neuron_model->update_activity(step);
    }

    NeuronModelHandleImpl::NeuronModelHandleImpl(void* _dev_ptr)
                : device_ptr(_dev_ptr) {
            _init();
        }

    void* NeuronModelHandleImpl::get_device_pointer() const {
            return device_ptr;
        }

    void NeuronModelHandleImpl::_init() {
            void* stimulus_ptr = (void*)execute_and_copy<void*>([=] __device__(void* neuron_model) { return &((NeuronModel*)neuron_model)->stimulus; }, device_ptr);
            void* syn_input_ptr = (void*)execute_and_copy<void*>([=] __device__(void* neuron_model) { return &((NeuronModel*)neuron_model)->syn_input; }, device_ptr);

            handle_stimulation = gpu::Vector::CudaArrayDeviceHandle<double>(stimulus_ptr);
            handle_syn_input = gpu::Vector::CudaArrayDeviceHandle<double>(syn_input_ptr);
        }

    void NeuronModelHandleImpl::set_extra_infos(const std::unique_ptr<gpu::neurons::NeuronsExtraInfosHandle>& extra_infos_handle) {
            cuda_generic_kernel<<<1, 1>>>([=] __device__(NeuronModel * neuron_model, gpu::neurons::NeuronsExtraInfos * extra_infos) { neuron_model->set_extra_infos(extra_infos); }, (NeuronModel*)device_ptr,
                                          (gpu::neurons::NeuronsExtraInfos*)(static_cast<neurons::NeuronsExtraInfosHandleImpl*>(extra_infos_handle.get())->get_device_pointer()));
        }

    void NeuronModelHandleImpl::init_neuron_model(const RelearnGPUTypes::number_neurons_type number_neurons) {
            gpu_check_last_error();
            cudaDeviceSynchronize();
            cuda_generic_kernel<<<1, 1>>>([=] __device__(NeuronModel * neuron_model, size_t number_neurons) {
                                              neuron_model->init(number_neurons);
                                          },
                                          (NeuronModel*)device_ptr, number_neurons);
            gpu_check_last_error();
            cudaDeviceSynchronize();
            gpu_check_last_error();

            void* fired_ptr = execute_and_copy<void*>([=] __device__(NeuronModel * neuron_model) -> void* { return (void*)&(neuron_model->fired); }, (NeuronModel*)device_ptr);
            handle_fired = gpu::Vector::CudaArrayDeviceHandle<FiredStatus>(fired_ptr);
            handle_fired.resize(number_neurons);
        }

    void NeuronModelHandleImpl::init_neurons(const RelearnGPUTypes::number_neurons_type start_id, const RelearnGPUTypes::number_neurons_type end_id) {
            cuda_generic_kernel<<<1, 1>>>([] __device__(NeuronModel * neuron_model, size_t start_id, size_t end_id) { neuron_model->init_neurons(start_id, end_id); }, (NeuronModel*)device_ptr, start_id, end_id);
            gpu_check_last_error();
            cudaDeviceSynchronize();
        }

    void NeuronModelHandleImpl::create_neurons(RelearnGPUTypes::neuron_id_type creation_count) {
            cuda_generic_kernel<<<1, 1>>>([] __device__(NeuronModel * neuron_model, size_t number_neurons) { neuron_model->create_neurons(number_neurons); }, (NeuronModel*)device_ptr, creation_count);
            gpu_check_last_error();
            cudaDeviceSynchronize();
        }

    void NeuronModelHandleImpl::disable_neurons(const std::vector<RelearnGPUTypes::neuron_id_type>& neuron_ids) {
            }

    void NeuronModelHandleImpl::enable_neurons(const std::vector<RelearnGPUTypes::neuron_id_type>& neuron_ids) {
            }

    std::vector<FiredStatus> NeuronModelHandleImpl::get_fired() {
                handle_fired.copy_to_host(vec_f);

                /*size_t fired = 0;
                for(const auto e:vec_f) {
                    if(e==FiredStatus::Fired) {
                        fired++;
                    }
                }

                std::cout << "Fired " << fired << "\n";*/

                return vec_f;
            }

    void NeuronModelHandleImpl::update_activity(RelearnGPUTypes::step_type step, const std::vector<double>& syn_input, const std::vector<double>& stimulation) {
                const auto number_local_neurons = syn_input.size();
                handle_stimulation.copy_to_device(stimulation.data(), number_local_neurons);
                handle_syn_input.copy_to_device(syn_input.data(), number_local_neurons);

                const auto num_threads = get_number_threads(update_activity_kernel, number_local_neurons);
                const auto num_blocks = get_number_blocks(num_threads, number_local_neurons);

                update_activity_kernel<<<num_blocks, num_threads>>>((NeuronModel*)device_ptr, step);
            }

};