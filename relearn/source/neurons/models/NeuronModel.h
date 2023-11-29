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

#include "Types.h"
#include "gpu/utils/CudaHelper.h"
#include "gpu/utils/Interface.h"
#include "mpi/CommunicationMap.h"
#include "enums/FiredStatus.h"
#include "enums/UpdateStatus.h"
#include "neurons/input/BackgroundActivityCalculator.h"
#include "neurons/input/Stimulus.h"
#include "neurons/input/SynapticInputCalculator.h"
#include "neurons/input/BackgroundActivityCalculator.h"
#include "neurons/models/ModelParameter.h"
#include "util/MemoryFootprint.h"
#include "util/RelearnException.h"
#include "util/NeuronID.h"
#include "util/Utility.h"

#include <algorithm>
#include <array>
#include <boost/circular_buffer.hpp>
#include <bitset>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include <range/v3/algorithm/fill.hpp>

template <typename T>
class AdapterNeuronModel;
class AreaMonitor;
class NetworkGraph;
class NeuronMonitor;
class NeuronsExtraInfo;

/**
 * This class provides the basic interface for every neuron model, that is, the rules by which a neuron spikes.
 * The calculations should focus solely on the spiking behavior, and should not account for any plasticity changes.
 * The object itself stores only the local portion of the neuron population.
 * This class performs communication with MPI.
 */
class NeuronModel {
    friend class AreaMonitor;
    template <typename T>
    friend class AdapterNeuronModel;
    friend class NeuronMonitor;

public:
    using number_neurons_type = RelearnTypes::number_neurons_type;
    using step_type = RelearnTypes::step_type;

    /**
     * @brief Constructs a new instance of type NeuronModel with 0 neurons and default values for all parameters
     */
    NeuronModel() = default;

    /**
     * @brief Constructs a new instance of type NeuronModel with 0 neurons.
     * @param h The step size for the numerical integration
     * @param synaptic_input_calculator The object that is responsible for calculating the synaptic input
     * @param background_activity_calculator The object that is responsible for calculating the background activity
     * @param stimulus_calculator The object that is responsible for calculating the stimulus
     */
    NeuronModel(const unsigned int h, std::unique_ptr<SynapticInputCalculator>&& synaptic_input_calculator,
        std::unique_ptr<BackgroundActivityCalculator>&& background_activity_calculator, std::unique_ptr<Stimulus>&& stimulus_calculator)
        : h(h)
        , input_calculator(std::move(synaptic_input_calculator))
        , background_calculator(std::move(background_activity_calculator))
        , stimulus_calculator(std::move(stimulus_calculator)) {
    }

    /**
     * @brief Sets the extra infos. These are used to determine which neuron updates its electrical activity
     * @param new_extra_info The new extra infos, must not be empty
     * @exception Throws a RelearnException if new_extra_info is empty
     */
    void set_extra_infos(std::shared_ptr<NeuronsExtraInfo> new_extra_info) {
        const auto is_filled = new_extra_info.operator bool();
        RelearnException::check(is_filled, "NeuronModel::set_extra_infos: new_extra_info is empty");
        extra_infos = std::move(new_extra_info);

        if (CudaHelper::is_cuda_available()) {
            RelearnException::check(gpu_handle != nullptr, "NeuronModel::set_extra_infos: GPU handle not set");
            gpu_handle->set_extra_infos(extra_infos->get_gpu_handle());
        }

        input_calculator->set_extra_infos(extra_infos);
        background_calculator->set_extra_infos(extra_infos);
        stimulus_calculator->set_extra_infos(extra_infos);
    }

    /**
     * @brief Sets the network graph. It is used to determine which neurons to notify in case of a firing one.
     * @param new_network_graph The new network graph, must not be empty
     * @exception Throws a RelearnException if new_network_graph is empty
     */
    void set_network_graph(std::shared_ptr<NetworkGraph> new_network_graph) {
        const auto is_filled = new_network_graph.operator bool();
        RelearnException::check(is_filled, "SynapticInputCalculator::set_network_graph: new_network_graph is empty");
        network_graph = std::move(new_network_graph);

        input_calculator->set_network_graph(network_graph);
    }

    virtual ~NeuronModel() = default;

    NeuronModel(const NeuronModel& other) = delete;
    NeuronModel& operator=(const NeuronModel& other) = delete;

    NeuronModel(NeuronModel&& other) = default;
    NeuronModel& operator=(NeuronModel&& other) = default;

    enum FireRecorderPeriod {
        NeuronMonitor = 0,
        AreaMonitor = 1,
        Plasticity = 2
    };

    constexpr static size_t number_fire_recorders = 3;

    /**
     * @brief Creates an object of type T wrapped inside an std::unique_ptr
     * @param ...args The arguments that shall be passed to the constructor of T
     * @tparam T The type of NeuronModel that shall be constructed, must inherit from NeuronModel
     * @tparam ...Ts The types of parameters for the constructor of T
     * @return A new instance of type T wrapped inside an std::unique_ptr
     */
    template <typename T, typename... Ts, std::enable_if_t<std::is_base_of<NeuronModel, T>::value, int> = 0>
    [[nodiscard]] static std::unique_ptr<T> create(Ts... args) {
        return std::make_unique<T>(args...);
    }

    /**
     * @brief Provides a way to clone the current NeuronModel, i.e., all parameters.
     *      The returned object shares all parameters, but has 0 neurons.
     *      Because of inheritance-shenanigans, the return value might need to be casted
     * @return A new instance of the class with the same parameters wrapped inside an std::unique_ptr
     */
    [[nodiscard]] virtual std::unique_ptr<NeuronModel> clone() const = 0;

    /**
     * @brief Returns a bool that indicates if the neuron with the passed local id spiked in the current simulation step
     * @param neuron_id The local neuron id that should be queried
     * @exception Throws a RelearnException if neuron_id is too large
     * @return True iff the neuron spiked
     */
    [[nodiscard]] bool get_fired(const NeuronID neuron_id) const {

        const auto local_neuron_id = neuron_id.get_neuron_id();

        RelearnException::check(local_neuron_id < number_local_neurons, "NeuronModels::get_fired: id is too large: {}", neuron_id);
        return get_fired()[local_neuron_id] == FiredStatus::Fired;
    }

    /**
     * @brief Returns a vector of flags that indicate if the neuron with the local id spiked in the current simulation step
     * @return A constant reference to the vector of flags. It is not invalidated by calls to other methods
     */
    [[nodiscard]] std::span<const FiredStatus> get_fired() const noexcept {
        if (CudaHelper::is_cuda_available()) {
            RelearnException::check(number_local_neurons > 0, "NeuronModels::get_fired: number_local_neurons not set");
            RelearnException::check(this->gpu_handle != nullptr, "NeuronModel::set_extra_infos: GPU handle not set");
            return std::span<const FiredStatus>(gpu_handle->get_fired());
        }
        return fired;
    }

    /**
     * @brief Resets the fired recorder to 0 spikes per neuron.
     * @param fire_recorder_period Type of recorder that will be resetted
     */
    void reset_fired_recorder(const FireRecorderPeriod fire_recorder_period) noexcept {
        ranges::fill(fired_recorder[fire_recorder_period], 0U);
    }

    /**
     * @brief Returns a vector of counts how often the neurons have spiked in the last period
     * @param fire_recorder_period Type of recorder period that shall be returned
     * @return A constant reference to the vector of counts. It is not invalidated by calls to other methods
     */
    [[nodiscard]] std::span<const unsigned int> get_fired_recorder(const FireRecorderPeriod fire_recorder_period) const noexcept {
        return fired_recorder[fire_recorder_period];
    }

    /**
     * @brief Returns a double that indicates the neuron's membrane potential in the current simulation step
     * @param neuron_id The local neuron id that should be queried
     * @exception Throws a RelearnException if neuron_id is too large
     * @return The neuron's membrane potential
     */
    [[nodiscard]] double get_x(const NeuronID neuron_id) const {
        const auto local_neuron_id = neuron_id.get_neuron_id();

        RelearnException::check(local_neuron_id < number_local_neurons, "NeuronModels::get_x: id is too large: {}", neuron_id);
        return x[local_neuron_id];
    }

    /**
     * @brief Returns a span of doubles that indicate the neurons' respective membrane potential in the current simulation step
     * @return A span of doubles. It is not invalidated by calls to other methods
     */
    [[nodiscard]] std::span<const double> get_x() const noexcept {
        return x;
    }

    /**
     * @brief Returns a span of doubles that indicate the neurons' respective synaptic input in the current simulation step
     * @return A span of doubles. It is not invalidated by calls to other methods
     */
    [[nodiscard]] std::span<const double> get_synaptic_input() const noexcept {
        return input_calculator->get_synaptic_input();
    }

    /**
     * @brief Returns a span of doubles that indicate the neurons' respective background activity in the current simulation step
     * @return A span of doubles. It is not invalidated by calls to other methods
     */
    [[nodiscard]] std::span<const double> get_background_activity() const noexcept {
        return background_calculator->get_background_activity();
    }

    /**
     * @brief Returns the numerical integration's step size
     * @return The step size
     */
    [[nodiscard]] unsigned int get_h() const noexcept {
        return h;
    }

    /**
     * @brief Returns the number of neurons that are stored in the object
     * @return The number of neurons that are stored in the object
     */
    [[nodiscard]] number_neurons_type get_number_neurons() const noexcept {
        return number_local_neurons;
    }

    /**
     * @brief Returns the secondary variable used for computation of the electrical activity.
     *      The meaning of the variable can vary between classes that inherit from NeuronModels
     * @param neuron_id The local neuron id for the neuron that should be queried
     * @exception Throws a RelearnException if neuron_id is too large
     * @return A double that indicates the secondary variable for the specified neuron
     */
    [[nodiscard]] virtual double get_secondary_variable(const NeuronID neuron_id) const = 0;

    /**
     * @brief Performs one step of simulating the electrical activity for all neurons.
     *      This method performs communication via MPI.
     * @param step The current update step
     */
    void update_electrical_activity(step_type step);

    /**
     * @brief Notifies this class and the input calculators that the plasticity has changed.
     *      Some might cache values, which than can be recalculated
     * @param step The current simulation step
     */
    void notify_of_plasticity_change(step_type step);

    /**
     * @brief Returns a vector with an std::unique_ptr for each class inherited from NeuronModels which can be cloned
     * @return A vector with all inherited classes
     */
    [[nodiscard]] static std::vector<std::unique_ptr<NeuronModel>> get_models();

    /**
     * @brief Returns a vector with all adjustable ModelParameter
     * @return A vector with all adjustable ModelParameter
     */
    [[nodiscard]] virtual std::vector<ModelParameter> get_parameter();

    /**
     * @brief Initializes the model to include number_neurons many local neurons.
     *      Sets the initial membrane potential and initial synaptic inputs to 0.0 and fired to false
     * @param number_neurons The number of local neurons to store in this class
     */
    void init(number_neurons_type number_neurons) {
        if (CudaHelper::is_cuda_available()) {
            init_gpu(number_neurons);
            for (auto& recorder : fired_recorder) {
                recorder.resize(number_neurons, 0U);
            }
            number_local_neurons = number_neurons;
            input_calculator->init(number_neurons);
            background_calculator->init(number_neurons);
            stimulus_calculator->init(number_neurons);
        } else {
            init_cpu(number_neurons);
        }
        init_neurons(0, number_neurons);
    }

    /**
     * @brief Creates new neurons and adds those to the local portion.
     * @param creation_count The number of local neurons that should be added
     */
    void create_neurons(number_neurons_type creation_count) {
        const auto old_size = get_number_neurons();
        const auto new_size = old_size + creation_count;
        if (CudaHelper::is_cuda_available()) {
            create_neurons_gpu(creation_count);
            for (auto& recorder : fired_recorder) {
                recorder.resize(new_size, 0U);
            }

            input_calculator->create_neurons(creation_count);
            background_calculator->create_neurons(creation_count);
            stimulus_calculator->create_neurons(creation_count);
            number_local_neurons = creation_count + old_size;
        } else {
            create_neurons_cpu(creation_count);
        }
        init_neurons(old_size, old_size + creation_count);
    }

    /**
     * @brief Returns the name of the current model
     * @return The name of the current model
     */
    [[nodiscard]] virtual std::string name() = 0;

    /**
     * @brief Performs all required steps to disable all neurons that are specified.
     *      Disables incrementally, i.e., previously disabled neurons are not enabled.
     * @param neuron_ids The local neuron ids that should be disabled
     * @exception Throws a RelearnException if a specified id is too large
     */
    void disable_neurons(const std::span<const NeuronID> neuron_ids) {
        if (CudaHelper::is_cuda_available()) {
            disable_neurons_gpu(neuron_ids);
        } else {
            disable_neurons_cpu(neuron_ids);
        }
    }

    void disable_neurons_gpu(const std::span<const NeuronID> neuron_ids) {
        const auto ids = CudaHelper::convert_neuron_ids_to_primitives(neuron_ids);
        RelearnException::check(gpu_handle != nullptr, "NeuronModel::set_extra_infos: GPU handle not set");
        gpu_handle->disable_neurons(ids);

        for (const auto neuron_id : neuron_ids) {
            const auto local_neuron_id = neuron_id.get_neuron_id();
            for (auto& recorder : fired_recorder) {
                recorder[local_neuron_id] = 0U;
            }
        }
    }

    void disable_neurons_cpu(const std::span<const NeuronID> neuron_ids) {
        for (const auto neuron_id : neuron_ids) {
            const auto local_neuron_id = neuron_id.get_neuron_id();

            RelearnException::check(local_neuron_id < number_local_neurons, "NeuronModels::disable_neurons: There is a too large id: {} vs {}", neuron_id, number_local_neurons);
            fired[local_neuron_id] = FiredStatus::Inactive;
            for (auto& recorder : fired_recorder) {
                recorder[local_neuron_id] = 0U;
            }
        }
    }

    /**
     * @brief Performs all required steps to disable all neurons that are specified.
     *      Disables incrementally, i.e., previously disabled neurons are not enabled.
     * @param neuron_ids The local neuron ids that should be disabled
     * @exception Throws a RelearnException if a specified id is too large
     */
    void enable_neurons(const std::span<const NeuronID> neuron_ids) {
        if (CudaHelper::is_cuda_available()) {
            enable_neurons_gpu(neuron_ids);
        } else {
            enable_neurons_cpu(neuron_ids);
        }
    }

    void enable_neurons_gpu(const std::span<const NeuronID> neuron_ids) {
        const auto ids = CudaHelper::convert_neuron_ids_to_primitives(neuron_ids);
        RelearnException::check(gpu_handle != nullptr, "NeuronModel::set_extra_infos: GPU handle not set");
        gpu_handle->enable_neurons(ids);
    }

    void enable_neurons_cpu(const std::span<const NeuronID> neuron_ids) {
    }

    /**
     * @brief Sets if a neuron fired for the specified neuron. Does not perform bound-checking
     * @param neuron_id The local neuron id
     * @param new_value True iff the neuron fired in the current simulation step
     */
    void set_fired(const NeuronID neuron_id, const FiredStatus new_value) {
        if (CudaHelper::is_cuda_available()) {
            set_fired_gpu(neuron_id, new_value);
        } else {
            set_fired_cpu(neuron_id, new_value);
        }
    }

    void set_fired_cpu(const NeuronID neuron_id, const FiredStatus new_value) {
        const auto local_neuron_id = neuron_id.get_neuron_id();
        fired[local_neuron_id] = new_value;

        extra_infos->set_fired(neuron_id, new_value);

        if (new_value == FiredStatus::Fired) {
            for (auto& recorder : fired_recorder) {
                recorder[local_neuron_id]++;
            }
        }
    }

    void set_fired_gpu(const NeuronID neuron_id, const FiredStatus new_value);

    /**
     * @brief Records the memory footprint of the current object
     * @param footprint Where to store the current footprint
     */
    virtual void record_memory_footprint(const std::unique_ptr<MemoryFootprint>& footprint);

    static constexpr unsigned int default_h{ 10 };
    static constexpr unsigned int min_h{ 1 };
    static constexpr unsigned int max_h{ 1000 };

protected:
    void update_activity(const step_type step) {
        if (CudaHelper::is_cuda_available()) {
            update_activity_gpu(step);
        } else {
            update_activity_cpu();
        }
    }

    virtual void update_activity_benchmark() {
        if (CudaHelper::is_cuda_available()) {
            RelearnException::fail("No utils support");
        } else {
            update_activity_cpu();
        }
    }

    /**
     * @brief Provides a hook to initialize all neurons with local id in [start_id, end_id)
     *      This method exists because of the order of operations when creating neurons
     * @param start_id The first local neuron id to initialize
     * @param end_id The next to last local neuron id to initialize
     */
    void init_neurons(number_neurons_type start_id, number_neurons_type end_id) {
        if (CudaHelper::is_cuda_available()) {
            init_neurons_gpu(start_id, end_id);
        } else {
            init_neurons_cpu(start_id, end_id);
        }
    }

    // CPU
    virtual void update_activity_cpu() = 0;
    virtual void init_neurons_cpu(number_neurons_type start_id, number_neurons_type end_id) = 0;
    virtual void create_neurons_cpu(number_neurons_type creation_count);
    virtual void init_cpu(number_neurons_type number_neurons) = 0;

    // GPU
    void update_activity_gpu(const step_type step) {
        RelearnException::check(gpu_handle != nullptr, "NeuronModel::set_extra_infos: GPU handle not set");

        gpu_handle->update_activity(step, Util::vectorify_span(get_synaptic_input()), Util::vectorify_span(get_stimulus()));
    }

    void init_neurons_gpu(number_neurons_type start_id, number_neurons_type end_id) {
        RelearnException::check(gpu_handle != nullptr, "NeuronModel::set_extra_infos: GPU handle not set");
        gpu_handle->init_neurons(start_id, end_id);
    }

    void create_neurons_gpu(number_neurons_type creation_count) {
        RelearnException::check(gpu_handle != nullptr, "NeuronModel::set_extra_infos: GPU handle not set");
        gpu_handle->create_neurons(creation_count);
    }

    void init_gpu(number_neurons_type number_neurons) {
        RelearnException::check(gpu_handle != nullptr, "NeuronModel::set_extra_infos: GPU handle not set");
        gpu_handle->init_neuron_model(number_neurons);
    }

    /**
     * @brief Sets the membrane potential for the specified neuron. Does not perform bound-checking
     * @param neuron_id The local neuron id
     * @param new_value The new membrane potential
     */
    void set_x(const NeuronID neuron_id, const double new_value) {
        const auto local_neuron_id = neuron_id.get_neuron_id();
        x[local_neuron_id] = new_value;
    }

    [[nodiscard]] double get_synaptic_input(const NeuronID neuron_id) const {
        return input_calculator->get_synaptic_input(neuron_id);
    }

    [[nodiscard]] double get_background_activity(const NeuronID neuron_id) const {
        return background_calculator->get_background_activity(neuron_id);
    }

    [[nodiscard]] double get_stimulus(const NeuronID neuron_id) const {
        return stimulus_calculator->get_stimulus(neuron_id);
    }

    [[nodiscard]] const std::span<const double> get_stimulus() const {
        return stimulus_calculator->get_stimulus();
    }

    [[nodiscard]] const std::unique_ptr<SynapticInputCalculator>& get_synaptic_input_calculator() const noexcept {
        return input_calculator;
    }

    [[nodiscard]] const std::unique_ptr<BackgroundActivityCalculator>& get_background_activity_calculator() const noexcept {
        return background_calculator;
    }

    [[nodiscard]] const std::unique_ptr<Stimulus>& get_stimulus_calculator() const noexcept {
        return stimulus_calculator;
    }

    [[nodiscard]] const std::shared_ptr<NeuronsExtraInfo>& get_extra_infos() const noexcept {
        return extra_infos;
    }

    std::shared_ptr<gpu::models::NeuronModelHandle> gpu_handle{};

private:
    // My local number of neurons
    number_neurons_type number_local_neurons{ 0 };

    // Model parameters for all neurons
    unsigned int h{ default_h }; // Precision for Euler integration

    // Variables for each neuron where the array index denotes the local neuron ID
    std::vector<double> x{}; // The membrane potential (in equations usually v(t))
    std::array<std::vector<unsigned int>, number_fire_recorders> fired_recorder{}; // How often the neurons have spiked
    std::vector<FiredStatus> fired{}; // If the neuron fired in the current update step

    std::unique_ptr<SynapticInputCalculator> input_calculator{};
    std::unique_ptr<BackgroundActivityCalculator> background_calculator{};
    std::unique_ptr<Stimulus> stimulus_calculator{};

    std::shared_ptr<NeuronsExtraInfo> extra_infos{};
    std::shared_ptr<NetworkGraph> network_graph{};
};
