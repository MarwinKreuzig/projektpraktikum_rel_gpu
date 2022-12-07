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

#include "gtest/gtest.h"

#include "Config.h"
#include "Types.h"
#include "algorithm/BarnesHutInternal/BarnesHutCell.h"
#include "algorithm/BarnesHutInternal/BarnesHutInvertedCell.h"
#include "algorithm/FMMInternal/FastMultipoleMethods.h"
#include "algorithm/FMMInternal/FastMultipoleMethodsBase.h"
#include "algorithm/FMMInternal/FastMultipoleMethodsCell.h"
#include "algorithm/Kernel/Gamma.h"
#include "algorithm/Kernel/Gaussian.h"
#include "algorithm/Kernel/Kernel.h"
#include "algorithm/Kernel/Linear.h"
#include "algorithm/Kernel/Weibull.h"
#include "io/LogFiles.h"
#include "mpi/CommunicationMap.h"
#include "mpi/MPIWrapper.h"
#include "neurons/CalciumCalculator.h"
#include "neurons/ElementType.h"
#include "neurons/FiredStatus.h"
#include "neurons/NetworkGraph.h"
#include "neurons/SignalType.h"
#include "neurons/helper/SynapseCreationRequests.h"
#include "neurons/models/SynapticElements.h"
#include "structure/Cell.h"
#include "structure/Octree.h"
#include "structure/OctreeNode.h"
#include "structure/Partition.h"
#include "util/Interval.h"
#include "util/MemoryHolder.h"
#include "util/RelearnException.h"
#include "util/StepParser.h"
#include "util/TaggedID.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <map>
#include <random>
#include <stack>
#include <tuple>
#include <vector>

class NeuronModel;
class NeuronsExtraInfo;

inline bool initialized = false;

/**
 * @brief Get the path to relearn/relearn
 *
 * @return std::filesystem::path path
 */
[[nodiscard]] std::filesystem::path get_relearn_path();

class RelearnTest : public ::testing::Test {
protected:
    static void init() {

        static bool template_initialized = false;

        if (template_initialized) {
            return;
        }

        if (!initialized) {
            initialized = true;

            char* argument = (char*)"./runTests";
            MPIWrapper::init(1, &argument);
        }
        template_initialized = true;
    }

protected:
    static void SetUpTestCaseTemplate() {
        RelearnException::hide_messages = true;
        LogFiles::disable = true;

        init();
    }

    static void SetUpTestSuite();

    static void TearDownTestSuite() {
        RelearnException::hide_messages = false;
        LogFiles::disable = false;
    }

    void SetUp() override {
        if (use_predetermined_seed) {
            std::cerr << "Using predetermined seed: " << predetermined_seed << '\n';
            mt.seed(predetermined_seed);
        } else {
            const auto now = std::chrono::high_resolution_clock::now();
            const auto time_since_epoch = now.time_since_epoch();
            const auto time_since_epoch_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(time_since_epoch).count();

            const auto seed = static_cast<unsigned int>(time_since_epoch_ns);

            std::cerr << "Test seed: " << seed << '\n';
            mt.seed(seed);
        }
        // Remove tmp files
        for (auto const& entry : std::filesystem::recursive_directory_iterator("./")) {
            if (std::filesystem::is_regular_file(entry) && entry.path().extension() == ".tmp") {
                std::filesystem::remove(entry);
                std::cerr << "REMOVED " << entry.path() << std::endl;
            }
        }
    }

    void TearDown() override {
        std::cerr << "Test finished\n";
    }

    template <typename AdditionalCellAttributes>
    void make_mpi_mem_available() {
        MemoryHolder<AdditionalCellAttributes>::make_all_available();
    }

    size_t round_to_next_exponent(size_t numToRound, size_t exponent) {
        auto log = std::log(static_cast<double>(numToRound)) / std::log(static_cast<double>(exponent));
        auto rounded_exp = std::ceil(log);
        auto new_val = std::pow(static_cast<double>(exponent), rounded_exp);
        return static_cast<size_t>(new_val);
    }

    double get_random_double(double min, double max) {
        uniform_real_distribution<double> urd(min, max);
        return urd(mt);
    }

    template <typename T>
    T get_random_integer(T min, T max) {
        uniform_int_distribution<T> uid(min, max);
        return uid(mt);
    }

    std::vector<RelearnTypes::area_name> get_random_area_names(size_t max_areas) {
        const auto nr_areas = get_random_integer<size_t>(1, max_areas);
        return get_random_area_names_specific(nr_areas);
    }

    std::vector<RelearnTypes::area_name> get_random_area_names_specific(size_t nr_areas) {
        std::vector<RelearnTypes::area_name> area_names{};
        for (int area_id = 0; area_id < nr_areas; area_id++) {
            RelearnTypes::area_name name;
            do {
                name = std::to_string(get_random_percentage());
            } while (name.empty() || std::find(area_names.begin(), area_names.end(), name) != area_names.end());

            area_names.emplace_back(name);
        }
        return area_names;
    }

    std::vector<RelearnTypes::area_id> get_random_area_ids(size_t num_areas, size_t num_neurons) {
        std::vector<RelearnTypes::area_id> area_ids{};
        for (const auto& neuron_id : NeuronID::range(num_neurons)) {
            area_ids.emplace_back(get_random_integer(static_cast<size_t>(0), num_areas - 1));
        }
        return area_ids;
    }

    static std::vector<RelearnTypes::area_name> get_neuron_id_vs_area_name(const std::vector<RelearnTypes::area_id>& neuron_id_vs_area_id, const std::vector<RelearnTypes::area_name>& area_id_vs_area_name) {
        std::vector<RelearnTypes::area_name> neuron_id_vs_area_name{};

        for (auto i : neuron_id_vs_area_id) {
            neuron_id_vs_area_name.emplace_back(area_id_vs_area_name[i]);
        }
        return neuron_id_vs_area_name;
    }

    std::tuple<Vec3d, Vec3d> get_random_simulation_box_size() {
        const auto rand_x_1 = get_random_double(-position_bounary, +position_bounary);
        const auto rand_x_2 = get_random_double(-position_bounary, +position_bounary);

        const auto rand_y_1 = get_random_double(-position_bounary, +position_bounary);
        const auto rand_y_2 = get_random_double(-position_bounary, +position_bounary);

        const auto rand_z_1 = get_random_double(-position_bounary, +position_bounary);
        const auto rand_z_2 = get_random_double(-position_bounary, +position_bounary);

        return {
            { std::min(rand_x_1, rand_x_2), std::min(rand_y_1, rand_y_2), std::min(rand_z_1, rand_z_2) },
            { std::max(rand_x_1, rand_x_2), std::max(rand_y_1, rand_y_2), std::max(rand_z_1, rand_z_2) }
        };
    }

    double get_random_position_element() {
        const auto val = get_random_double(-position_bounary, +position_bounary);
        return val;
    }

    Vec3d get_random_position() {
        const auto x = get_random_double(-position_bounary, +position_bounary);
        const auto y = get_random_double(-position_bounary, +position_bounary);
        const auto z = get_random_double(-position_bounary, +position_bounary);

        return { x, y, z };
    }

    Vec3d get_minimum_position() {
        return { -position_bounary, -position_bounary, -position_bounary };
    }

    Vec3d get_maximum_position() {
        return { position_bounary, position_bounary, position_bounary };
    }

    Vec3d get_random_position_in_box(const Vec3d& min, const Vec3d& max) {
        const auto x = get_random_double(min.get_x(), max.get_x());
        const auto y = get_random_double(min.get_y(), max.get_y());
        const auto z = get_random_double(min.get_z(), max.get_z());

        return { x, y, z };
    }

    size_t get_random_number_ranks() {
        return uid_num_ranks(mt);
    }

    int get_random_rank(size_t number_ranks) {
        uniform_int_distribution<int> uid(0, static_cast<int>(number_ranks) - 1);
        return uid(mt);
    }

    int get_random_rank(size_t number_ranks, int exclude_rank) {
        uniform_int_distribution<int> uid(0, static_cast<int>(number_ranks) - 1);

        auto rank = uid(mt);
        while (rank == exclude_rank) {
            rank = uid(mt);
        }

        return rank;
    }

    size_t get_adjusted_random_number_ranks() {
        const auto random_rank = get_random_number_ranks();
        return round_to_next_exponent(random_rank, 2);
    }

    size_t get_random_number_neurons() {
        return uid_num_neurons(mt);
    }

    size_t get_random_number_synapses() {
        return uid_num_synapses(mt);
    }

    NeuronID get_random_neuron_id(size_t number_neurons, size_t offset = 0) {
        uniform_int_distribution<size_t> uid(offset, offset + number_neurons - 1);
        return NeuronID{ uid(mt) };
    }

    NeuronID get_random_neuron_id(size_t number_neurons, NeuronID except) {
        NeuronID nid;
        do {
            nid = get_random_neuron_id(number_neurons);
        } while (nid == except);
        return nid;
    }

    int get_random_synapse_weight() {
        int weight = uid_synapse_weight(mt);

        while (weight == 0) {
            weight = uid_synapse_weight(mt);
        }

        return weight;
    }

    unsigned int get_random_synaptic_element_connected_count(unsigned int maximum) {
        return get_random_integer<unsigned int>(0, maximum);
    }

    std::vector<std::tuple<NeuronID, NeuronID, int>> get_random_synapses(size_t number_neurons, size_t number_synapses) {
        std::vector<std::tuple<NeuronID, NeuronID, int>> synapses(number_synapses);

        for (auto i = 0; i < number_synapses; i++) {
            const auto source_id = get_random_neuron_id(number_neurons);
            const auto target_id = get_random_neuron_id(number_neurons);
            const auto weight = get_random_synapse_weight();

            synapses[i] = { source_id, target_id, weight };
        }

        return synapses;
    }

    double get_random_percentage() {
        return get_random_double(0.0, std::nextafter(1.0, 2.0));
    }

    double get_random_synaptic_element_count() {
        return get_random_double(min_grown_elements, std::nextafter(max_grown_elements, max_grown_elements * 2.0));
    }

    uint8_t get_random_refinement_level() noexcept {
        return static_cast<uint8_t>(uid_refinement(mt));
    }

    uint8_t get_small_refinement_level() noexcept {
        return static_cast<uint8_t>(uid_small_refinement(mt));
    }

    uint8_t get_large_refinement_level() noexcept {
        return static_cast<uint8_t>(uid_large_refinement(mt));
    }

    bool get_random_bool() noexcept {
        uniform_int_distribution<unsigned short> uid_bool(0, 1);
        return uid_bool(mt) == 0;
    }

    ElementType get_random_element_type() noexcept {
        return get_random_bool() ? ElementType::Axon : ElementType::Dendrite;
    }

    SignalType get_random_signal_type() noexcept {
        return get_random_bool() ? SignalType::Excitatory : SignalType::Inhibitory;
    }

    double get_random_gamma_k() noexcept {
        return get_random_double(0.001, 10.0);
    }

    double get_random_gamma_theta() noexcept {
        return get_random_double(0.001, 100.0);
    }

    double get_random_gaussian_mu() noexcept {
        return get_random_double(-10000.0, 10000.0);
    }

    double get_random_gaussian_sigma() noexcept {
        return get_random_double(0.001, 10000.0);
    }

    double get_random_linear_cutoff() noexcept {
        return get_random_double(0.001, 1000.0);
    }

    double get_random_weibull_k() noexcept {
        return get_random_double(0.001, 10.0);
    }

    double get_random_weibull_b() noexcept {
        return get_random_double(0.001, 10000.0);
    }

    KernelType get_random_kernel_type() noexcept {
        const auto choice = get_random_integer<int>(0, 3);

        switch (choice) {
        case 0:
            return KernelType::Gamma;
        case 1:
            return KernelType::Gaussian;
        case 2:
            return KernelType::Linear;
        case 3:
            return KernelType::Weibull;
        }

        return KernelType::Gamma;
    }

    template <typename AdditionalCellAttributes>
    std::string set_random_kernel() {
        const auto kernel_choice = get_random_kernel_type();

        Kernel<AdditionalCellAttributes>::set_kernel_type(kernel_choice);

        std::stringstream ss{};

        ss << kernel_choice;

        if (kernel_choice == KernelType::Gamma) {
            const auto k = get_random_gamma_k();
            const auto theta = get_random_gamma_theta();

            ss << '\t' << k << '\t' << theta;

            GammaDistributionKernel::set_k(k);
            GammaDistributionKernel::set_theta(theta);
        }

        if (kernel_choice == KernelType::Gaussian) {
            const auto sigma = get_random_gaussian_sigma();
            const auto mu = get_random_gaussian_mu();

            ss << '\t' << sigma << '\t' << mu;

            GaussianDistributionKernel::set_sigma(sigma);
            GaussianDistributionKernel::set_mu(mu);
        }

        if (kernel_choice == KernelType::Linear) {
            const auto cutoff = get_random_linear_cutoff();

            ss << '\t' << cutoff;

            LinearDistributionKernel::set_cutoff(cutoff);
        }

        if (kernel_choice == KernelType::Weibull) {
            const auto k = get_random_weibull_k();
            const auto b = get_random_weibull_b();

            ss << '\t' << k << '\t' << b;

            WeibullDistributionKernel::set_k(k);
            WeibullDistributionKernel::set_b(b);
        }

        return ss.str();
    }

    std::vector<LocalSynapse> generate_local_synapses(size_t number_neurons) {
        const auto number_synapses = get_random_number_synapses();

        std::map<std::pair<NeuronID, NeuronID>, int> synapse_map{};
        for (auto i = 0; i < number_synapses; i++) {
            const auto source = get_random_neuron_id(number_neurons);
            const auto target = get_random_neuron_id(number_neurons);
            const auto weight = get_random_synapse_weight();

            synapse_map[{ target, source }] += weight;
        }

        std::vector<LocalSynapse> synapses{};
        synapses.reserve(synapse_map.size());

        for (const auto& [pair, weight] : synapse_map) {
            const auto& [target, source] = pair;
            if (weight != 0) {
                synapses.emplace_back(target, source, weight);
            }
        }

        return synapses;
    }

    std::tuple<CommunicationMap<SynapseCreationRequest>, std::vector<size_t>, std::vector<size_t>> create_incoming_requests(size_t number_ranks, int current_rank,
        size_t number_neurons, size_t number_requests_lower_bound, size_t number_requests_upper_bound) {

        CommunicationMap<SynapseCreationRequest> cm(static_cast<int>(number_ranks));
        std::vector<size_t> number_excitatory_requests(number_neurons, 0);
        std::vector<size_t> number_inhibitory_requests(number_neurons, 0);

        for (const auto& target_id : NeuronID::range(number_neurons)) {
            const auto number_requests = get_random_integer<size_t>(number_requests_lower_bound, number_requests_upper_bound);

            const auto id = target_id.get_neuron_id();

            for (auto r = 0; r < number_requests; r++) {
                const auto source_rank = get_random_rank(number_ranks);
                const auto source_id = get_random_neuron_id(number_neurons);
                const auto fixed_source_id = (source_id.get_neuron_id() == target_id.get_neuron_id() && current_rank == source_rank) ? NeuronID{ (source_id.get_neuron_id() + 1) % number_neurons } : source_id;

                const auto signal_type = get_random_signal_type();

                const SynapseCreationRequest scr{ target_id, fixed_source_id, signal_type };

                if (signal_type == SignalType::Excitatory) {
                    number_excitatory_requests[id]++;
                } else {
                    number_inhibitory_requests[id]++;
                }

                cm.append(source_rank, scr);
            }
        }

        return { cm, number_excitatory_requests, number_inhibitory_requests };
    }

    std::tuple<SynapticElements, std::vector<double>, std::vector<unsigned int>, std::vector<SignalType>>
    create_random_synaptic_elements(size_t number_elements, ElementType element_type, double min_calcium_to_grow,
        double growth_factor = SynapticElements::default_nu, double retract_ratio = SynapticElements::default_vacant_retract_ratio,
        double lb_free_elements = SynapticElements::default_vacant_elements_initially_lower_bound, double ub_free_elements = SynapticElements::default_vacant_elements_initially_upper_bound) {

        SynapticElements se(element_type, min_calcium_to_grow, growth_factor, retract_ratio, lb_free_elements, ub_free_elements);
        se.init(number_elements);

        std::vector<double> grown_elements(number_elements);
        std::vector<unsigned int> connected_elements(number_elements);
        std::vector<SignalType> signal_types(number_elements);

        for (const auto& neuron_id : NeuronID::range(number_elements)) {
            const auto number_grown_elements = get_random_synaptic_element_count();
            const auto number_connected_elements = get_random_synaptic_element_connected_count(static_cast<unsigned int>(number_grown_elements));
            const auto signal_type = get_random_signal_type();

            se.update_grown_elements(neuron_id, number_grown_elements);
            se.update_connected_elements(neuron_id, number_connected_elements);
            se.set_signal_type(neuron_id, signal_type);

            const auto i = neuron_id.get_neuron_id();

            grown_elements[i] = number_grown_elements;
            connected_elements[i] = number_connected_elements;
            signal_types[i] = signal_type;
        }

        return std::make_tuple<SynapticElements, std::vector<double>, std::vector<unsigned int>, std::vector<SignalType>>(std::move(se), std::move(grown_elements), std::move(connected_elements), std::move(signal_types));
    }

    std::shared_ptr<SynapticElements> create_axons(size_t number_elements, double minimal_grown, double maximal_grown) {
        SynapticElements axons(ElementType::Axon, CalciumCalculator::default_C_target);
        axons.init(number_elements);

        for (const auto& neuron_id : NeuronID::range(number_elements)) {
            const auto number_grown_elements = get_random_double(minimal_grown, maximal_grown);
            const auto signal_type = get_random_signal_type();

            axons.update_grown_elements(neuron_id, number_grown_elements);
            axons.update_connected_elements(neuron_id, 0);
            axons.set_signal_type(neuron_id, signal_type);
        }

        return std::make_shared<SynapticElements>(std::move(axons));
    }

    std::shared_ptr<SynapticElements> create_dendrites(size_t number_elements, SignalType signal_type, double minimal_grown, double maximal_grown) {
        SynapticElements dendrites(ElementType::Axon, CalciumCalculator::default_C_target);
        dendrites.init(number_elements);

        for (const auto& neuron_id : NeuronID::range(number_elements)) {
            const auto number_grown_elements = get_random_double(minimal_grown, maximal_grown);

            dendrites.update_grown_elements(neuron_id, number_grown_elements);
            dendrites.update_connected_elements(neuron_id, 0);
            dendrites.set_signal_type(neuron_id, signal_type);
        }

        return std::make_shared<SynapticElements>(std::move(dendrites));
    }

    std::shared_ptr<SynapticElements> create_axons(size_t number_elements) {
        SynapticElements axons(ElementType::Axon, CalciumCalculator::default_C_target);
        axons.init(number_elements);

        for (const auto& neuron_id : NeuronID::range(number_elements)) {
            const auto number_grown_elements = get_random_synaptic_element_count();
            const auto signal_type = get_random_signal_type();

            axons.update_grown_elements(neuron_id, number_grown_elements);
            axons.update_connected_elements(neuron_id, 0);
            axons.set_signal_type(neuron_id, signal_type);
        }

        return std::make_shared<SynapticElements>(std::move(axons));
    }

    std::shared_ptr<SynapticElements> create_dendrites(size_t number_elements, SignalType signal_type) {
        SynapticElements dendrites(ElementType::Axon, CalciumCalculator::default_C_target);
        dendrites.init(number_elements);

        for (const auto& neuron_id : NeuronID::range(number_elements)) {
            const auto number_grown_elements = get_random_synaptic_element_count();

            dendrites.update_grown_elements(neuron_id, number_grown_elements);
            dendrites.update_connected_elements(neuron_id, 0);
            dendrites.set_signal_type(neuron_id, signal_type);
        }

        return std::make_shared<SynapticElements>(std::move(dendrites));
    }

    std::vector<std::tuple<Vec3d, NeuronID>> generate_random_neurons(const Vec3d& min, const Vec3d& max, size_t count, size_t max_id) {
        std::vector<NeuronID> ids(max_id);
        for (auto i = 0; i < max_id; i++) {
            ids[i] = NeuronID(i);
        }
        shuffle(ids.begin(), ids.end());

        std::vector<std::tuple<Vec3d, NeuronID>> return_value(count);
        for (auto i = 0; i < count; i++) {
            return_value[i] = { get_random_position_in_box(min, max), ids[i] };
        }

        return return_value;
    }

    std::vector<size_t> get_random_derangement(size_t size) {
        std::vector<size_t> derangement(size);
        std::iota(derangement.begin(), derangement.end(), 0);

        auto check = [](const std::vector<size_t>& vec) -> bool {
            for (auto i = 0; i < vec.size(); i++) {
                if (i == vec[i]) {
                    return false;
                }
            }
            return true;
        };

        do {
            shuffle(derangement.begin(), derangement.end());
        } while (!check(derangement));

        return derangement;
    }

    std::vector<UpdateStatus> get_update_status(size_t number_neurons, size_t number_disabled) {
        std::vector<UpdateStatus> status(number_disabled, UpdateStatus::Disabled);
        status.resize(number_neurons, UpdateStatus::Enabled);

        shuffle(status.begin(), status.end());

        return status;
    }

    std::vector<UpdateStatus> get_update_status(size_t number_neurons) {
        const auto number_disabled = get_random_integer<size_t>(0, number_neurons);
        return get_update_status(number_neurons, number_disabled);
    }

    std::vector<FiredStatus> get_fired_status(size_t number_neurons, size_t number_inactive) {
        std::vector<FiredStatus> status(number_inactive, FiredStatus::Inactive);
        status.resize(number_neurons, FiredStatus::Fired);

        shuffle(status.begin(), status.end());

        return status;
    }

    std::vector<FiredStatus> get_fired_status(size_t number_neurons) {
        const auto number_disabled = get_random_integer<size_t>(0, number_neurons);
        return get_fired_status(number_neurons, number_disabled);
    }

    std::shared_ptr<NetworkGraph> create_network_graph_all_to_all(size_t number_neurons, int mpi_rank) {
        auto ptr = std::make_shared<NetworkGraph>(number_neurons, mpi_rank);

        for (const auto& source_id : NeuronID::range(number_neurons)) {
            for (const auto& target_id : NeuronID::range(number_neurons)) {
                if (source_id.get_neuron_id() == target_id.get_neuron_id()) {
                    continue;
                }

                const auto weight = get_random_synapse_weight();
                LocalSynapse ls(target_id, source_id, weight);

                ptr->add_synapse(ls);
            }
        }

        return ptr;
    }

    std::shared_ptr<NetworkGraph> create_network_graph(size_t number_neurons, int mpi_rank, unsigned long long number_connections_per_vertex) {
        auto ptr = std::make_shared<NetworkGraph>(number_neurons, mpi_rank);

        for (auto i = 0ULL; i < number_connections_per_vertex; i++) {
            const auto& source_ids = NeuronID::range(number_neurons);
            const auto& target_ids = get_random_derangement(number_neurons);

            for (auto j = 0; j < number_neurons; j++) {

                const auto weight = get_random_synapse_weight();
                LocalSynapse ls(NeuronID(false, target_ids[j]), source_ids[j], weight);
                ptr->add_synapse(ls);
            }
        }

        return ptr;
    }

    std::shared_ptr<NetworkGraph> create_empty_network_graph(size_t number_neurons, int mpi_rank) {
        auto ptr = std::make_shared<NetworkGraph>(number_neurons, mpi_rank);
        return ptr;
    }

    template <typename AdditionalCellAttributes>
    std::vector<std::tuple<Vec3d, size_t>> extract_virtual_neurons(OctreeNode<AdditionalCellAttributes>* root) {
        std::vector<std::tuple<Vec3d, size_t>> return_value{};

        std::stack<std::pair<OctreeNode<AdditionalCellAttributes>*, size_t>> octree_nodes{};
        octree_nodes.emplace(root, 0);

        while (!octree_nodes.empty()) {
            // Don't change this to a reference
            const auto [current_node, level] = octree_nodes.top();
            octree_nodes.pop();

            if (current_node->get_cell().get_neuron_id().is_virtual()) {
                return_value.emplace_back(current_node->get_cell().get_neuron_position().value(), level);
            }

            if (current_node->is_parent()) {
                const auto& childs = current_node->get_children();
                for (auto i = 0; i < 8; i++) {
                    const auto child = childs[i];
                    if (child != nullptr) {
                        octree_nodes.emplace(child, level + 1);
                    }
                }
            }
        }

        return return_value;
    }

    template <typename AdditionalCellAttributes>
    std::vector<OctreeNode<AdditionalCellAttributes>*> extract_branch_nodes(OctreeNode<AdditionalCellAttributes>* root) {
        std::vector<OctreeNode<AdditionalCellAttributes>*> return_value{};

        std::stack<OctreeNode<AdditionalCellAttributes>*> octree_nodes{};
        octree_nodes.push(root);

        while (!octree_nodes.empty()) {
            OctreeNode<AdditionalCellAttributes>* current_node = octree_nodes.top();
            octree_nodes.pop();

            if (current_node->is_leaf()) {
                return_value.emplace_back(current_node);
                continue;
            }

            const auto& childs = current_node->get_children();
            for (auto* child : childs) {
                if (child != nullptr) {
                    octree_nodes.push(child);
                }
            }
        }

        return return_value;
    }

    template <typename AdditionalCellAttributes>
    std::vector<std::tuple<Vec3d, NeuronID>> extract_neurons(OctreeNode<AdditionalCellAttributes>* root) {
        std::vector<std::tuple<Vec3d, NeuronID>> return_value{};

        std::stack<OctreeNode<AdditionalCellAttributes>*> octree_nodes{};
        octree_nodes.push(root);

        while (!octree_nodes.empty()) {
            OctreeNode<AdditionalCellAttributes>* current_node = octree_nodes.top();
            octree_nodes.pop();

            if (current_node->is_parent()) {
                const auto& childs = current_node->get_children();
                for (auto* child : childs) {
                    if (child != nullptr) {
                        octree_nodes.push(child);
                    }
                }
            } else {
                const Cell<AdditionalCellAttributes>& cell = current_node->get_cell();
                const auto neuron_id = cell.get_neuron_id();
                const auto& opt_position = cell.get_neuron_position();

                EXPECT_TRUE(opt_position.has_value());

                const auto& position = opt_position.value();

                if (neuron_id.is_initialized() && !neuron_id.is_virtual()) {
                    return_value.emplace_back(position, neuron_id);
                }
            }
        }

        return return_value;
    }

    template <typename AdditionalCellAttributes>
    std::vector<std::tuple<Vec3d, NeuronID>> extract_neurons_tree(const OctreeImplementation<AdditionalCellAttributes>& octree) {
        const auto root = octree.get_root();
        if (root == nullptr) {
            return {};
        }

        return extract_neurons<AdditionalCellAttributes>(root);
    }

    template <typename Iterator>
    void shuffle(Iterator begin, Iterator end) {
        detail::shuffle(begin, end, mt);
    }

    constexpr static double min_grown_elements = 0.0;
    constexpr static double max_grown_elements = 10.0;

    constexpr static unsigned short small_refinement_level = 5;
    constexpr static unsigned short max_refinement_level = Constants::max_lvl_subdomains;

    constexpr static int upper_bound_my_rank = 32;
    constexpr static int upper_bound_num_ranks = 32;

    constexpr static int upper_bound_num_neurons = 1000;
    constexpr static int upper_bound_num_synapses = 1000;

    constexpr static int number_neurons_out_of_scope = 100;

    constexpr static int bound_synapse_weight = 10;

    mt19937 mt;

    static int iterations;
    static double eps;

private:
    static uniform_int_distribution<unsigned short> uid_refinement;
    static uniform_int_distribution<unsigned short> uid_small_refinement;
    static uniform_int_distribution<unsigned short> uid_large_refinement;

    static uniform_int_distribution<size_t> uid_num_ranks;
    static uniform_int_distribution<size_t> uid_num_neurons;
    static uniform_int_distribution<size_t> uid_num_synapses;

    static uniform_int_distribution<int> uid_synapse_weight;

    static double position_bounary;

    static bool use_predetermined_seed;
    static unsigned int predetermined_seed;
};

class RelearnTestWithAdditionalCellAttribute : public RelearnTest {
protected:
    template <typename AdditionalCellAttributes>
    static void init() {
        RelearnTest::init();
        MPIWrapper::init_buffer_octree<AdditionalCellAttributes>();
    }

protected:
    template <typename AdditionalCellAttributes>
    static void SetUpTestCaseTemplate() {
        RelearnTest::SetUpTestCaseTemplate();
        init<AdditionalCellAttributes>();
    }
};

class NetworkGraphTest : public RelearnTest {
protected:
    static int num_ranks;
    static int num_synapses_per_neuron;

    static void SetUpTestSuite() {
        SetUpTestCaseTemplate();
    }

    template <typename T>
    void erase_empty(std::map<T, int>& edges) {
        for (auto iterator = edges.begin(); iterator != edges.end();) {
            if (iterator->second == 0) {
                iterator = edges.erase(iterator);
            } else {
                ++iterator;
            }
        }
    }

    template <typename T>
    void erase_empties(std::map<T, std::map<T, int>>& edges) {
        for (auto iterator = edges.begin(); iterator != edges.end();) {
            erase_empty<T>(iterator->second);

            if (iterator->second.empty()) {
                iterator = edges.erase(iterator);
            } else {
                ++iterator;
            }
        }
    }
};

class NeuronAssignmentTest : public RelearnTest {
protected:
    static void SetUpTestSuite() {
        SetUpTestCaseTemplate();
    }
    double calculate_box_length(const size_t number_neurons, const double um_per_neuron) const noexcept {
        return ceil(pow(static_cast<double>(number_neurons), 1 / 3.)) * um_per_neuron;
    }

    void generate_random_neurons(std::vector<Vec3d>& positions,
        std::vector<RelearnTypes::area_id>& neuron_id_to_area_ids, std::vector<RelearnTypes::area_name>& area_id_to_area_name, std::vector<SignalType>& types);
};

class NeuronModelsTest : public RelearnTest {
protected:
    static void SetUpTestSuite() {
        SetUpTestCaseTemplate();
    }

    void test_update(std::unique_ptr<NeuronModel> model, std::shared_ptr<NetworkGraph> ng, size_t number_neurons);
};

class RankNeuronIdTest : public RelearnTest {
protected:
    static void SetUpTestSuite() {
        SetUpTestCaseTemplate();
    }
};

class NeuronsTest : public RelearnTest {
protected:
    static void SetUpTestSuite() {
        SetUpTestCaseTemplate();
    }

    void assert_empty(const NeuronsExtraInfo& nei, size_t number_neurons);

    void assert_contains(const NeuronsExtraInfo& nei, size_t number_neurons, size_t num_neurons_check, const std::vector<Vec3d>& expected_positions);
};

class CellTest : public RelearnTest {
protected:
    static void SetUpTestSuite() {
        SetUpTestCaseTemplate();
    }

    template <typename AdditionalCellAttributes>
    void test_cell_size();

    template <typename AdditionalCellAttributes>
    void test_cell_dendrites_position();

    template <typename AdditionalCellAttributes>
    void test_cell_dendrites_position_exception();

    template <typename AdditionalCellAttributes>
    void test_cell_set_number_dendrites();

    template <typename AdditionalCellAttributes>
    void test_cell_dendrites_position_combined();

    template <typename AdditionalCellAttributes>
    void test_cell_axons_position();

    template <typename AdditionalCellAttributes>
    void test_cell_axons_position_exception();

    template <typename AdditionalCellAttributes>
    void test_cell_set_number_axons();

    template <typename AdditionalCellAttributes>
    void test_cell_axons_position_combined();

    template <typename AdditionalCellAttributes>
    void test_cell_set_neuron_id();

    template <typename AdditionalCellAttributes>
    void test_cell_octants();

    template <typename AdditionalCellAttributes>
    void test_cell_octants_exception();

    template <typename AdditionalCellAttributes>
    void test_cell_octants_size();

    template <typename VirtualPlasticityElement>
    void test_vpe_number_elements();

    template <typename VirtualPlasticityElement>
    void test_vpe_position();

    template <typename VirtualPlasticityElement>
    void test_vpe_mixed();
};

template <typename AdditionalCellAttributes>
class OctreeTest : public RelearnTestWithAdditionalCellAttribute {
protected:
    static void SetUpTestSuite() {
        SetUpTestCaseTemplate<AdditionalCellAttributes>();
    }
};

class BarnesHutTest : public RelearnTestWithAdditionalCellAttribute {
protected:
    static void SetUpTestSuite() {
        SetUpTestCaseTemplate<BarnesHutCell>();
    }
};

class BarnesHutInvertedTest : public RelearnTestWithAdditionalCellAttribute {
protected:
    static void SetUpTestSuite() {
        SetUpTestCaseTemplate<BarnesHutInvertedCell>();
    }
};

// class FMMTest : public RelearnTest {
// protected:
//     static void SetUpTestSuite() {
//         SetUpTestCaseTemplate<FastMultipoleMethodsCell>();
//     }
/*   Stack<FastMultipoleMethodsBase::stack_entry> init_stack(FastMultipoleMethods fmm, const SignalType signal_type_needed) { return FastMultipoleMethodsBase::init_stack(signal_type_needed); }

   void unpack_node_pair(FastMultipoleMethods fmm, Stack<FastMultipoleMethodsBase::stack_entry>& stack) { return FastMultipoleMethodsBase::unpack_node_pair(stack); }

   FastMultipoleMethodsBase::interaction_list_type align_interaction_list(FastMultipoleMethods fmm, OctreeNode<FastMultipoleMethods::AdditionalCellAttributes>* source_node, OctreeNode<FastMultipoleMethods::AdditionalCellAttributes>* target_parent, const SignalType signal_type) { return fmm.align_interaction_list(source_node, target_parent, signal_type); }*/

// std::array<double, Constants::p3> calc_hermite_coefficients(const OctreeNode<FastMultipoleMethodsCell>* source, double sigma, SignalType signal_type_needed) { return FastMultipoleMethods::calc_hermite_coefficients(source, sigma, signal_type_needed); }
// CalculationType check_calculation_requirements(const OctreeNode<FastMultipoleMethodsCell>* source, const OctreeNode<FastMultipoleMethodsCell>* target, double sigma, SignalType signal_type_needed){return FastMultipoleMethods::check_calculation_requirements(source, target, sigma, signal_type_needed);}
// const std::vector<std::pair<FastMultipoleMethods::position_type, FastMultipoleMethods::counter_type>> get_all_positions_for(OctreeNode<FastMultipoleMethodsCell>* node, const ElementType type, const SignalType signal_type_needed){return FastMultipoleMethodsBase<FastMultipoleMethodsCell>::get_all_positions_for(node, type, signal_type_needed);}
// double calc_taylor(const OctreeNode<FastMultipoleMethodsCell>* source, OctreeNode<FastMultipoleMethodsCell>* target, double sigma, SignalType signal_type_needed){return FastMultipoleMethods::calc_taylor(source, target, sigma, signal_type_needed);}
// double calc_direct_gauss(OctreeNode<FastMultipoleMethodsCell>* source, OctreeNode<FastMultipoleMethodsCell>* target, double sigma, SignalType signal_type_needed){return FastMultipoleMethods::calc_direct_gauss(source, target, sigma, signal_type_needed);}
// double calc_hermite(const OctreeNode<FastMultipoleMethodsCell>* source, OctreeNode<FastMultipoleMethodsCell>* target, const std::array<double, Constants::p3>& coefficients_buffer, double sigma, SignalType signal_type_needed){return FastMultipoleMethods::calc_hermite(source, target, coefficients_buffer, sigma, signal_type_needed);}
//};

class PartitionTest : public RelearnTest {
protected:
    static void SetUpTestSuite() {
        SetUpTestCaseTemplate();
    }
};

class NeuronIdTest : public RelearnTest {
protected:
    static void SetUpTestSuite() {
        SetUpTestCaseTemplate();
    }
};

class ConnectorTest : public RelearnTestWithAdditionalCellAttribute {
protected:
    static void SetUpTestSuite() {
        SetUpTestCaseTemplate<BarnesHutCell>();
    }
};

class KernelTest : public RelearnTest {
protected:
    static void SetUpTestSuite() {
        SetUpTestCaseTemplate();
    }
};

class ProbabilityKernelTest : public RelearnTest {
protected:
    static void SetUpTestSuite() {
        SetUpTestCaseTemplate();
    }
};

class SynapticElementsTest : public RelearnTest {
protected:
    static void SetUpTestSuite() {
        SetUpTestCaseTemplate();
    }
};

class VectorTest : public RelearnTest {
protected:
    static void SetUpTestSuite() {
        SetUpTestCaseTemplate();
    }
};

class SpaceFillingCurveTest : public RelearnTest {
protected:
    static void SetUpTestSuite() {
        SetUpTestCaseTemplate();
    }
};

template <typename T>
class TaggedIDTest : public RelearnTest {
protected:
    static void SetUpTestSuite() {
        SetUpTestCaseTemplate();
    }

    static bool get_initialized(const TaggedID<T>& id) {
        return id.is_initialized_;
    }

    static bool get_virtual(const TaggedID<T>& id) {
        return id.is_virtual_;
    }

    static bool get_global(const TaggedID<T>& id) {
        return id.is_global_;
    }

    static typename TaggedID<T>::value_type get_id(const TaggedID<T>& id) {
        return id.id_;
    }

    static_assert(sizeof(typename TaggedID<T>::value_type) == sizeof(T));
};

class IOTest : public RelearnTest {
protected:
    static void SetUpTestSuite() {
        SetUpTestCaseTemplate();
    }
};

class CalciumCalculatorTest : public RelearnTest {
protected:
    static void SetUpTestSuite() {
        SetUpTestCaseTemplate();
    }
};

class SynapticInputTest : public RelearnTest {
protected:
    static void SetUpTestSuite() {
        SetUpTestCaseTemplate();
    }
};

class BackgroundActivityTest : public RelearnTest {
protected:
    static void SetUpTestSuite() {
        SetUpTestCaseTemplate();
    }
};

class StepParserTest : public RelearnTest {
protected:
    static void SetUpTestSuite() {
        SetUpTestCaseTemplate();
    }

    Interval generate_random_interval() {
        using int_type = Interval::step_type;

        constexpr auto min = std::numeric_limits<int_type>::min();
        constexpr auto max = std::numeric_limits<int_type>::max();

        const auto begin = get_random_integer<int_type>(min, max);
        const auto end = get_random_integer<int_type>(min, max);
        const auto frequency = get_random_integer<int_type>(min, max);

        return Interval{ std::min(begin, end), std::max(begin, end), frequency };
    }

    std::string codify_interval(const Interval& interval) {
        std::stringstream ss{};
        ss << interval.begin << '-' << interval.end << ':' << interval.frequency;
        return ss.str();
    }

    std::pair<Interval, std::string> generate_random_interval_description() {
        auto interval = generate_random_interval();
        auto description = codify_interval(interval);
        return { std::move(interval), std::move(description) };
    }
};

class LocalAreaTranslatorTest : public RelearnTest {
protected:
    static void SetUpTestSuite() {
        SetUpTestCaseTemplate();
    }
};

class MonitorParserTest : public RelearnTest {
protected:
    static void SetUpTestSuite() {
        SetUpTestCaseTemplate();
    }

    RankNeuronId generate_random_rank_neuron_id() {
        using rank_type = int;
        using neuron_id_type = NeuronID::value_type;

        const auto rank = get_random_integer<rank_type>(0, std::numeric_limits<rank_type>::max());
        const auto neuron_id = get_random_integer<neuron_id_type>(std::numeric_limits<neuron_id_type>::min(), std::numeric_limits<neuron_id_type>::min());

        return { rank, NeuronID(neuron_id) };
    }

    std::string codify_rank_neuron_id(const RankNeuronId& rni) {
        std::stringstream ss{};
        ss << rni.get_rank() << ':' << rni.get_neuron_id();
        return ss.str();
    }

    std::pair<RankNeuronId, std::string> generate_random_rank_neuron_id_description() {
        auto rank_neuron_id = generate_random_rank_neuron_id();
        auto description = codify_rank_neuron_id(rank_neuron_id);
        return { std::move(rank_neuron_id), std::move(description) };
    }
};

class MPIRankTest : public RelearnTest {
protected:
    static void SetUpTestSuite() {
        SetUpTestCaseTemplate();
    }
};

class MiscTest : public RelearnTest {
protected:
    static void SetUpTestSuite() {
        SetUpTestCaseTemplate();
    }
};
