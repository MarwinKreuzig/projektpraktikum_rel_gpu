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
#include "gtest/gtest-typed-test.h"

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
#include "neurons/helper/DistantNeuronRequests.h"
#include "neurons/helper/SynapseCreationRequests.h"
#include "neurons/models/SynapticElements.h"
#include "structure/Cell.h"
#include "structure/Octree.h"
#include "structure/OctreeNode.h"
#include "structure/Partition.h"
#include "util/Interval.h"
#include "util/MemoryHolder.h"
#include "util/MPIRank.h"
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

    unsigned int get_random_synaptic_element_connected_count(unsigned int maximum) {
        return get_random_integer<unsigned int>(0, maximum);
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

    DistantNeuronRequest::TargetNeuronType get_random_target_neuron_type() {
        uniform_int_distribution<unsigned short> uid_3(0, 2);
        const auto drawn = uid_3(mt);
    
        if (drawn == 0) {
            return DistantNeuronRequest::TargetNeuronType::BranchNode;
        } 

        if (drawn == 1) {
            return DistantNeuronRequest::TargetNeuronType::Leaf;
        }

        return DistantNeuronRequest::TargetNeuronType::VirtualNode;
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

    constexpr static int number_neurons_out_of_scope = 100;

    mt19937 mt;

    static int iterations;
    static double eps;

private:
    static uniform_int_distribution<unsigned short> uid_refinement;
    static uniform_int_distribution<unsigned short> uid_small_refinement;
    static uniform_int_distribution<unsigned short> uid_large_refinement;

    static uniform_int_distribution<size_t> uid_num_ranks;
    static uniform_int_distribution<size_t> uid_num_neurons;

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
