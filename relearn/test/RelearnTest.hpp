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
#include "RandomAdapter.h"
#include "Types.h"
#include "io/LogFiles.h"
#include "mpi/MPIWrapper.h"
#include "neurons/ElementType.h"
#include "neurons/FiredStatus.h"
#include "neurons/NetworkGraph.h"
#include "neurons/SignalType.h"
#include "neurons/helper/DistantNeuronRequests.h"
#include "neurons/helper/SynapseCreationRequests.h"
#include "structure/Cell.h"
#include "structure/Octree.h"
#include "structure/OctreeNode.h"
#include "util/MemoryHolder.h"
#include "util/RelearnException.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <map>
#include <random>
#include <stack>
#include <tuple>
#include <vector>

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
        return RandomAdapter::get_random_double(min, max, mt);
    }

    template <typename T>
    T get_random_integer(T min, T max) {
        return RandomAdapter::get_random_integer<T>(min, max, mt);
    }

    static std::vector<RelearnTypes::area_name> get_neuron_id_vs_area_name(const std::vector<RelearnTypes::area_id>& neuron_id_vs_area_id, const std::vector<RelearnTypes::area_name>& area_id_vs_area_name) {
        std::vector<RelearnTypes::area_name> neuron_id_vs_area_name{};

        for (auto i : neuron_id_vs_area_id) {
            neuron_id_vs_area_name.emplace_back(area_id_vs_area_name[i]);
        }
        return neuron_id_vs_area_name;
    }

    std::tuple<Vec3d, Vec3d> get_random_simulation_box_size() {
        const auto rand_x_1 = RandomAdapter::get_random_double(-position_bounary, +position_bounary, mt);
        const auto rand_x_2 = RandomAdapter::get_random_double(-position_bounary, +position_bounary, mt);
                                                                                                  
        const auto rand_y_1 = RandomAdapter::get_random_double(-position_bounary, +position_bounary, mt);
        const auto rand_y_2 = RandomAdapter::get_random_double(-position_bounary, +position_bounary, mt);
                                                                                                   
        const auto rand_z_1 = RandomAdapter::get_random_double(-position_bounary, +position_bounary, mt);
        const auto rand_z_2 = RandomAdapter::get_random_double(-position_bounary, +position_bounary, mt);

        return {
            { std::min(rand_x_1, rand_x_2), std::min(rand_y_1, rand_y_2), std::min(rand_z_1, rand_z_2) },
            { std::max(rand_x_1, rand_x_2), std::max(rand_y_1, rand_y_2), std::max(rand_z_1, rand_z_2) }
        };
    }

    double get_random_position_element() {
        const auto val = RandomAdapter::get_random_double(-position_bounary, +position_bounary, mt);
        return val;
    }

    Vec3d get_random_position() {
        const auto x = RandomAdapter::get_random_double(-position_bounary, +position_bounary, mt);
        const auto y = RandomAdapter::get_random_double(-position_bounary, +position_bounary, mt);
        const auto z = RandomAdapter::get_random_double(-position_bounary, +position_bounary, mt);

        return { x, y, z };
    }

    Vec3d get_minimum_position() {
        return { -position_bounary, -position_bounary, -position_bounary };
    }

    Vec3d get_maximum_position() {
        return { position_bounary, position_bounary, position_bounary };
    }

    Vec3d get_random_position_in_box(const Vec3d& min, const Vec3d& max) {
        const auto x = RandomAdapter::get_random_double(min.get_x(), max.get_x(), mt);
        const auto y = RandomAdapter::get_random_double(min.get_y(), max.get_y(), mt);
        const auto z = RandomAdapter::get_random_double(min.get_z(), max.get_z(), mt);

        return { x, y, z };
    }

    double get_random_percentage() {
        return RandomAdapter::get_random_percentage<double>(mt);
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
        return RandomAdapter::get_random_bool(mt);
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
        const auto number_disabled = RandomAdapter::get_random_integer<size_t>(0, number_neurons, mt);
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
        RandomAdapter::shuffle(begin, end, mt);
    }

    constexpr static unsigned short small_refinement_level = 5;
    constexpr static unsigned short max_refinement_level = Constants::max_lvl_subdomains;

    constexpr static int upper_bound_my_rank = 32;
    constexpr static int upper_bound_num_ranks = 32;

    constexpr static int upper_bound_num_neurons = 1000;

    constexpr static int number_neurons_out_of_scope = 100;

    std::mt19937 mt;

    static int iterations;
    static double eps;

private:
    static boost::random::uniform_int_distribution<unsigned short> uid_refinement;
    static boost::random::uniform_int_distribution<unsigned short> uid_small_refinement;
    static boost::random::uniform_int_distribution<unsigned short> uid_large_refinement;
           
    static boost::random::uniform_int_distribution<size_t> uid_num_ranks;
    static boost::random::uniform_int_distribution<size_t> uid_num_neurons;

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
