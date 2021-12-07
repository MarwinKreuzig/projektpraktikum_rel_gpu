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

#include "../../../source/algorithm/BarnesHutCell.h"
#include "../../../source/structure/OctreeNode.h"
#include "../../../source/mpi/MPIWrapper.h"
#include "../../../source/io/LogFiles.h"
#include "../../../source/util/MemoryHolder.h"
#include "../../../source/util/RelearnException.h"

#include <chrono>
#include <cmath>
#include <map>
#include <random>
#include <tuple>
#include <vector>

class NeuronsExtraInfo;

class RelearnTest : public ::testing::Test {
private:
    static void init() {
        static bool initialized = false;

        if (initialized) {
            return;
        }

        initialized = true;

        char* argument = (char*)"./runTests";
        MPIWrapper::init(1, &argument);
        MPIWrapper::init_buffer_octree<BarnesHutCell>();
    }

protected:
    static void SetUpTestCase() {
        RelearnException::hide_messages = true;
        LogFiles::disable = true;

        init();
    }

    static void TearDownTestCase() {
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
    }

    void TearDown() override {
        std::cerr << "Test finished\n";
    }

    void make_mpi_mem_available() {
        MemoryHolder<BarnesHutCell>::make_all_available();
    }

    std::tuple<Vec3d, Vec3d> get_random_simulation_box_size() {
        std::uniform_real_distribution<double> urd(-position_bounary, +position_bounary);

        const auto rand_x_1 = urd(mt);
        const auto rand_x_2 = urd(mt);

        const auto rand_y_1 = urd(mt);
        const auto rand_y_2 = urd(mt);

        const auto rand_z_1 = urd(mt);
        const auto rand_z_2 = urd(mt);

        return {
            { std::min(rand_x_1, rand_x_2), std::min(rand_y_1, rand_y_2), std::min(rand_z_1, rand_z_2) },
            { std::max(rand_x_1, rand_x_2), std::max(rand_y_1, rand_y_2), std::max(rand_z_1, rand_z_2) }
        };
    }

    Vec3d get_random_position() {
        std::uniform_real_distribution<double> urd(-position_bounary, +position_bounary);

        const auto x = urd(mt);
        const auto y = urd(mt);
        const auto z = urd(mt);

        return { x, y, z };
    }

    Vec3d get_random_position_in_box(const Vec3d& min, const Vec3d& max) {
        std::uniform_real_distribution urd_x(min.get_x(), max.get_x());
        std::uniform_real_distribution urd_y(min.get_y(), max.get_y());
        std::uniform_real_distribution urd_z(min.get_z(), max.get_z());

        return {
            urd_x(mt), urd_y(mt), urd_z(mt)
        };
    }

    size_t round_to_next_exponent(size_t numToRound, size_t exponent) {
        auto log = std::log(static_cast<double>(numToRound)) / std::log(static_cast<double>(exponent));
        auto rounded_exp = std::ceil(log);
        auto new_val = std::pow(static_cast<double>(exponent), rounded_exp);
        return static_cast<size_t>(new_val);
    }

    size_t get_random_number_ranks() {
        return uid_num_ranks(mt);
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

    size_t get_random_neuron_id(size_t number_neurons) {
        std::uniform_int_distribution<size_t> uid(0, number_neurons - 1);
        return uid(mt);
    }

    int get_random_synapse_weight() {
        return uid_synapse_weight(mt);
    }

    std::vector<std::tuple<size_t, size_t, int>> get_random_synapses(size_t number_neurons, size_t number_synapses) {
        std::vector<std::tuple<size_t, size_t, int>> synapses(number_synapses);

        for (auto i = 0; i < number_synapses; i++) {
            const auto source_id = get_random_neuron_id(number_neurons);
            const auto target_id = get_random_neuron_id(number_neurons);
            const auto weight = get_random_synapse_weight();

            synapses[i] = { source_id, target_id, weight };
        }

        return synapses;
    }

    double get_random_percentage() {
        return urd_percentage(mt);
    }

    std::mt19937 mt;

    constexpr static int upper_bound_my_rank = 32;
    constexpr static int upper_bound_num_ranks = 32;

    constexpr static int upper_bound_num_neurons = 1000;
    constexpr static int upper_bound_num_synapses = 1000;

    constexpr static int number_neurons_out_of_scope = 100;

    constexpr static int bound_synapse_weight = 10;

    static std::uniform_int_distribution<size_t> uid_num_ranks;
    static std::uniform_int_distribution<size_t> uid_num_neurons;
    static std::uniform_int_distribution<size_t> uid_num_synapses;

    static std::uniform_int_distribution<int> uid_synapse_weight;

    static std::uniform_real_distribution<double> urd_percentage;

    static double position_bounary;

    static int iterations;
    static double eps;

    static bool use_predetermined_seed;
    static unsigned int predetermined_seed;
};

class NetworkGraphTest : public RelearnTest {
protected:
    static int num_ranks;
    static int num_synapses_per_neuron;

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
    double calculate_box_length(const size_t number_neurons, const double um_per_neuron) const noexcept {
        return ceil(pow(static_cast<double>(number_neurons), 1 / 3.)) * um_per_neuron;
    }

    void generate_neuron_positions(std::vector<Vec3d>& positions,
        std::vector<std::string>& area_names, std::vector<SignalType>& types);

    void generate_synapses(std::vector<std::tuple<size_t, size_t, int>>& synapses, size_t number_neurons);
};

class NeuronModelsTest : public RelearnTest {
};

class NeuronsTest : public RelearnTest {
protected:
    void assert_empty(const NeuronsExtraInfo& nei, size_t number_neurons);

    void assert_contains(const NeuronsExtraInfo& nei, size_t number_neurons, size_t num_neurons_check,
        const std::vector<std::string>& expected_area_names, const std::vector<Vec3d>& expected_positions);
};

class CellTest : public RelearnTest {
protected:
    template <typename AdditionalCellAttributes>
    void test_cell_size();

    template <typename AdditionalCellAttributes>
    void test_cell_position();

    template <typename AdditionalCellAttributes>
    void test_cell_position_exception();

    template <typename AdditionalCellAttributes>
    void test_cell_position_combined();

    template <typename AdditionalCellAttributes>
    void test_cell_set_number_dendrites();

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

class OctreeTest : public RelearnTest {
};

class PartitionTest : public RelearnTest {
};

class SynapticElementsTest : public RelearnTest {
};

class VectorTest : public RelearnTest {
protected:
    double get_random_vector_element() noexcept {
        return uniform_vector_elements(mt);
    }

private:
    constexpr static double lower_bound = -100.0;
    constexpr static double upper_bound = 100.0;

    static std::uniform_real_distribution<double> uniform_vector_elements;
};

class SpaceFillingCurveTest : public RelearnTest {
protected:
    uint8_t get_random_refinement_level() noexcept {
        return static_cast<uint8_t>(uid_refinement(mt));
    }

    uint8_t get_small_refinement_level() noexcept {
        return static_cast<uint8_t>(uid_small_refinement(mt));
    }

    uint8_t get_large_refinement_level() noexcept {
        return static_cast<uint8_t>(uid_large_refinement(mt));
    }

    constexpr static unsigned short small_refinement_level = 5;
    constexpr static unsigned short max_refinement_level = Constants::max_lvl_subdomains;

private:
    static std::uniform_int_distribution<unsigned short> uid_refinement;
    static std::uniform_int_distribution<unsigned short> uid_small_refinement;
    static std::uniform_int_distribution<unsigned short> uid_large_refinement;
};
