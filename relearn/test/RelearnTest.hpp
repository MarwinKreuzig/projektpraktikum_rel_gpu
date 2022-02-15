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

#include "../source/algorithm/BarnesHutCell.h"
#include "../source/algorithm/FastMultipoleMethodsCell.h"
#include "../source/io/LogFiles.h"
#include "../source/mpi/MPIWrapper.h"
#include "../source/neurons/ElementType.h"
#include "../source/neurons/SignalType.h"
#include "../source/neurons/models/SynapticElements.h"
#include "../source/structure/OctreeNode.h"
#include "../source/util/MemoryHolder.h"
#include "../source/util/RelearnException.h"
#include "../source/util/TaggedID.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <map>
#include <random>
#include <tuple>
#include <vector>

class NeuronsExtraInfo;

inline bool initialized = false;

class RelearnTest : public ::testing::Test {
protected:
    template <typename AdditionalCellAttributes>
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
        MPIWrapper::init_buffer_octree<AdditionalCellAttributes>();
        template_initialized = true;
    }

protected:
    template <typename AdditionalCellAttributes>
    static void SetUpTestCaseTemplate() {
        RelearnException::hide_messages = true;
        LogFiles::disable = true;

        init<AdditionalCellAttributes>();
    }

    static void SetUpTestCase();

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
        std::uniform_real_distribution<double> urd(min, max);
        return urd(mt);
    }

    template <typename T>
    T get_random_integer(T min, T max) {
        std::uniform_int_distribution<T> uid(min, max);
        return uid(mt);
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

    Vec3d get_random_position() {
        const auto x = get_random_double(-position_bounary, +position_bounary);
        const auto y = get_random_double(-position_bounary, +position_bounary);
        const auto z = get_random_double(-position_bounary, +position_bounary);

        return { x, y, z };
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
        std::uniform_int_distribution<size_t> uid(offset, offset + number_neurons - 1);
        return NeuronID{ uid(mt) };
    }

    int get_random_synapse_weight() {
        return uid_synapse_weight(mt);
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
        std::uniform_int_distribution<unsigned short> uid_bool(0, 1);
        return uid_bool(mt) == 0;
    }

    ElementType get_random_element_type() noexcept {
        return get_random_bool() ? ElementType::AXON : ElementType::DENDRITE;
    }

    SignalType get_random_signal_type() noexcept {
        return get_random_bool() ? SignalType::EXCITATORY : SignalType::INHIBITORY;
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

        for (auto neuron_id : NeuronID::range(number_elements)) { 
            const auto number_grown_elements = get_random_synaptic_element_count();
            const auto number_connected_elements = get_random_synaptic_element_connected_count(static_cast<unsigned int>(number_grown_elements));
            const auto signal_type = get_random_signal_type();

            se.update_grown_elements(neuron_id, number_grown_elements);
            se.update_connected_elements(neuron_id, number_connected_elements);
            se.set_signal_type(neuron_id, signal_type);

            const auto i = neuron_id.get_local_id();

            grown_elements[i] = number_grown_elements;
            connected_elements[i] = number_connected_elements;
            signal_types[i] = signal_type;
        }

        return std::make_tuple<SynapticElements, std::vector<double>, std::vector<unsigned int>, std::vector<SignalType>>(std::move(se), std::move(grown_elements), std::move(connected_elements), std::move(signal_types));
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

    std::mt19937 mt;

    static int iterations;
    static double eps;

private:
    static std::uniform_int_distribution<unsigned short> uid_refinement;
    static std::uniform_int_distribution<unsigned short> uid_small_refinement;
    static std::uniform_int_distribution<unsigned short> uid_large_refinement;

    static std::uniform_int_distribution<size_t> uid_num_ranks;
    static std::uniform_int_distribution<size_t> uid_num_neurons;
    static std::uniform_int_distribution<size_t> uid_num_synapses;

    static std::uniform_int_distribution<int> uid_synapse_weight;

    static double position_bounary;

    static bool use_predetermined_seed;
    static unsigned int predetermined_seed;
};

class NetworkGraphTest : public RelearnTest {
protected:
    static int num_ranks;
    static int num_synapses_per_neuron;

    static void SetUpTestCase() {
        SetUpTestCaseTemplate<BarnesHutCell>();
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
    static void SetUpTestCase() {
        SetUpTestCaseTemplate<BarnesHutCell>();
    }
    double calculate_box_length(const size_t number_neurons, const double um_per_neuron) const noexcept {
        return ceil(pow(static_cast<double>(number_neurons), 1 / 3.)) * um_per_neuron;
    }

    void generate_neuron_positions(std::vector<Vec3d>& positions,
        std::vector<std::string>& area_names, std::vector<SignalType>& types);

    void generate_synapses(std::vector<std::tuple<NeuronID, NeuronID, int>>& synapses, size_t number_neurons);
};

class NeuronModelsTest : public RelearnTest {
protected:
    static void SetUpTestCase() {
        SetUpTestCaseTemplate<BarnesHutCell>();
    }
};

class RankNeuronIdTest : public RelearnTest {
protected:
    static void SetUpTestCase() {
        SetUpTestCaseTemplate<BarnesHutCell>();
    }
};

class NeuronsTest : public RelearnTest {
protected:
    static void SetUpTestCase() {
        SetUpTestCaseTemplate<BarnesHutCell>();
    }

    void assert_empty(const NeuronsExtraInfo& nei, size_t number_neurons);

    void assert_contains(const NeuronsExtraInfo& nei, size_t number_neurons, size_t num_neurons_check,
        const std::vector<std::string>& expected_area_names, const std::vector<Vec3d>& expected_positions);
};

class CellTest : public RelearnTest {
protected:
    static void SetUpTestCase() {
        SetUpTestCaseTemplate<BarnesHutCell>();
    }

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
protected:
    static void SetUpTestCase() {
        SetUpTestCaseTemplate<BarnesHutCell>();
    }
};

class OctreeTestFMM : public RelearnTest {
protected:
    static void SetUpTestCase() {
        SetUpTestCaseTemplate<FastMultipoleMethodsCell>();
    }
};

class PartitionTest : public RelearnTest {
protected:
    static void SetUpTestCase() {
        SetUpTestCaseTemplate<BarnesHutCell>();
    }
};

class NeuronIdTest : public RelearnTest {
protected:
    static void SetUpTestCase() {
        SetUpTestCaseTemplate<BarnesHutCell>();
    }
};

class SynapticElementsTest : public RelearnTest {
protected:
    static void SetUpTestCase() {
        SetUpTestCaseTemplate<BarnesHutCell>();
    }
};

class VectorTest : public RelearnTest {
protected:
    static void SetUpTestCase() {
        SetUpTestCaseTemplate<BarnesHutCell>();
    }
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
    static void SetUpTestCase() {
        SetUpTestCaseTemplate<BarnesHutCell>();
    }
};

template <typename T>
class TaggedIDTest : public RelearnTest {
protected:
    static void SetUpTestCase() {
        SetUpTestCaseTemplate<BarnesHutCell>();
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

    static TaggedID<T>::value_type get_id(const TaggedID<T>& id) {
        return id.id_;
    }

    static_assert(sizeof(typename TaggedID<T>::value_type) == sizeof(T));
};
