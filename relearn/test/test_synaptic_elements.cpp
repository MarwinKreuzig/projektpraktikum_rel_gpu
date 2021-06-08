#include "../googletest/include/gtest/gtest.h"

#include <map>
#include <random>
#include <tuple>
#include <vector>

#include "RelearnTest.hpp"

#include "../source/util/RelearnException.h"
#include "../source/neurons/models/SynapticElements.h"
#include "../source/neurons/ElementType.h"

constexpr const size_t upper_bound_num_neurons = 10000;

namespace TestSynapticElements {

TEST_F(RelearnTest, testSynapticElementsConstructor) {
    std::uniform_int_distribution<size_t> uid_num_neurons(0, upper_bound_num_neurons);
    std::uniform_int_distribution<int> uid_bool(0, 1);

    for (auto i = 0; i < iterations; i++) {
        size_t num_neurons = uid_num_neurons(mt);

        ElementType element_type = uid_bool(mt) == 0 ? ElementType::AXON : ElementType::DENDRITE;

        SynapticElements synaptic_elements(element_type, 0.0);
        synaptic_elements.init(num_neurons);

        for (size_t neuron_id = 0; neuron_id < num_neurons; neuron_id++) {
            const double cnt = synaptic_elements.get_cnt(neuron_id);
            const double conn_cnt = synaptic_elements.get_connected_cnt(neuron_id);
            const double delta_cnt = synaptic_elements.get_delta_cnt(neuron_id);

            ASSERT_EQ(cnt, 0.0);
            ASSERT_EQ(conn_cnt, 0.0);
            ASSERT_EQ(delta_cnt, 0.0);
        }

        const std::vector<double>& cnts = synaptic_elements.get_cnts();
        const std::vector<double>& conn_cnts = synaptic_elements.get_cnts();
        const std::vector<double>& delta_cnts = synaptic_elements.get_cnts();
        const std::vector<SignalType>& types = synaptic_elements.get_signal_types();

        ASSERT_EQ(cnts.size(), num_neurons);
        ASSERT_EQ(conn_cnts.size(), num_neurons);
        ASSERT_EQ(delta_cnts.size(), num_neurons);
        ASSERT_EQ(types.size(), num_neurons);

        for (double d : cnts) {
            ASSERT_EQ(d, 0.0);
        }

        for (double d : conn_cnts) {
            ASSERT_EQ(d, 0.0);
        }

        for (double d : delta_cnts) {
            ASSERT_EQ(d, 0.0);
        }
    }
}

TEST_F(RelearnTest, testSynapticElementsConstructorException) {
    std::uniform_int_distribution<size_t> uid_num_neurons(0, upper_bound_num_neurons);
    std::uniform_int_distribution<int> uid_bool(0, 1);

    std::uniform_real_distribution<double> urd_C;

    for (auto i = 0; i < iterations; i++) {
        size_t num_neurons = uid_num_neurons(mt);

        ElementType element_type = uid_bool(mt) == 0 ? ElementType::AXON : ElementType::DENDRITE;

        SynapticElements synaptic_elements(element_type, 0.0);
        synaptic_elements.init(num_neurons);

        std::vector<double> golden_cnts(num_neurons);
        std::vector<unsigned int> golden_conn_cnts(num_neurons);
        std::vector<double> golden_delta_cnts(num_neurons);
        std::vector<SignalType> golden_signal_types(num_neurons);

        for (size_t neuron_id = 0; neuron_id < num_neurons; neuron_id++) {
            const double cnt = urd_C(mt);
            const double conn_cnt = urd_C(mt);
            const double delta_cnt = urd_C(mt);
            const SignalType signal_type = uid_bool(mt) == 0 ? SignalType::EXCITATORY : SignalType::INHIBITORY;

            golden_cnts[neuron_id] = cnt;
            golden_conn_cnts[neuron_id] = static_cast<unsigned int>(conn_cnt);
            golden_delta_cnts[neuron_id] = delta_cnt;
            golden_signal_types[neuron_id] = signal_type;

            synaptic_elements.update_cnt(neuron_id, cnt);
            synaptic_elements.update_conn_cnt(neuron_id, conn_cnt);
            synaptic_elements.update_delta_cnt(neuron_id, delta_cnt);
            synaptic_elements.set_signal_type(neuron_id, signal_type);
        }

        for (size_t neuron_id = num_neurons; neuron_id < num_neurons + 10; neuron_id++) {
            const double cnt = urd_C(mt);
            const unsigned int conn_cnt = urd_C(mt);
            const double delta_cnt = urd_C(mt);
            const SignalType signal_type = uid_bool(mt) == 0 ? SignalType::EXCITATORY : SignalType::INHIBITORY;

            ASSERT_THROW(synaptic_elements.update_cnt(neuron_id, cnt), RelearnException);
            ASSERT_THROW(synaptic_elements.update_conn_cnt(neuron_id, conn_cnt), RelearnException);
            ASSERT_THROW(synaptic_elements.update_delta_cnt(neuron_id, delta_cnt), RelearnException);
            ASSERT_THROW(synaptic_elements.set_signal_type(neuron_id, signal_type), RelearnException);
        }

        const std::vector<double>& cnts = synaptic_elements.get_cnts();
        const std::vector<unsigned int>& conn_cnts = synaptic_elements.get_connected_cnts();
        const std::vector<double>& delta_cnts = synaptic_elements.get_delta_cnts();
        const std::vector<SignalType>& types = synaptic_elements.get_signal_types();

        ASSERT_EQ(cnts.size(), num_neurons);
        ASSERT_EQ(conn_cnts.size(), num_neurons);
        ASSERT_EQ(delta_cnts.size(), num_neurons);
        ASSERT_EQ(types.size(), num_neurons);

        for (size_t neuron_id = 0; neuron_id < num_neurons; neuron_id++) {
            const auto a1 = golden_cnts[neuron_id];
            const auto a2 = synaptic_elements.get_cnt(neuron_id);
            const auto a3 = cnts[neuron_id];

            const auto a_is_correct = a1 == a2 && a1 == a3;
            ASSERT_TRUE(a_is_correct);

            const auto b1 = golden_conn_cnts[neuron_id];
            const auto b2 = synaptic_elements.get_connected_cnt(neuron_id);
            const auto b3 = conn_cnts[neuron_id];

            const auto b_is_correct = b1 == b2 && b1 == b3;
            ASSERT_TRUE(b_is_correct);

            const auto c1 = golden_delta_cnts[neuron_id];
            const auto c2 = synaptic_elements.get_delta_cnt(neuron_id);
            const auto c3 = delta_cnts[neuron_id];

            const auto c_is_correct = c1 == c2 && c1 == c3;
            ASSERT_TRUE(c_is_correct);

            const auto d1 = golden_signal_types[neuron_id];
            const auto d2 = synaptic_elements.get_signal_type(neuron_id);
            const auto d3 = types[neuron_id];

            const auto d_is_correct = d1 == d2 && d1 == d3;
            ASSERT_TRUE(d_is_correct);
        }

        for (size_t neuron_id = num_neurons; neuron_id < num_neurons + 10; neuron_id++) {
            ASSERT_THROW(auto val = synaptic_elements.get_cnt(neuron_id), RelearnException);
            ASSERT_THROW(auto val = synaptic_elements.get_connected_cnt(neuron_id), RelearnException);
            ASSERT_THROW(auto val = synaptic_elements.get_delta_cnt(neuron_id), RelearnException);
            ASSERT_THROW(auto val = synaptic_elements.get_signal_type(neuron_id), RelearnException);
        }
    }
}

TEST_F(RelearnTest, testSynapticElementsParameters) {
    std::uniform_int_distribution<size_t> uid_num_neurons(0, upper_bound_num_neurons);
    std::uniform_int_distribution<int> uid_bool(0, 1);

    std::uniform_real_distribution<double> urd_C;

    for (auto i = 0; i < iterations; i++) {
        const size_t num_neurons = uid_num_neurons(mt);
        const double C = urd_C(mt);

        ElementType element_type = uid_bool(mt) == 0 ? ElementType::AXON : ElementType::DENDRITE;

        SynapticElements synaptic_elements(element_type, C);
        synaptic_elements.init(num_neurons);

        std::vector<ModelParameter> parameters = synaptic_elements.get_parameter();

        Parameter<double> param_min_C = std::get<Parameter<double>>(parameters[0]);
        Parameter<double> param_target_C = std::get<Parameter<double>>(parameters[1]);
        Parameter<double> param_nu = std::get<Parameter<double>>(parameters[2]);
        Parameter<double> param_vacant = std::get<Parameter<double>>(parameters[3]);

        ASSERT_EQ(param_min_C.min(), SynapticElements::min_min_C_level_to_grow);
        ASSERT_EQ(param_min_C.value(), C);
        ASSERT_EQ(param_min_C.max(), SynapticElements::max_min_C_level_to_grow);

        ASSERT_EQ(param_target_C.min(), SynapticElements::min_C_target);
        ASSERT_EQ(param_target_C.value(), SynapticElements::default_C_target);
        ASSERT_EQ(param_target_C.max(), SynapticElements::max_C_target);

        ASSERT_EQ(param_nu.min(), SynapticElements::min_nu);
        ASSERT_EQ(param_nu.value(), SynapticElements::default_nu);
        ASSERT_EQ(param_nu.max(), SynapticElements::max_nu);

        ASSERT_EQ(param_vacant.min(), SynapticElements::min_vacant_retract_ratio);
        ASSERT_EQ(param_vacant.value(), SynapticElements::default_vacant_retract_ratio);
        ASSERT_EQ(param_vacant.max(), SynapticElements::max_vacant_retract_ratio);

        const double d1 = urd_C(mt);
        const double d2 = urd_C(mt);
        const double d3 = urd_C(mt);
        const double d4 = urd_C(mt);

        param_min_C.value() = d1;
        param_target_C.value() = d2;
        param_nu.value() = d3;
        param_vacant.value() = d4;

        ASSERT_EQ(param_min_C.value(), d1);
        ASSERT_EQ(param_target_C.value(), d2);
        ASSERT_EQ(param_nu.value(), d3);
        ASSERT_EQ(param_vacant.value(), d4);
    }
}

TEST_F(RelearnTest, testSynapticElementsUpdate) {
    std::uniform_int_distribution<size_t> uid_num_neurons(0, upper_bound_num_neurons);
    std::uniform_int_distribution<int> uid_bool(0, 1);

    std::uniform_real_distribution<double> urd_cnt(0, 10);
    std::uniform_int_distribution<size_t> uid_connected(0, 10);
    std::uniform_real_distribution<double> urd_delta(0, 10);

    for (auto i = 0; i < iterations; i++) {
        size_t num_neurons = uid_num_neurons(mt);

        ElementType element_type = uid_bool(mt) == 0 ? ElementType::AXON : ElementType::DENDRITE;

        SynapticElements synaptic_elements(element_type, 0.0);
        synaptic_elements.init(num_neurons);

        std::vector<double> golden_cnts(num_neurons);
        std::vector<double> golden_conn_cnts(num_neurons);
        std::vector<double> golden_delta_cnts(num_neurons);
        std::vector<SignalType> golden_signal_types(num_neurons);

        for (size_t neuron_id = 0; neuron_id < num_neurons; neuron_id++) {
            const double cnt = urd_cnt(mt);
            const double conn_cnt = uid_connected(mt);
            const double delta_cnt = urd_delta(mt);
            const SignalType signal_type = uid_bool(mt) == 0 ? SignalType::EXCITATORY : SignalType::INHIBITORY;

            golden_cnts[neuron_id] = cnt;
            golden_conn_cnts[neuron_id] = conn_cnt;
            golden_delta_cnts[neuron_id] = delta_cnt;
            golden_signal_types[neuron_id] = signal_type;

            synaptic_elements.update_cnt(neuron_id, cnt);
            synaptic_elements.update_conn_cnt(neuron_id, conn_cnt);
            synaptic_elements.update_delta_cnt(neuron_id, delta_cnt);
            synaptic_elements.set_signal_type(neuron_id, signal_type);
        }

        for (size_t neuron_id = num_neurons; neuron_id < num_neurons + 10; neuron_id++) {
            const double cnt = urd_cnt(mt);
            const double conn_cnt = uid_connected(mt);
            const double delta_cnt = urd_delta(mt);
            const SignalType signal_type = uid_bool(mt) == 0 ? SignalType::EXCITATORY : SignalType::INHIBITORY;

            ASSERT_THROW(synaptic_elements.update_cnt(neuron_id, cnt), RelearnException);
            ASSERT_THROW(synaptic_elements.update_conn_cnt(neuron_id, conn_cnt), RelearnException);
            ASSERT_THROW(synaptic_elements.update_delta_cnt(neuron_id, delta_cnt), RelearnException);
            ASSERT_THROW(synaptic_elements.set_signal_type(neuron_id, signal_type), RelearnException);
        }

        const std::vector<double>& cnts = synaptic_elements.get_cnts();
        const std::vector<unsigned int>& conn_cnts = synaptic_elements.get_connected_cnts();
        const std::vector<double>& delta_cnts = synaptic_elements.get_delta_cnts();
        const std::vector<SignalType>& types = synaptic_elements.get_signal_types();

        ASSERT_EQ(cnts.size(), num_neurons);
        ASSERT_EQ(conn_cnts.size(), num_neurons);
        ASSERT_EQ(delta_cnts.size(), num_neurons);
        ASSERT_EQ(types.size(), num_neurons);

        for (size_t neuron_id = 0; neuron_id < num_neurons; neuron_id++) {
            ASSERT_EQ(golden_cnts[neuron_id], synaptic_elements.get_cnt(neuron_id));
            ASSERT_EQ(golden_cnts[neuron_id], cnts[neuron_id]);

            ASSERT_EQ(golden_conn_cnts[neuron_id], synaptic_elements.get_connected_cnt(neuron_id));
            ASSERT_EQ(golden_conn_cnts[neuron_id], conn_cnts[neuron_id]);

            ASSERT_EQ(golden_delta_cnts[neuron_id], synaptic_elements.get_delta_cnt(neuron_id));
            ASSERT_EQ(golden_delta_cnts[neuron_id], delta_cnts[neuron_id]);

            ASSERT_EQ(golden_signal_types[neuron_id], synaptic_elements.get_signal_type(neuron_id));
            ASSERT_EQ(golden_signal_types[neuron_id], types[neuron_id]);
        }

        for (size_t neuron_id = num_neurons; neuron_id < num_neurons + 10; neuron_id++) {
            ASSERT_THROW(auto val = synaptic_elements.get_cnt(neuron_id), RelearnException);
            ASSERT_THROW(auto val = synaptic_elements.get_connected_cnt(neuron_id), RelearnException);
            ASSERT_THROW(auto val = synaptic_elements.get_delta_cnt(neuron_id), RelearnException);
            ASSERT_THROW(auto val = synaptic_elements.get_signal_type(neuron_id), RelearnException);
        }
    }
}

TEST_F(RelearnTest, testSynapticElementsMultipleUpdate) {
    std::uniform_int_distribution<size_t> uid_num_neurons(0, upper_bound_num_neurons);
    std::uniform_int_distribution<int> uid_bool(0, 1);

    std::uniform_real_distribution<double> urd_cnt(0, 10);
    std::uniform_int_distribution<size_t> uid_connected(0, 10);
    std::uniform_real_distribution<double> urd_delta(0, 10);

    for (auto i = 0; i < iterations; i++) {
        size_t num_neurons = uid_num_neurons(mt);

        ElementType element_type = uid_bool(mt) == 0 ? ElementType::AXON : ElementType::DENDRITE;

        SynapticElements synaptic_elements(element_type, 0.0);
        synaptic_elements.init(num_neurons);

        std::vector<double> golden_cnts(num_neurons, 0.0);
        std::vector<double> golden_conn_cnts(num_neurons, 0.0);
        std::vector<double> golden_delta_cnts(num_neurons, 0.0);
        std::vector<SignalType> golden_signal_types(num_neurons);

        for (size_t neuron_id = 0; neuron_id < num_neurons; neuron_id++) {
            const double cnt = urd_cnt(mt);
            const double conn_cnt = uid_connected(mt);
            const double delta_cnt = urd_delta(mt);
            const SignalType signal_type = uid_bool(mt) == 0 ? SignalType::EXCITATORY : SignalType::INHIBITORY;

            golden_cnts[neuron_id] += cnt;
            golden_conn_cnts[neuron_id] += conn_cnt;
            golden_delta_cnts[neuron_id] += delta_cnt;
            golden_signal_types[neuron_id] = signal_type;

            synaptic_elements.update_cnt(neuron_id, cnt);
            synaptic_elements.update_conn_cnt(neuron_id, conn_cnt);
            synaptic_elements.update_delta_cnt(neuron_id, delta_cnt);
            synaptic_elements.set_signal_type(neuron_id, signal_type);
        }

        const std::vector<double>& cnts = synaptic_elements.get_cnts();
        const std::vector<unsigned int>& conn_cnts = synaptic_elements.get_connected_cnts();
        const std::vector<double>& delta_cnts = synaptic_elements.get_delta_cnts();
        const std::vector<SignalType>& types = synaptic_elements.get_signal_types();

        ASSERT_EQ(cnts.size(), num_neurons);
        ASSERT_EQ(conn_cnts.size(), num_neurons);
        ASSERT_EQ(delta_cnts.size(), num_neurons);
        ASSERT_EQ(types.size(), num_neurons);

        for (size_t neuron_id = 0; neuron_id < num_neurons; neuron_id++) {
            ASSERT_EQ(golden_cnts[neuron_id], synaptic_elements.get_cnt(neuron_id));
            ASSERT_EQ(golden_cnts[neuron_id], cnts[neuron_id]);

            ASSERT_EQ(golden_conn_cnts[neuron_id], synaptic_elements.get_connected_cnt(neuron_id));
            ASSERT_EQ(golden_conn_cnts[neuron_id], conn_cnts[neuron_id]);

            ASSERT_EQ(golden_delta_cnts[neuron_id], synaptic_elements.get_delta_cnt(neuron_id));
            ASSERT_EQ(golden_delta_cnts[neuron_id], delta_cnts[neuron_id]);

            ASSERT_EQ(golden_signal_types[neuron_id], synaptic_elements.get_signal_type(neuron_id));
            ASSERT_EQ(golden_signal_types[neuron_id], types[neuron_id]);
        }
    }
}

}
