#include "../googletest/include/gtest/gtest.h"

#include "RelearnTest.hpp"

#include "../source/neurons/models/SynapticElements.h"

#include <sstream>

TEST_F(SynapticElementsTest, testSynapticElementsConstructor) {
    for (auto i = 0; i < iterations; i++) {
        const auto& calcium_to_grow = get_random_double(SynapticElements::min_min_C_level_to_grow, SynapticElements::max_min_C_level_to_grow);
        const auto& nu = get_random_double(SynapticElements::min_nu, SynapticElements::max_nu);
        const auto& retract_ratio = get_random_double(SynapticElements::min_vacant_retract_ratio, SynapticElements::max_vacant_retract_ratio);
        const auto& vacant_elements_lb = get_random_double(SynapticElements::min_vacant_elements_initially, SynapticElements::max_vacant_elements_initially);
        const auto& vacant_elements_ub = get_random_double(SynapticElements::min_vacant_elements_initially, SynapticElements::max_vacant_elements_initially);

        const auto& element_type = get_random_element_type();

        std::stringstream ss{};
        ss << element_type << ' ';
        ss << calcium_to_grow << ' ';
        ss << nu << ' ';
        ss << retract_ratio << ' ';
        ss << vacant_elements_lb << ' ';
        ss << vacant_elements_ub << '\n';

        SynapticElements synaptic_elements(element_type, calcium_to_grow, nu, retract_ratio, vacant_elements_lb, vacant_elements_lb);

        const auto& parameters = synaptic_elements.get_parameter();

        Parameter<double> param_min_C = std::get<Parameter<double>>(parameters[0]);
        Parameter<double> param_nu = std::get<Parameter<double>>(parameters[1]);
        Parameter<double> param_vacant = std::get<Parameter<double>>(parameters[2]);
        Parameter<double> param_lower_bound = std::get<Parameter<double>>(parameters[3]);
        Parameter<double> param_upper_bound = std::get<Parameter<double>>(parameters[4]);

        ASSERT_EQ(param_min_C.value(), calcium_to_grow) << ss.str();
        ASSERT_EQ(param_nu.value(), nu) << ss.str();
        ASSERT_EQ(param_vacant.value(), retract_ratio) << ss.str();
        ASSERT_EQ(param_lower_bound.value(), vacant_elements_lb) << ss.str();
        ASSERT_EQ(param_upper_bound.value(), vacant_elements_ub) << ss.str();
    }
}

TEST_F(SynapticElementsTest, testSynapticElementsInitialize) {
    for (auto i = 0; i < iterations; i++) {
        const auto& number_neurons = get_random_number_neurons();
        const auto& element_type = get_random_element_type();

        std::stringstream ss{};
        ss << number_neurons << ' ' << element_type << '\n';

        SynapticElements synaptic_elements(element_type, 0.0);
        synaptic_elements.init(number_neurons);

        for (auto neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
            const auto& grown_element = synaptic_elements.get_count(neuron_id);
            const auto& connected_grown_element = synaptic_elements.get_connected_count(neuron_id);
            const auto& delta_grown_element = synaptic_elements.get_delta_count(neuron_id);

            ASSERT_EQ(grown_element, 0.0) << ss.str() << neuron_id;
            ASSERT_EQ(connected_grown_element, 0.0) << ss.str() << neuron_id;
            ASSERT_EQ(delta_grown_element, 0.0) << ss.str() << neuron_id;
        }

        const auto& grown_elements = synaptic_elements.get_total_counts();
        const auto& connected_grown_elements = synaptic_elements.get_total_counts();
        const auto& delta_grown_elements = synaptic_elements.get_total_counts();
        const auto& signal_types = synaptic_elements.get_signal_types();

        ASSERT_EQ(grown_elements.size(), number_neurons) << ss.str();
        ASSERT_EQ(connected_grown_elements.size(), number_neurons) << ss.str();
        ASSERT_EQ(delta_grown_elements.size(), number_neurons) << ss.str();
        ASSERT_EQ(signal_types.size(), number_neurons) << ss.str();

        for (const auto& grown_element : grown_elements) {
            ASSERT_EQ(grown_element, 0.0) << ss.str();
        }

        for (const auto& connected_grown_element : connected_grown_elements) {
            ASSERT_EQ(connected_grown_element, 0.0) << ss.str();
        }

        for (const auto& delta_grown_element : delta_grown_elements) {
            ASSERT_EQ(delta_grown_element, 0.0) << ss.str();
        }
    }
}

TEST_F(SynapticElementsTest, testSynapticElementsInitializeException) {
    for (auto i = 0; i < iterations; i++) {
        const auto& number_neurons = get_random_number_neurons();
        const auto& element_type = get_random_element_type();

        std::stringstream ss{};
        ss << number_neurons << ' ' << element_type << '\n';

        SynapticElements synaptic_elements(element_type, 0.0);
        synaptic_elements.init(number_neurons);

        std::vector<double> golden_cnts(number_neurons);
        std::vector<unsigned int> golden_conn_cnts(number_neurons);
        std::vector<double> golden_delta_cnts(number_neurons);
        std::vector<SignalType> golden_signal_types(number_neurons);

        for (auto neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
            const auto& grown_element = get_random_percentage();
            const auto& connected_grown_element = get_random_percentage();
            const auto& delta_grown_element = get_random_percentage();
            const auto& signal_type = get_random_signal_type();

            golden_cnts[neuron_id] = grown_element;
            golden_conn_cnts[neuron_id] = static_cast<unsigned int>(connected_grown_element);
            golden_signal_types[neuron_id] = signal_type;

            synaptic_elements.update_count(neuron_id, grown_element);
            synaptic_elements.update_connected_counts(neuron_id, static_cast<int>(connected_grown_element));
            synaptic_elements.set_signal_type(neuron_id, signal_type);
        }

        for (auto neuron_id = number_neurons; neuron_id < number_neurons + 10; neuron_id++) {
            const auto& grown_element = get_random_percentage();
            const auto& connected_grown_element = static_cast<unsigned int>(get_random_percentage());
            const auto& signal_type = get_random_signal_type();

            ASSERT_THROW(synaptic_elements.update_count(neuron_id, grown_element), RelearnException) << ss.str() << neuron_id;
            ASSERT_THROW(synaptic_elements.update_connected_counts(neuron_id, connected_grown_element), RelearnException) << ss.str() << neuron_id;
            ASSERT_THROW(synaptic_elements.set_signal_type(neuron_id, signal_type), RelearnException) << ss.str() << neuron_id;
        }

        const auto& grown_elements = synaptic_elements.get_total_counts();
        const auto& connected_grown_elements = synaptic_elements.get_connected_count();
        const auto& delta_grown_elements = synaptic_elements.get_delta_counts();
        const auto& signal_types = synaptic_elements.get_signal_types();

        ASSERT_EQ(grown_elements.size(), number_neurons) << ss.str();
        ASSERT_EQ(connected_grown_elements.size(), number_neurons) << ss.str();
        ASSERT_EQ(delta_grown_elements.size(), number_neurons) << ss.str();
        ASSERT_EQ(signal_types.size(), number_neurons) << ss.str();

        for (auto neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
            const auto& a1 = golden_cnts[neuron_id];
            const auto& a2 = synaptic_elements.get_count(neuron_id);
            const auto& a3 = grown_elements[neuron_id];

            const auto& a_is_correct = a1 == a2 && a1 == a3;
            ASSERT_TRUE(a_is_correct) << ss.str() << neuron_id;

            const auto& b1 = golden_conn_cnts[neuron_id];
            const auto& b2 = synaptic_elements.get_connected_count(neuron_id);
            const auto& b3 = connected_grown_elements[neuron_id];

            const auto& b_is_correct = b1 == b2 && b1 == b3;
            ASSERT_TRUE(b_is_correct) << ss.str() << neuron_id;

            const auto& c1 = golden_delta_cnts[neuron_id];
            const auto& c2 = synaptic_elements.get_delta_count(neuron_id);
            const auto& c3 = delta_grown_elements[neuron_id];

            const auto& c_is_correct = c1 == c2 && c1 == c3;
            ASSERT_TRUE(c_is_correct) << ss.str() << neuron_id;

            const auto& d1 = golden_signal_types[neuron_id];
            const auto& d2 = synaptic_elements.get_signal_type(neuron_id);
            const auto& d3 = signal_types[neuron_id];

            const auto& d_is_correct = d1 == d2 && d1 == d3;
            ASSERT_TRUE(d_is_correct) << ss.str() << neuron_id;
        }

        for (auto neuron_id = number_neurons; neuron_id < number_neurons + 10; neuron_id++) {
            ASSERT_THROW(auto val = synaptic_elements.get_count(neuron_id), RelearnException) << ss.str() << neuron_id;
            ASSERT_THROW(auto val = synaptic_elements.get_connected_count(neuron_id), RelearnException) << ss.str() << neuron_id;
            ASSERT_THROW(auto val = synaptic_elements.get_delta_count(neuron_id), RelearnException) << ss.str() << neuron_id;
            ASSERT_THROW(auto val = synaptic_elements.get_signal_type(neuron_id), RelearnException) << ss.str() << neuron_id;
        }
    }
}

TEST_F(SynapticElementsTest, testSynapticElementsParameters) {
    for (auto i = 0; i < iterations; i++) {
        const auto& number_neurons = get_random_number_neurons();
        const auto& C = get_random_percentage();

        const auto& element_type = get_random_element_type();

        std::stringstream ss{};
        ss << number_neurons << ' ' << element_type << ' ' << C << '\n';

        SynapticElements synaptic_elements(element_type, C);
        synaptic_elements.init(number_neurons);

        const auto& parameters = synaptic_elements.get_parameter();

        ASSERT_EQ(parameters.size(), 5) << ss.str();

        Parameter<double> param_min_C = std::get<Parameter<double>>(parameters[0]);
        Parameter<double> param_nu = std::get<Parameter<double>>(parameters[1]);
        Parameter<double> param_vacant = std::get<Parameter<double>>(parameters[2]);
        Parameter<double> param_lower_bound = std::get<Parameter<double>>(parameters[3]);
        Parameter<double> param_upper_bound = std::get<Parameter<double>>(parameters[4]);

        ASSERT_EQ(param_min_C.min(), SynapticElements::min_min_C_level_to_grow) << ss.str();
        ASSERT_EQ(param_min_C.value(), C) << ss.str();
        ASSERT_EQ(param_min_C.max(), SynapticElements::max_min_C_level_to_grow) << ss.str();

        ASSERT_EQ(param_nu.min(), SynapticElements::min_nu) << ss.str();
        ASSERT_EQ(param_nu.value(), SynapticElements::default_nu) << ss.str();
        ASSERT_EQ(param_nu.max(), SynapticElements::max_nu) << ss.str();

        ASSERT_EQ(param_vacant.min(), SynapticElements::min_vacant_retract_ratio) << ss.str();
        ASSERT_EQ(param_vacant.value(), SynapticElements::default_vacant_retract_ratio) << ss.str();
        ASSERT_EQ(param_vacant.max(), SynapticElements::max_vacant_retract_ratio) << ss.str();

        ASSERT_EQ(param_lower_bound.min(), SynapticElements::min_vacant_elements_initially) << ss.str();
        ASSERT_EQ(param_lower_bound.value(), SynapticElements::default_vacant_elements_initially_lower_bound) << ss.str();
        ASSERT_EQ(param_lower_bound.max(), SynapticElements::max_vacant_elements_initially) << ss.str();

        ASSERT_EQ(param_upper_bound.min(), SynapticElements::min_vacant_elements_initially) << ss.str();
        ASSERT_EQ(param_upper_bound.value(), SynapticElements::default_vacant_elements_initially_upper_bound) << ss.str();
        ASSERT_EQ(param_upper_bound.max(), SynapticElements::max_vacant_elements_initially) << ss.str();

        const auto& d1 = get_random_double(SynapticElements::min_min_C_level_to_grow, SynapticElements::max_min_C_level_to_grow);
        const auto& d2 = get_random_double(SynapticElements::min_nu, SynapticElements::max_nu);
        const auto& d3 = get_random_double(SynapticElements::min_vacant_retract_ratio, SynapticElements::max_vacant_retract_ratio);
        const auto& d4 = get_random_double(SynapticElements::min_vacant_elements_initially, SynapticElements::max_vacant_elements_initially);
        const auto& d5 = get_random_double(SynapticElements::min_vacant_elements_initially, SynapticElements::max_vacant_elements_initially);

        param_min_C.set_value(d1);
        param_nu.set_value(d2);
        param_vacant.set_value(d3);
        param_lower_bound.set_value(d4);
        param_upper_bound.set_value(d5);

        ASSERT_EQ(param_min_C.value(), d1) << ss.str();
        ASSERT_EQ(param_nu.value(), d2) << ss.str();
        ASSERT_EQ(param_vacant.value(), d3) << ss.str();
        ASSERT_EQ(param_lower_bound.value(), d4) << ss.str();
        ASSERT_EQ(param_upper_bound.value(), d5) << ss.str();
    }
}

TEST_F(SynapticElementsTest, testSynapticElementsUpdate) {
    std::uniform_int_distribution<unsigned int> uid_connected(0, 10);

    for (auto i = 0; i < iterations; i++) {
        const auto& number_neurons = get_random_number_neurons();
        const auto& element_type = get_random_element_type();

        std::stringstream ss{};
        ss << number_neurons << ' ' << element_type << '\n';

        SynapticElements synaptic_elements(element_type, 0.0);
        synaptic_elements.init(number_neurons);

        std::vector<double> golden_cnts(number_neurons);
        std::vector<unsigned int> golden_conn_cnts(number_neurons);
        std::vector<double> golden_delta_cnts(number_neurons);
        std::vector<SignalType> golden_signal_types(number_neurons);

        for (auto neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
            const auto& grown_element = get_random_synaptic_element_count();
            const auto& connected_grown_element = get_random_synaptic_element_connected_count();
            const auto& delta_grown_element = get_random_synaptic_element_count();
            const auto& signal_type = get_random_signal_type();

            golden_cnts[neuron_id] = grown_element;
            golden_conn_cnts[neuron_id] = connected_grown_element;
            golden_delta_cnts[neuron_id] = delta_grown_element;
            golden_signal_types[neuron_id] = signal_type;

            synaptic_elements.update_count(neuron_id, grown_element);
            synaptic_elements.update_connected_counts(neuron_id, static_cast<int>(connected_grown_element));
            synaptic_elements.set_signal_type(neuron_id, signal_type);
        }

        for (auto neuron_id = number_neurons; neuron_id < number_neurons + 10; neuron_id++) {
            const auto& grown_element = get_random_synaptic_element_count();
            const auto& connected_grown_element = get_random_synaptic_element_connected_count();
            const auto& signal_type = get_random_signal_type();

            ASSERT_THROW(synaptic_elements.update_count(neuron_id, grown_element), RelearnException) << ss.str() << neuron_id;
            ASSERT_THROW(synaptic_elements.update_connected_counts(neuron_id, static_cast<int>(connected_grown_element)), RelearnException) << ss.str() << neuron_id;
            ASSERT_THROW(synaptic_elements.set_signal_type(neuron_id, signal_type), RelearnException) << ss.str() << neuron_id;
        }

        const auto& grown_elements = synaptic_elements.get_total_counts();
        const auto& connected_grown_elements = synaptic_elements.get_connected_count();
        const auto& delta_grown_elements = synaptic_elements.get_delta_counts();
        const auto& signal_types = synaptic_elements.get_signal_types();

        ASSERT_EQ(grown_elements.size(), number_neurons) << ss.str();
        ASSERT_EQ(connected_grown_elements.size(), number_neurons) << ss.str();
        ASSERT_EQ(delta_grown_elements.size(), number_neurons) << ss.str();
        ASSERT_EQ(signal_types.size(), number_neurons) << ss.str();

        for (auto neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
            ASSERT_EQ(golden_cnts[neuron_id], synaptic_elements.get_count(neuron_id)) << ss.str() << neuron_id;
            ASSERT_EQ(golden_cnts[neuron_id], grown_elements[neuron_id]) << ss.str() << neuron_id;

            ASSERT_EQ(golden_conn_cnts[neuron_id], synaptic_elements.get_connected_count(neuron_id)) << ss.str() << neuron_id;
            ASSERT_EQ(golden_conn_cnts[neuron_id], connected_grown_elements[neuron_id]) << ss.str() << neuron_id;

            ASSERT_EQ(golden_signal_types[neuron_id], synaptic_elements.get_signal_type(neuron_id)) << ss.str() << neuron_id;
            ASSERT_EQ(golden_signal_types[neuron_id], signal_types[neuron_id]) << ss.str() << neuron_id;
        }

        for (auto neuron_id = number_neurons; neuron_id < number_neurons + 10; neuron_id++) {
            ASSERT_THROW(auto val = synaptic_elements.get_count(neuron_id), RelearnException) << ss.str() << neuron_id;
            ASSERT_THROW(auto val = synaptic_elements.get_connected_count(neuron_id), RelearnException) << ss.str() << neuron_id;
            ASSERT_THROW(auto val = synaptic_elements.get_delta_count(neuron_id), RelearnException) << ss.str() << neuron_id;
            ASSERT_THROW(auto val = synaptic_elements.get_signal_type(neuron_id), RelearnException) << ss.str() << neuron_id;
        }
    }
}

TEST_F(SynapticElementsTest, testSynapticElementsMultipleUpdate) {
    std::uniform_int_distribution<unsigned int> uid_connected(0, 10);

    for (auto i = 0; i < iterations; i++) {
        const auto& number_neurons = get_random_number_neurons();
        const auto& element_type = get_random_element_type();

        std::stringstream ss{};
        ss << number_neurons << ' ' << element_type << '\n';

        SynapticElements synaptic_elements(element_type, 0.0);
        synaptic_elements.init(number_neurons);

        std::vector<double> golden_cnts(number_neurons, 0.0);
        std::vector<unsigned int> golden_conn_cnts(number_neurons, 0);
        std::vector<double> golden_delta_cnts(number_neurons, 0.0);
        std::vector<SignalType> golden_signal_types(number_neurons);

        for (auto neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
            const auto& grown_element = get_random_synaptic_element_count();
            const auto& connected_grown_element = get_random_synaptic_element_connected_count();
            const auto& delta_grown_element = get_random_synaptic_element_count();
            const auto& signal_type = get_random_signal_type();

            golden_cnts[neuron_id] += grown_element;
            golden_conn_cnts[neuron_id] += connected_grown_element;
            golden_signal_types[neuron_id] = signal_type;

            synaptic_elements.update_count(neuron_id, grown_element);
            synaptic_elements.update_connected_counts(neuron_id, connected_grown_element);
            synaptic_elements.set_signal_type(neuron_id, signal_type);
        }

        const auto& grown_elements = synaptic_elements.get_total_counts();
        const auto& connected_grown_elements = synaptic_elements.get_connected_count();
        const auto& delta_grown_elements = synaptic_elements.get_delta_counts();
        const auto& signal_types = synaptic_elements.get_signal_types();

        ASSERT_EQ(grown_elements.size(), number_neurons) << ss.str();
        ASSERT_EQ(connected_grown_elements.size(), number_neurons) << ss.str();
        ASSERT_EQ(delta_grown_elements.size(), number_neurons) << ss.str();
        ASSERT_EQ(signal_types.size(), number_neurons) << ss.str();

        for (auto neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
            ASSERT_EQ(golden_cnts[neuron_id], synaptic_elements.get_count(neuron_id)) << ss.str() << neuron_id;
            ASSERT_EQ(golden_cnts[neuron_id], grown_elements[neuron_id]) << ss.str() << neuron_id;

            ASSERT_EQ(golden_conn_cnts[neuron_id], synaptic_elements.get_connected_count(neuron_id)) << ss.str() << neuron_id;
            ASSERT_EQ(golden_conn_cnts[neuron_id], connected_grown_elements[neuron_id]) << ss.str() << neuron_id;

            ASSERT_EQ(golden_signal_types[neuron_id], synaptic_elements.get_signal_type(neuron_id)) << ss.str() << neuron_id;
            ASSERT_EQ(golden_signal_types[neuron_id], signal_types[neuron_id]) << ss.str() << neuron_id;
        }
    }
}
