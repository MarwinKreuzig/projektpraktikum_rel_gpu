#include "gtest/gtest.h"

#include "RelearnTest.hpp"

#include "neurons/models/SynapticElements.h"
#include "neurons/UpdateStatus.h"

#include <numeric>
#include <sstream>

TEST_F(SynapticElementsTest, testGaussianGrowthCurve) {
    const auto intersection_1 = get_random_double(-100.0, 100.0);
    const auto intersection_2 = get_random_double(-100.0, 100.0);

    const auto left_intersection = std::min(intersection_1, intersection_2);
    const auto right_intersection = std::max(intersection_1, intersection_2);

    const auto middle = (left_intersection + right_intersection) / 2;

    const auto growth_factor = get_random_double(1e-6, 100.0);

    std::stringstream ss{};
    ss << "Left intersection: " << left_intersection << '\n'
       << "Right intersection: " << right_intersection << '\n'
       << "Maximum: " << growth_factor << '\n';

    ASSERT_NEAR(gaussian_growth_curve(left_intersection, left_intersection, right_intersection, growth_factor), 0.0, eps) << ss.str();
    ASSERT_NEAR(gaussian_growth_curve(right_intersection, left_intersection, right_intersection, growth_factor), 0.0, eps) << ss.str();
    ASSERT_NEAR(gaussian_growth_curve(middle, left_intersection, right_intersection, growth_factor), growth_factor, eps) << ss.str();

    std::vector<double> smaller_negatives(100);
    std::vector<double> smaller_positives(100);
    std::vector<double> larger_positives(100);
    std::vector<double> larger_negatives(100);

    for (auto i = 0; i < 100; i++) {
        smaller_negatives[i] = get_random_double(-100000.0, left_intersection);
        smaller_positives[i] = get_random_double(left_intersection, middle);
        larger_positives[i] = get_random_double(middle, right_intersection);
        larger_negatives[i] = get_random_double(right_intersection, 100000.0);
    }

    std::sort(smaller_negatives.begin(), smaller_negatives.end());
    std::sort(smaller_positives.begin(), smaller_positives.end());
    std::sort(larger_positives.begin(), larger_positives.end());
    std::sort(larger_negatives.begin(), larger_negatives.end());

    auto last_value = -100000.0;
    auto last_change = gaussian_growth_curve(last_value, left_intersection, right_intersection, growth_factor);

    for (auto i = 0; i < 100; i++) {
        const auto current_value = smaller_negatives[i];
        const auto current_change = gaussian_growth_curve(current_value, left_intersection, right_intersection, growth_factor);

        std::stringstream ss_loop{};
        ss_loop << "Current value: " << current_value << '\n';
        ss_loop << "Current change: " << current_change << '\n';
        ss_loop << "Last value: " << last_value << '\n';
        ss_loop << "Last change: " << last_change << '\n';

        ASSERT_LE(last_change, current_change) << ss.str() << ss_loop.str();
        ASSERT_LE(current_change, 0.0) << ss.str() << ss_loop.str();

        ss_loop.clear();

        last_value = current_value;
        last_change = current_change;
    }

    last_value = left_intersection;
    last_change = gaussian_growth_curve(last_value, left_intersection, right_intersection, growth_factor);

    for (auto i = 0; i < 100; i++) {
        const auto current_value = smaller_positives[i];
        const auto current_change = gaussian_growth_curve(current_value, left_intersection, right_intersection, growth_factor);

        std::stringstream ss_loop{};
        ss_loop << "Current value: " << current_value << '\n';
        ss_loop << "Current change: " << current_change << '\n';
        ss_loop << "Last value: " << last_value << '\n';
        ss_loop << "Last change: " << last_change << '\n';

        ASSERT_LE(last_change, current_change) << ss.str() << ss_loop.str();
        ASSERT_GE(current_change, 0.0) << ss.str() << ss_loop.str();
        ASSERT_GE(growth_factor, current_change) << ss.str() << ss_loop.str();

        ss_loop.clear();

        last_value = current_value;
        last_change = current_change;
    }

    last_value = middle;
    last_change = gaussian_growth_curve(last_value, left_intersection, right_intersection, growth_factor);

    for (auto i = 0; i < 100; i++) {
        const auto current_value = larger_positives[i];
        const auto current_change = gaussian_growth_curve(current_value, left_intersection, right_intersection, growth_factor);

        std::stringstream ss_loop{};
        ss_loop << "Current value: " << current_value << '\n';
        ss_loop << "Current change: " << current_change << '\n';
        ss_loop << "Last value: " << last_value << '\n';
        ss_loop << "Last change: " << last_change << '\n';

        ASSERT_GE(last_change, current_change) << ss.str() << ss_loop.str();
        ASSERT_GE(current_change, 0.0) << ss.str() << ss_loop.str();
        ASSERT_LE(current_change, growth_factor) << ss.str() << ss_loop.str();

        ss_loop.clear();

        last_value = current_value;
        last_change = current_change;
    }

    last_value = right_intersection;
    last_change = gaussian_growth_curve(last_value, left_intersection, right_intersection, growth_factor);

    for (auto i = 0; i < 100; i++) {
        const auto current_value = larger_negatives[i];
        const auto current_change = gaussian_growth_curve(current_value, left_intersection, right_intersection, growth_factor);

        std::stringstream ss_loop{};
        ss_loop << "Current value: " << current_value << '\n';
        ss_loop << "Current change: " << current_change << '\n';
        ss_loop << "Last value: " << last_value << '\n';
        ss_loop << "Last change: " << last_change << '\n';

        ASSERT_LE(current_change, last_change) << ss.str() << ss_loop.str();
        ASSERT_LE(current_change, 0.0) << ss.str() << ss_loop.str();

        ss_loop.clear();

        last_value = current_value;
        last_change = current_change;
    }
}

TEST_F(SynapticElementsTest, testSynapticElementsConstructor) {
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

    SynapticElements synaptic_elements(element_type, calcium_to_grow, nu, retract_ratio, vacant_elements_lb, vacant_elements_ub);

    const auto& parameters = synaptic_elements.get_parameter();

    Parameter<double> param_min_C = std::get<Parameter<double>>(parameters[0]);
    Parameter<double> param_nu = std::get<Parameter<double>>(parameters[1]);
    Parameter<double> param_vacant = std::get<Parameter<double>>(parameters[2]);
    Parameter<double> param_lower_bound = std::get<Parameter<double>>(parameters[3]);
    Parameter<double> param_upper_bound = std::get<Parameter<double>>(parameters[4]);

    ASSERT_EQ(element_type, synaptic_elements.get_element_type()) << ss.str();

    ASSERT_EQ(param_min_C.value(), calcium_to_grow) << ss.str();
    ASSERT_EQ(param_nu.value(), nu) << ss.str();
    ASSERT_EQ(param_vacant.value(), retract_ratio) << ss.str();
    ASSERT_EQ(param_lower_bound.value(), vacant_elements_lb) << ss.str();
    ASSERT_EQ(param_upper_bound.value(), vacant_elements_ub) << ss.str();
}

TEST_F(SynapticElementsTest, testSynapticElementsParameters) {
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

TEST_F(SynapticElementsTest, testSynapticElementsClone) {
    const auto& number_neurons = get_random_number_neurons();
    const auto& C = get_random_percentage();

    const auto& element_type = get_random_element_type();

    std::stringstream ss{};
    ss << number_neurons << ' ' << element_type << ' ' << C << '\n';

    SynapticElements synaptic_elements(element_type, C);
    synaptic_elements.init(number_neurons);

    const auto& parameters = synaptic_elements.get_parameter();

    Parameter<double> param_min_C = std::get<Parameter<double>>(parameters[0]);
    Parameter<double> param_nu = std::get<Parameter<double>>(parameters[1]);
    Parameter<double> param_vacant = std::get<Parameter<double>>(parameters[2]);
    Parameter<double> param_lower_bound = std::get<Parameter<double>>(parameters[3]);
    Parameter<double> param_upper_bound = std::get<Parameter<double>>(parameters[4]);

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

    auto cloned_synaptic_elements_ptr = synaptic_elements.clone();
    auto& cloned_synaptic_elements = *cloned_synaptic_elements_ptr;

    const auto& cloned_parameters = cloned_synaptic_elements.get_parameter();

    Parameter<double> cloned_param_min_C = std::get<Parameter<double>>(cloned_parameters[0]);
    Parameter<double> cloned_param_nu = std::get<Parameter<double>>(cloned_parameters[1]);
    Parameter<double> cloned_param_vacant = std::get<Parameter<double>>(cloned_parameters[2]);
    Parameter<double> cloned_param_lower_bound = std::get<Parameter<double>>(cloned_parameters[3]);
    Parameter<double> cloned_param_upper_bound = std::get<Parameter<double>>(cloned_parameters[4]);

    ASSERT_EQ(param_min_C.value(), d1) << ss.str();
    ASSERT_EQ(param_nu.value(), d2) << ss.str();
    ASSERT_EQ(param_vacant.value(), d3) << ss.str();
    ASSERT_EQ(param_lower_bound.value(), d4) << ss.str();
    ASSERT_EQ(param_upper_bound.value(), d5) << ss.str();

    ASSERT_EQ(cloned_param_min_C.value(), d1) << ss.str();
    ASSERT_EQ(cloned_param_nu.value(), d2) << ss.str();
    ASSERT_EQ(cloned_param_vacant.value(), d3) << ss.str();
    ASSERT_EQ(cloned_param_lower_bound.value(), d4) << ss.str();
    ASSERT_EQ(cloned_param_upper_bound.value(), d5) << ss.str();
}

TEST_F(SynapticElementsTest, testSynapticElementsInitialize) {
    const auto& number_neurons = get_random_number_neurons();
    const auto& element_type = get_random_element_type();

    std::stringstream ss{};
    ss << number_neurons << ' ' << element_type << '\n';

    SynapticElements synaptic_elements(element_type, 0.0);
    synaptic_elements.init(number_neurons);

    ASSERT_EQ(synaptic_elements.get_size(), number_neurons);

    for (const auto& neuron_id : NeuronID::range(number_neurons)) {
        const auto& grown_element = synaptic_elements.get_grown_elements(neuron_id);
        const auto& connected_grown_element = synaptic_elements.get_connected_elements(neuron_id);
        const auto& delta_grown_element = synaptic_elements.get_delta(neuron_id);

        ASSERT_EQ(grown_element, 0.0) << ss.str() << neuron_id;
        ASSERT_EQ(connected_grown_element, 0.0) << ss.str() << neuron_id;
        ASSERT_EQ(delta_grown_element, 0.0) << ss.str() << neuron_id;
    }

    for (auto iteration = 0; iteration < number_neurons_out_of_scope; iteration++) {
        const auto neuron_id = get_random_neuron_id(number_neurons, number_neurons);

        ASSERT_THROW(auto ret = synaptic_elements.get_grown_elements(neuron_id), RelearnException) << ss.str() << neuron_id;
        ASSERT_THROW(auto ret = synaptic_elements.get_connected_elements(neuron_id), RelearnException) << ss.str() << neuron_id;
        ASSERT_THROW(auto ret = synaptic_elements.get_delta(neuron_id), RelearnException) << ss.str() << neuron_id;
    }

    const auto& grown_elements = synaptic_elements.get_grown_elements();
    const auto& connected_grown_elements = synaptic_elements.get_grown_elements();
    const auto& delta_grown_elements = synaptic_elements.get_grown_elements();
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

TEST_F(SynapticElementsTest, testSynapticElementsCreateNeurons) {
    const auto& number_neurons_initially = get_random_number_neurons();
    const auto& number_neurons_added = get_random_number_neurons();
    const auto& element_type = get_random_element_type();

    std::stringstream ss{};
    ss << number_neurons_initially << ' ' << number_neurons_added << ' ' << element_type << '\n';

    SynapticElements synaptic_elements(element_type, 0.0);
    synaptic_elements.init(number_neurons_initially);
    synaptic_elements.create_neurons(number_neurons_added);

    const auto& number_neurons = number_neurons_initially + number_neurons_added;

    ASSERT_EQ(synaptic_elements.get_size(), number_neurons);

    for (auto neuron_id : NeuronID::range(number_neurons)) {
        const auto& grown_element = synaptic_elements.get_grown_elements(neuron_id);
        const auto& connected_grown_element = synaptic_elements.get_connected_elements(neuron_id);
        const auto& delta_grown_element = synaptic_elements.get_delta(neuron_id);

        ASSERT_EQ(grown_element, 0.0) << ss.str() << neuron_id;
        ASSERT_EQ(connected_grown_element, 0.0) << ss.str() << neuron_id;
        ASSERT_EQ(delta_grown_element, 0.0) << ss.str() << neuron_id;
    }

    for (auto iteration = 0; iteration < number_neurons_out_of_scope; iteration++) {
        const auto neuron_id = get_random_neuron_id(number_neurons, number_neurons);

        ASSERT_THROW(auto ret = synaptic_elements.get_grown_elements(neuron_id), RelearnException) << ss.str() << neuron_id;
        ASSERT_THROW(auto ret = synaptic_elements.get_connected_elements(neuron_id), RelearnException) << ss.str() << neuron_id;
        ASSERT_THROW(auto ret = synaptic_elements.get_delta(neuron_id), RelearnException) << ss.str() << neuron_id;
    }

    const auto& grown_elements = synaptic_elements.get_grown_elements();
    const auto& connected_grown_elements = synaptic_elements.get_grown_elements();
    const auto& delta_grown_elements = synaptic_elements.get_grown_elements();
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

TEST_F(SynapticElementsTest, testSynapticElementsInitialElementsConstant) {
    const auto& number_neurons_initially = get_random_number_neurons();
    const auto& number_neurons_added = get_random_number_neurons();
    const auto& element_type = get_random_element_type();

    const auto& number_neurons = number_neurons_initially + number_neurons_added;

    std::stringstream ss{};
    ss << number_neurons_initially << ' ' << number_neurons_added << ' ' << element_type << '\n';

    const auto& min_c = get_random_percentage();
    const auto& nu = get_random_percentage();
    const auto& retract_ratio = get_random_percentage();

    const auto& bound = get_random_double(0.0, 10.0);

    ss << min_c << ' ' << nu << ' ' << retract_ratio << ' ' << bound << '\n';

    SynapticElements synaptic_elements(element_type, min_c, nu, retract_ratio, bound, bound);

    synaptic_elements.init(number_neurons_initially);
    synaptic_elements.create_neurons(number_neurons_added);

    const auto& counts = synaptic_elements.get_grown_elements();

    for (auto neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
        const auto grown_elements_1 = synaptic_elements.get_grown_elements(NeuronID{ neuron_id });
        const auto grown_elements_2 = counts[neuron_id];

        ASSERT_EQ(grown_elements_1, grown_elements_2) << ss.str() << neuron_id;
        ASSERT_EQ(grown_elements_1, bound) << ss.str() << neuron_id;
    }
}

TEST_F(SynapticElementsTest, testSynapticElementsInitialElements) {
    const auto& number_neurons_initially = get_random_number_neurons();
    const auto& number_neurons_added = get_random_number_neurons();
    const auto& element_type = get_random_element_type();

    const auto& number_neurons = number_neurons_initially + number_neurons_added;

    std::stringstream ss{};
    ss << number_neurons_initially << ' ' << number_neurons_added << ' ' << element_type << '\n';

    const auto& min_c = get_random_percentage();
    const auto& nu = get_random_percentage();
    const auto& retract_ratio = get_random_percentage();

    const auto& bound_1 = get_random_double(0.0, 10.0);
    const auto& bound_2 = get_random_double(0.0, 10.0);

    const auto& lower_bound = std::min(bound_1, bound_2);
    const auto& upper_bound = std::max(bound_1, bound_2) + 1.0;

    ss << min_c << ' ' << nu << ' ' << retract_ratio << ' ' << lower_bound << ' ' << upper_bound << '\n';

    SynapticElements synaptic_elements(element_type, min_c, nu, retract_ratio, lower_bound, upper_bound);

    synaptic_elements.init(number_neurons_initially);
    synaptic_elements.create_neurons(number_neurons_added);

    const auto& counts = synaptic_elements.get_grown_elements();

    for (auto neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
        const auto grown_elements_1 = synaptic_elements.get_grown_elements(NeuronID{ neuron_id });
        const auto grown_elements_2 = counts[neuron_id];

        ASSERT_EQ(grown_elements_1, grown_elements_2) << ss.str() << neuron_id;

        ASSERT_TRUE(lower_bound <= grown_elements_1) << ss.str() << neuron_id;
        ASSERT_TRUE(grown_elements_1 <= upper_bound) << ss.str() << neuron_id;
    }
}

TEST_F(SynapticElementsTest, testSynapticElementsInitialElementsException) {
    const auto& number_neurons_initially = get_random_number_neurons();
    const auto& number_neurons_added = get_random_number_neurons();
    const auto& element_type = get_random_element_type();

    std::stringstream ss{};
    ss << number_neurons_initially << ' ' << number_neurons_added << ' ' << element_type << '\n';

    const auto& min_c = get_random_percentage();
    const auto& nu = get_random_percentage();
    const auto& retract_ratio = get_random_percentage();

    const auto& bound_1 = get_random_double(0.0, 10.0);
    const auto& bound_2 = get_random_double(0.0, 10.0);

    const auto& lower_bound = std::min(bound_1, bound_2);
    const auto& upper_bound = std::max(bound_1, bound_2) + 1.0;

    ss << min_c << ' ' << nu << ' ' << retract_ratio << ' ' << lower_bound << ' ' << upper_bound << '\n';

    SynapticElements synaptic_elements(element_type, min_c, nu, retract_ratio, upper_bound, lower_bound);

    ASSERT_THROW(synaptic_elements.init(number_neurons_initially), RelearnException) << ss.str();
    ASSERT_THROW(synaptic_elements.create_neurons(number_neurons_added), RelearnException) << ss.str();
}

TEST_F(SynapticElementsTest, testSynapticElementsInitialElementsMultipleBounds) {
    const auto& number_neurons_initially = get_random_number_neurons();
    const auto& number_neurons_added = get_random_number_neurons();
    const auto& element_type = get_random_element_type();

    const auto& number_neurons = number_neurons_initially + number_neurons_added;

    std::stringstream ss{};
    ss << number_neurons_initially << ' ' << number_neurons_added << ' ' << element_type << '\n';

    const auto& min_c = get_random_percentage();
    const auto& nu = get_random_percentage();
    const auto& retract_ratio = get_random_percentage();

    const auto& bound_1 = get_random_double(0.0, 10.0);
    const auto& bound_2 = get_random_double(0.0, 10.0);

    const auto& lower_bound_1 = std::min(bound_1, bound_2);
    const auto& upper_bound_1 = std::max(bound_1, bound_2) + 1.0;

    const auto& bound_3 = get_random_double(0.0, 10.0);
    const auto& bound_4 = get_random_double(0.0, 10.0);

    const auto& lower_bound_2 = std::min(bound_3, bound_4);
    const auto& upper_bound_2 = std::max(bound_3, bound_4) + 1.0;

    ss << min_c << ' ' << nu << ' ' << retract_ratio << ' ' << lower_bound_1 << ' ' << upper_bound_1 << ' ' << lower_bound_2 << ' ' << upper_bound_2 << '\n';

    SynapticElements synaptic_elements(element_type, min_c, nu, retract_ratio, lower_bound_1, upper_bound_1);
    synaptic_elements.init(number_neurons_initially);

    auto parameters = synaptic_elements.get_parameter();

    Parameter<double> param_lower_bound = std::get<Parameter<double>>(parameters[3]);
    Parameter<double> param_upper_bound = std::get<Parameter<double>>(parameters[4]);

    param_lower_bound.set_value(lower_bound_2);
    param_upper_bound.set_value(upper_bound_2);

    synaptic_elements.create_neurons(number_neurons_added);

    const auto& counts = synaptic_elements.get_grown_elements();

    for (auto neuron_id : NeuronID::range(number_neurons_initially)) {
        const auto grown_elements_1 = synaptic_elements.get_grown_elements(neuron_id);
        const auto grown_elements_2 = counts[neuron_id.get_neuron_id()];

        ASSERT_EQ(grown_elements_1, grown_elements_2) << ss.str() << neuron_id;

        ASSERT_TRUE(lower_bound_1 <= grown_elements_1) << ss.str() << neuron_id;
        ASSERT_TRUE(grown_elements_1 <= upper_bound_1) << ss.str() << neuron_id;
    }

    for (auto neuron_id = number_neurons_initially; neuron_id < number_neurons; neuron_id++) {
        const auto id = NeuronID{ neuron_id };
        const auto grown_elements_1 = synaptic_elements.get_grown_elements(id);
        const auto grown_elements_2 = counts[neuron_id];

        ASSERT_EQ(grown_elements_1, grown_elements_2) << ss.str() << id;

        ASSERT_TRUE(lower_bound_2 <= grown_elements_1) << ss.str() << id;
        ASSERT_TRUE(grown_elements_1 <= upper_bound_2) << ss.str() << id;
    }
}

TEST_F(SynapticElementsTest, testSynapticElementsSignalTypes) {
    const auto& number_neurons = get_random_number_neurons();
    const auto& element_type = get_random_element_type();

    std::stringstream ss{};
    ss << number_neurons << ' ' << element_type << '\n';

    SynapticElements synaptic_elements(element_type, 0.0);
    synaptic_elements.init(number_neurons);

    std::vector<SignalType> signal_types(number_neurons);
    std::vector<SignalType> golden_signal_types(number_neurons);

    for (auto neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
        const auto& signal_type = get_random_signal_type();

        signal_types[neuron_id] = signal_type;
        golden_signal_types[neuron_id] = signal_type;
    }

    synaptic_elements.set_signal_types(std::move(signal_types));

    for (auto neuron_id : NeuronID::range(number_neurons)) {
        ASSERT_EQ(synaptic_elements.get_signal_type(neuron_id), golden_signal_types[neuron_id.get_neuron_id()]) << ss.str() << neuron_id;
    }
}

TEST_F(SynapticElementsTest, testSynapticElementsSingleUpdate) {
    const auto& number_neurons = get_random_number_neurons();
    const auto& element_type = get_random_element_type();

    std::stringstream ss{};
    ss << number_neurons << ' ' << element_type << '\n';

    auto [synaptic_elements, golden_counts, golden_connected_counts, golden_signal_types]
        = create_random_synaptic_elements(number_neurons, element_type, 0.0);

    const auto& grown_elements = synaptic_elements.get_grown_elements();
    const auto& connected_grown_elements = synaptic_elements.get_connected_elements();
    const auto& delta_grown_elements = synaptic_elements.get_deltas();
    const auto& signal_types = synaptic_elements.get_signal_types();

    ASSERT_EQ(grown_elements.size(), number_neurons) << ss.str();
    ASSERT_EQ(connected_grown_elements.size(), number_neurons) << ss.str();
    ASSERT_EQ(delta_grown_elements.size(), number_neurons) << ss.str();
    ASSERT_EQ(signal_types.size(), number_neurons) << ss.str();

    for (auto neuron_id : NeuronID::range(number_neurons)) {
        const auto& a1 = golden_counts[neuron_id.get_neuron_id()];
        const auto& a2 = synaptic_elements.get_grown_elements(neuron_id);
        const auto& a3 = grown_elements[neuron_id.get_neuron_id()];

        const auto& a_is_correct = a1 == a2 && a1 == a3;
        ASSERT_TRUE(a_is_correct) << ss.str() << neuron_id;

        const auto& b1 = golden_connected_counts[neuron_id.get_neuron_id()];
        const auto& b2 = synaptic_elements.get_connected_elements(neuron_id);
        const auto& b3 = connected_grown_elements[neuron_id.get_neuron_id()];

        const auto& b_is_correct = b1 == b2 && b1 == b3;
        ASSERT_TRUE(b_is_correct) << ss.str() << neuron_id;

        const auto& c1 = 0.0;
        const auto& c2 = synaptic_elements.get_delta(neuron_id);
        const auto& c3 = delta_grown_elements[neuron_id.get_neuron_id()];

        const auto& c_is_correct = c1 == c2 && c1 == c3;
        ASSERT_TRUE(c_is_correct) << ss.str() << neuron_id;

        const auto& d1 = golden_signal_types[neuron_id.get_neuron_id()];
        const auto& d2 = synaptic_elements.get_signal_type(neuron_id);
        const auto& d3 = signal_types[neuron_id.get_neuron_id()];

        const auto& d_is_correct = d1 == d2 && d1 == d3;
        ASSERT_TRUE(d_is_correct) << ss.str() << neuron_id;
    }
}

TEST_F(SynapticElementsTest, testSynapticElementsHistogram) {
    const auto& number_neurons = get_random_number_neurons();
    const auto& element_type = get_random_element_type();

    std::stringstream ss{};
    ss << number_neurons << ' ' << element_type << '\n';

    auto [synaptic_elements, golden_counts, golden_connected_counts, golden_signal_types]
        = create_random_synaptic_elements(number_neurons, element_type, 0.0);

    const auto& histogram = synaptic_elements.get_historgram();

    std::vector<std::pair<unsigned int, unsigned int>> golden_histogram{};

    for (auto i = 0; i < number_neurons; i++) {
        golden_histogram.emplace_back(golden_connected_counts[i], static_cast<unsigned int>(golden_counts[i]));
    }

    for (const auto& [pair, count] : histogram) {
        const auto golden_count = std::count(golden_histogram.begin(), golden_histogram.end(), pair);

        ASSERT_EQ(count, golden_count) << ss.str() << ' ' << pair.first << ' ' << pair.second;
    }
}

TEST_F(SynapticElementsTest, testSynapticElementsUpdateException) {
    const auto& number_neurons = get_random_number_neurons();
    const auto& element_type = get_random_element_type();

    std::stringstream ss{};
    ss << number_neurons << ' ' << element_type << '\n';

    SynapticElements synaptic_elements(element_type, 0.0);
    synaptic_elements.init(number_neurons);

    for (auto iteration = 0; iteration < number_neurons_out_of_scope; iteration++) {
        const auto neuron_id = get_random_neuron_id(number_neurons, number_neurons);

        ASSERT_THROW(auto ret = synaptic_elements.get_grown_elements(neuron_id), RelearnException) << ss.str() << neuron_id;
        ASSERT_THROW(auto ret = synaptic_elements.get_connected_elements(neuron_id), RelearnException) << ss.str() << neuron_id;
        ASSERT_THROW(auto ret = synaptic_elements.get_delta(neuron_id), RelearnException) << ss.str() << neuron_id;
    }

    for (auto iteration = 0; iteration < number_neurons_out_of_scope; iteration++) {
        const auto neuron_id = get_random_neuron_id(number_neurons, number_neurons);

        const auto& grown_element = get_random_synaptic_element_count();
        const auto& connected_grown_element = get_random_synaptic_element_connected_count(static_cast<unsigned int>(grown_element));
        const auto& signal_type = get_random_signal_type();

        ASSERT_THROW(synaptic_elements.update_grown_elements(neuron_id, grown_element), RelearnException) << ss.str() << neuron_id;
        ASSERT_THROW(synaptic_elements.update_connected_elements(neuron_id, static_cast<int>(connected_grown_element)), RelearnException) << ss.str() << neuron_id;
        ASSERT_THROW(synaptic_elements.set_signal_type(neuron_id, signal_type), RelearnException) << ss.str() << neuron_id;
    }
}

TEST_F(SynapticElementsTest, testSynapticElementsMultipleUpdate) {
    uniform_int_distribution<unsigned int> uid_connected(0, 10);

    const auto& number_neurons = get_random_number_neurons();
    const auto& element_type = get_random_element_type();

    std::stringstream ss{};
    ss << number_neurons << ' ' << element_type << '\n';

    SynapticElements synaptic_elements(element_type, 0.0);
    synaptic_elements.init(number_neurons);

    std::vector<double> golden_counts(number_neurons, 0.0);
    std::vector<unsigned int> golden_connected_counts(number_neurons, 0);
    std::vector<SignalType> golden_signal_types(number_neurons);

    for (auto iteration = 0; iteration < 10; iteration++) {
        for (auto neuron_id : NeuronID::range(number_neurons)) {
            const auto& grown_element = get_random_synaptic_element_count();
            const auto& connected_grown_element = get_random_synaptic_element_connected_count(static_cast<unsigned int>(grown_element));
            const auto& signal_type = get_random_signal_type();

            golden_counts[neuron_id.get_neuron_id()] += grown_element;
            golden_connected_counts[neuron_id.get_neuron_id()] += connected_grown_element;
            golden_signal_types[neuron_id.get_neuron_id()] = signal_type;

            synaptic_elements.update_grown_elements(neuron_id, grown_element);
            synaptic_elements.update_connected_elements(neuron_id, connected_grown_element);
            synaptic_elements.set_signal_type(neuron_id, signal_type);
        }
    }

    const auto& grown_elements = synaptic_elements.get_grown_elements();
    const auto& connected_grown_elements = synaptic_elements.get_connected_elements();
    const auto& delta_grown_elements = synaptic_elements.get_deltas();
    const auto& signal_types = synaptic_elements.get_signal_types();

    ASSERT_EQ(grown_elements.size(), number_neurons) << ss.str();
    ASSERT_EQ(connected_grown_elements.size(), number_neurons) << ss.str();
    ASSERT_EQ(delta_grown_elements.size(), number_neurons) << ss.str();
    ASSERT_EQ(signal_types.size(), number_neurons) << ss.str();

    for (auto neuron_id : NeuronID::range(number_neurons)) {
        ASSERT_EQ(golden_counts[neuron_id.get_neuron_id()], synaptic_elements.get_grown_elements(neuron_id)) << ss.str() << neuron_id;
        ASSERT_EQ(golden_counts[neuron_id.get_neuron_id()], grown_elements[neuron_id.get_neuron_id()]) << ss.str() << neuron_id;

        ASSERT_EQ(golden_connected_counts[neuron_id.get_neuron_id()], synaptic_elements.get_connected_elements(neuron_id)) << ss.str() << neuron_id;
        ASSERT_EQ(golden_connected_counts[neuron_id.get_neuron_id()], connected_grown_elements[neuron_id.get_neuron_id()]) << ss.str() << neuron_id;

        ASSERT_EQ(golden_signal_types[neuron_id.get_neuron_id()], synaptic_elements.get_signal_type(neuron_id)) << ss.str() << neuron_id;
        ASSERT_EQ(golden_signal_types[neuron_id.get_neuron_id()], signal_types[neuron_id.get_neuron_id()]) << ss.str() << neuron_id;
    }
}

TEST_F(SynapticElementsTest, testSynapticElementsFreeElements) {
    uniform_int_distribution<unsigned int> uid_connected(0, 10);

    const auto& number_neurons = get_random_number_neurons();
    const auto& element_type = get_random_element_type();

    std::stringstream ss{};
    ss << number_neurons << ' ' << element_type << '\n';

    SynapticElements synaptic_elements(element_type, 0.0);
    synaptic_elements.init(number_neurons);

    std::vector<double> golden_counts(number_neurons, 0.0);
    std::vector<unsigned int> golden_connected_counts(number_neurons, 0);
    std::vector<SignalType> golden_signal_types(number_neurons);

    for (auto iteration = 0; iteration < 10; iteration++) {
        for (auto neuron_id : NeuronID::range(number_neurons)) {
            const auto& grown_element = get_random_synaptic_element_count();
            const auto& connected_grown_element = get_random_synaptic_element_connected_count(static_cast<unsigned int>(grown_element));
            const auto& signal_type = get_random_signal_type();

            golden_counts[neuron_id.get_neuron_id()] += grown_element;
            golden_connected_counts[neuron_id.get_neuron_id()] += connected_grown_element;
            golden_signal_types[neuron_id.get_neuron_id()] = signal_type;

            synaptic_elements.update_grown_elements(neuron_id, grown_element);
            synaptic_elements.update_connected_elements(neuron_id, connected_grown_element);
            synaptic_elements.set_signal_type(neuron_id, signal_type);
        }
    }

    for (auto neuron_id : NeuronID::range(number_neurons)) {
        const auto nid = neuron_id.get_neuron_id();
        const auto expected_number_free_elements = golden_counts[nid] - golden_connected_counts[nid];
        const auto expected_number_free_elements_cast = static_cast<unsigned int>(expected_number_free_elements);

        ASSERT_EQ(expected_number_free_elements_cast, synaptic_elements.get_free_elements(neuron_id)) << ss.str() << neuron_id;

        if (golden_signal_types[nid] == SignalType::Excitatory) {
            ASSERT_EQ(expected_number_free_elements_cast, synaptic_elements.get_free_elements(neuron_id, SignalType::Excitatory)) << ss.str() << neuron_id;
            ASSERT_EQ(0, synaptic_elements.get_free_elements(neuron_id, SignalType::Inhibitory)) << ss.str() << neuron_id;
        } else {
            ASSERT_EQ(expected_number_free_elements_cast, synaptic_elements.get_free_elements(neuron_id, SignalType::Inhibitory)) << ss.str() << neuron_id;
            ASSERT_EQ(0, synaptic_elements.get_free_elements(neuron_id, SignalType::Excitatory)) << ss.str() << neuron_id;
        }
    }
}

TEST_F(SynapticElementsTest, testSynapticElementsDisable) {
    const auto& number_neurons = get_random_number_neurons();
    const auto& element_type = get_random_element_type();

    std::stringstream ss{};
    ss << number_neurons << ' ' << element_type << '\n';

    auto [synaptic_elements, golden_counts, golden_connected_counts, golden_signal_types]
        = create_random_synaptic_elements(number_neurons, element_type, 0.0);

    std::vector<unsigned int> changes(number_neurons, 0);
    std::vector<NeuronID> disabled_neurons{};
    std::vector<bool> disabled(number_neurons, false);

    for (auto neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
        const auto disable = get_random_bool();
        if (disable) {
            disabled_neurons.emplace_back(neuron_id);
            disabled[neuron_id] = true;
        }
    }

    std::shuffle(disabled_neurons.begin(), disabled_neurons.end(), mt);

    synaptic_elements.update_after_deletion(changes, disabled_neurons);

    for (auto neuron_id : NeuronID::range(number_neurons)) {
        const auto is_disabled = disabled[neuron_id.get_neuron_id()];

        ASSERT_EQ(synaptic_elements.get_signal_type(neuron_id), golden_signal_types[neuron_id.get_neuron_id()]) << ss.str() << neuron_id;

        if (is_disabled) {
            ASSERT_EQ(synaptic_elements.get_connected_elements(neuron_id), 0) << ss.str() << neuron_id << " disabled";
            ASSERT_EQ(synaptic_elements.get_grown_elements(neuron_id), 0.0) << ss.str() << neuron_id << " disabled";
            ASSERT_EQ(synaptic_elements.get_delta(neuron_id), 0.0) << ss.str() << neuron_id << " disabled";
        } else {
            ASSERT_EQ(synaptic_elements.get_connected_elements(neuron_id), golden_connected_counts[neuron_id.get_neuron_id()]) << ss.str() << neuron_id << " enabled";
            ASSERT_EQ(synaptic_elements.get_grown_elements(neuron_id), golden_counts[neuron_id.get_neuron_id()]) << ss.str() << neuron_id << " enabled";
        }
    }
}

TEST_F(SynapticElementsTest, testSynapticElementsDisableException) {
    const auto& number_neurons = get_random_number_neurons();
    const auto& element_type = get_random_element_type();

    std::stringstream ss{};
    ss << number_neurons << ' ' << element_type << '\n';

    auto [synaptic_elements, golden_counts, golden_connected_counts, golden_signal_types]
        = create_random_synaptic_elements(number_neurons, element_type, 0.0);

    for (auto iteration = 0; iteration < 10; iteration++) {
        std::vector<unsigned int> changes(number_neurons, 0);
        std::vector<NeuronID> disabled_neurons{};

        for (auto neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
            const auto disable = get_random_bool();
            if (disable) {
                disabled_neurons.emplace_back(neuron_id);
            }
        }

        const auto faulty_id = get_random_number_neurons() + number_neurons;
        disabled_neurons.emplace_back(faulty_id);

        std::shuffle(disabled_neurons.begin(), disabled_neurons.end(), mt);

        ASSERT_THROW(synaptic_elements.update_after_deletion(changes, disabled_neurons), RelearnException) << ss.str() << ' ' << faulty_id;
    }
}

TEST_F(SynapticElementsTest, testSynapticElementsDelete) {
    const auto& number_neurons = get_random_number_neurons();
    const auto& element_type = get_random_element_type();

    std::stringstream ss{};
    ss << number_neurons << ' ' << element_type << '\n';

    auto [synaptic_elements, golden_counts, golden_connected_counts, golden_signal_types]
        = create_random_synaptic_elements(number_neurons, element_type, 0.0);

    std::vector<unsigned int> changes(number_neurons, 0);
    std::vector<NeuronID> disabled_neurons{};

    for (auto neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
        const auto change = get_random_integer<unsigned int>(0, golden_connected_counts[neuron_id]);
        changes[neuron_id] = change;
    }

    synaptic_elements.update_after_deletion(changes, disabled_neurons);

    for (auto neuron_id : NeuronID::range(number_neurons)) {
        const auto new_connected_count = golden_connected_counts[neuron_id.get_neuron_id()] - changes[neuron_id.get_neuron_id()];

        ASSERT_EQ(synaptic_elements.get_signal_type(neuron_id), golden_signal_types[neuron_id.get_neuron_id()]) << ss.str() << neuron_id;
        ASSERT_EQ(synaptic_elements.get_connected_elements(neuron_id), new_connected_count) << ss.str() << neuron_id;
        ASSERT_EQ(synaptic_elements.get_grown_elements(neuron_id), golden_counts[neuron_id.get_neuron_id()]) << ss.str() << neuron_id;
    }
}

TEST_F(SynapticElementsTest, testSynapticElementsDeleteException) {
    const auto& number_neurons = get_random_number_neurons();
    const auto& element_type = get_random_element_type();

    std::stringstream ss{};
    ss << number_neurons << ' ' << element_type << '\n';

    auto [synaptic_elements, golden_counts, golden_connected_counts, golden_signal_types]
        = create_random_synaptic_elements(number_neurons, element_type, 0.0);

    for (auto iteration = 0; iteration < 10; iteration++) {
        auto wrong_number_neurons = get_random_number_neurons();
        if (wrong_number_neurons == number_neurons) {
            wrong_number_neurons++;
        }

        std::vector<NeuronID> disabled_neurons{};
        std::vector<unsigned int> wrong_changes(wrong_number_neurons, 0);

        ASSERT_THROW(synaptic_elements.update_after_deletion(wrong_changes, disabled_neurons), RelearnException) << ss.str() << ' ' << wrong_number_neurons;

        std::vector<unsigned int> changes(number_neurons, 0);

        for (auto neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
            const auto change = get_random_integer<unsigned int>(0, golden_connected_counts[neuron_id]) + 1;
            changes[neuron_id] = change;
        }

        const auto faulty_id = get_random_neuron_id(number_neurons);
        changes[faulty_id.get_neuron_id()] += golden_connected_counts[faulty_id.get_neuron_id()];

        ASSERT_THROW(synaptic_elements.update_after_deletion(changes, disabled_neurons), RelearnException) << ss.str() << ' ' << faulty_id;
    }
}

TEST_F(SynapticElementsTest, testSynapticElementsUpdateNumberElements) {
    const auto minimum_calcium_to_grow = get_random_double(-100.0, 100.0);
    const auto growth_factor = get_random_double(1e-6, 100.0);

    const auto& number_neurons = get_random_number_neurons();
    const auto& element_type = get_random_element_type();

    std::stringstream ss{};
    ss << number_neurons << ' ' << element_type << ' ' << minimum_calcium_to_grow << ' ' << growth_factor << '\n';

    auto [synaptic_elements, golden_counts, golden_connected_counts, golden_signal_types]
        = create_random_synaptic_elements(number_neurons, element_type, minimum_calcium_to_grow, growth_factor);

    std::vector<double> calcium(number_neurons, 0.0);
    std::vector<double> target_calcium(number_neurons, 0.0);
    std::vector<UpdateStatus> disable_flags(number_neurons, UpdateStatus::Disabled);

    for (auto neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
        calcium[neuron_id] = get_random_double(-100.0, 100.0);
        target_calcium[neuron_id] = get_random_double(minimum_calcium_to_grow, minimum_calcium_to_grow + 200.0);
        if (get_random_bool()) {
            disable_flags[neuron_id] = UpdateStatus::Enabled;
        }
    }

    synaptic_elements.update_number_elements_delta(calcium, target_calcium, disable_flags);

    for (auto neuron_id : NeuronID::range(number_neurons)) {
        ASSERT_EQ(synaptic_elements.get_connected_elements(neuron_id), golden_connected_counts[neuron_id.get_neuron_id()]) << ss.str() << neuron_id;
        ASSERT_EQ(synaptic_elements.get_grown_elements(neuron_id), golden_counts[neuron_id.get_neuron_id()]) << ss.str() << neuron_id;
    }

    const auto& actual_deltas = synaptic_elements.get_deltas();
    for (auto id : NeuronID::range(number_neurons)) {
        const auto actual_delta = synaptic_elements.get_delta(id);
        const auto computed_delta = gaussian_growth_curve(calcium[id.get_neuron_id()], minimum_calcium_to_grow, target_calcium[id.get_neuron_id()], growth_factor);

        if (disable_flags[id.get_neuron_id()] == UpdateStatus::Disabled) {
            ASSERT_NEAR(actual_delta, 0.0, eps) << ss.str() << id;
            ASSERT_NEAR(actual_deltas[id.get_neuron_id()], 0.0, eps) << ss.str() << id;
        } else {
            ASSERT_NEAR(actual_delta, computed_delta, eps) << ss.str() << id;
            ASSERT_NEAR(actual_deltas[id.get_neuron_id()], computed_delta, eps) << ss.str() << id;
        }
    }
}

TEST_F(SynapticElementsTest, testSynapticElementsMultipleUpdateNumberElements) {
    const auto minimum_calcium_to_grow = get_random_double(-100.0, 100.0);
    const auto growth_factor = get_random_double(1e-6, 100.0);

    const auto& number_neurons = get_random_number_neurons();
    const auto& element_type = get_random_element_type();

    std::stringstream ss{};
    ss << number_neurons << ' ' << element_type << '\n';

    auto [synaptic_elements, golden_counts, golden_connected_counts, golden_signal_types]
        = create_random_synaptic_elements(number_neurons, element_type, minimum_calcium_to_grow, growth_factor);

    std::vector<double> golden_delta_counts(number_neurons, 0.0);

    for (auto i = 0; i < 10; i++) {
        std::vector<double> calcium(number_neurons, 0.0);
        std::vector<double> target_calcium(number_neurons, 0.0);
        std::vector<UpdateStatus> disable_flags(number_neurons, UpdateStatus::Disabled);

        for (auto neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
            calcium[neuron_id] = get_random_double(-100.0, 100.0);
            target_calcium[neuron_id] = get_random_double(minimum_calcium_to_grow, minimum_calcium_to_grow + 200.0);
            if (get_random_bool()) {
                disable_flags[neuron_id] = UpdateStatus::Enabled;
                const auto current_expected_delta = gaussian_growth_curve(calcium[neuron_id], minimum_calcium_to_grow, target_calcium[neuron_id], growth_factor);
                golden_delta_counts[neuron_id] += current_expected_delta;
            }
        }

        synaptic_elements.update_number_elements_delta(calcium, target_calcium, disable_flags);
    }

    for (auto neuron_id : NeuronID::range(number_neurons)) {
        ASSERT_EQ(synaptic_elements.get_connected_elements(neuron_id), golden_connected_counts[neuron_id.get_neuron_id()]) << ss.str() << neuron_id;
        ASSERT_EQ(synaptic_elements.get_grown_elements(neuron_id), golden_counts[neuron_id.get_neuron_id()]) << ss.str() << neuron_id;
    }

    const auto& actual_deltas = synaptic_elements.get_deltas();
    for (auto neuron_id : NeuronID::range(number_neurons)) {
        const auto actual_delta = synaptic_elements.get_delta(neuron_id);
        const auto expected_delta = golden_delta_counts[neuron_id.get_neuron_id()];

        ASSERT_NEAR(actual_delta, expected_delta, eps) << ss.str() << neuron_id;
        ASSERT_NEAR(actual_deltas[neuron_id.get_neuron_id()], expected_delta, eps) << ss.str() << neuron_id;
    }
}

TEST_F(SynapticElementsTest, testSynapticElementsUpdateNumberElementsException) {
    const auto& number_neurons = get_random_number_neurons();
    const auto& element_type = get_random_element_type();

    std::stringstream ss{};
    ss << number_neurons << ' ' << element_type << '\n';

    auto [synaptic_elements, grown_elements, connected_elements, signal_types]
        = create_random_synaptic_elements(number_neurons, element_type, 0.0);

    const auto number_too_small_calcium = get_random_neuron_id(number_neurons);
    const auto number_too_large_calcium = get_random_neuron_id(number_neurons, number_neurons + 1);

    const auto number_too_small_target_calcium = get_random_neuron_id(number_neurons);
    const auto number_too_large_target_calcium = get_random_neuron_id(number_neurons, number_neurons + 1);

    const auto number_too_small_disable_flags = get_random_neuron_id(number_neurons);
    const auto number_too_large_disable_flags = get_random_neuron_id(number_neurons, number_neurons + 1);

    std::vector<double> calcium_too_small(number_too_small_calcium.get_neuron_id(), 0.0);
    std::vector<double> calcium(number_neurons, 0.0);
    std::vector<double> calcium_too_large(number_too_large_calcium.get_neuron_id(), 0.0);

    std::vector<double> target_calcium_too_small(number_too_small_target_calcium.get_neuron_id(), 0.0);
    std::vector<double> target_calcium(number_neurons, 0.0);
    std::vector<double> target_calcium_too_large(number_too_large_target_calcium.get_neuron_id(), 0.0);

    std::vector<UpdateStatus> disable_flags_too_small(number_too_small_disable_flags.get_neuron_id(), UpdateStatus::Disabled);
    std::vector<UpdateStatus> disable_flags(number_neurons, UpdateStatus::Disabled);
    std::vector<UpdateStatus> disable_flags_too_large(number_too_large_disable_flags.get_neuron_id(), UpdateStatus::Disabled);

    auto lambda = [&ss, &synaptic_elements](auto calcium, auto target_calcium, auto disable_flags) {
        ASSERT_THROW(synaptic_elements.update_number_elements_delta(calcium, target_calcium, disable_flags), RelearnException) << ss.str()
                                                                                                                               << calcium.size() << ' '
                                                                                                                               << target_calcium.size() << ' '
                                                                                                                               << disable_flags.size();
    };

    lambda(calcium_too_small, target_calcium_too_small, disable_flags_too_small);
    lambda(calcium_too_small, target_calcium, disable_flags_too_small);
    lambda(calcium_too_small, target_calcium_too_large, disable_flags_too_small);

    lambda(calcium_too_small, target_calcium_too_small, disable_flags);
    lambda(calcium_too_small, target_calcium, disable_flags);
    lambda(calcium_too_small, target_calcium_too_large, disable_flags);

    lambda(calcium_too_small, target_calcium_too_small, disable_flags_too_large);
    lambda(calcium_too_small, target_calcium, disable_flags_too_large);
    lambda(calcium_too_small, target_calcium_too_large, disable_flags_too_large);

    lambda(calcium, target_calcium_too_small, disable_flags_too_small);
    lambda(calcium, target_calcium, disable_flags_too_small);
    lambda(calcium, target_calcium_too_large, disable_flags_too_small);

    lambda(calcium, target_calcium_too_small, disable_flags);
    lambda(calcium, target_calcium_too_large, disable_flags);

    lambda(calcium, target_calcium_too_small, disable_flags_too_large);
    lambda(calcium, target_calcium, disable_flags_too_large);
    lambda(calcium, target_calcium_too_large, disable_flags_too_large);

    lambda(calcium_too_large, target_calcium_too_small, disable_flags_too_small);
    lambda(calcium_too_large, target_calcium, disable_flags_too_small);
    lambda(calcium_too_large, target_calcium_too_large, disable_flags_too_small);

    lambda(calcium_too_large, target_calcium_too_small, disable_flags);
    lambda(calcium_too_large, target_calcium, disable_flags);
    lambda(calcium_too_large, target_calcium_too_large, disable_flags);

    lambda(calcium_too_large, target_calcium_too_small, disable_flags_too_large);
    lambda(calcium_too_large, target_calcium, disable_flags_too_large);
    lambda(calcium_too_large, target_calcium_too_large, disable_flags_too_large);
}

TEST_F(SynapticElementsTest, testSynapticElementsCommitUpdates) {
    const auto minimum_calcium_to_grow = get_random_double(-100.0, 100.0);
    const auto growth_factor = get_random_double(1e-6, 100.0);

    const auto number_neurons = get_random_number_neurons();
    const auto element_type = get_random_element_type();

    const auto retract_ratio = get_random_double(0.0, 1.0);

    std::stringstream ss{};
    ss << number_neurons << ' ' << element_type << ' ' << minimum_calcium_to_grow << ' ' << growth_factor << ' ' << retract_ratio << '\n';

    SynapticElements synaptic_elements(element_type, minimum_calcium_to_grow, growth_factor, retract_ratio);
    synaptic_elements.init(number_neurons);

    std::vector<double> golden_counts(number_neurons);
    std::vector<unsigned int> golden_connected_counts(number_neurons);
    std::vector<SignalType> golden_signal_types(number_neurons);

    for (auto neuron_id : NeuronID::range(number_neurons)) {
        const auto& grown_element = get_random_double(0, 10.0);
        const auto& connected_grown_element = get_random_integer<unsigned int>(0, static_cast<unsigned int>(grown_element));
        const auto& signal_type = get_random_signal_type();

        golden_counts[neuron_id.get_neuron_id()] = grown_element;
        golden_connected_counts[neuron_id.get_neuron_id()] = static_cast<unsigned int>(connected_grown_element);
        golden_signal_types[neuron_id.get_neuron_id()] = signal_type;

        synaptic_elements.update_grown_elements(neuron_id, grown_element);
        synaptic_elements.update_connected_elements(neuron_id, static_cast<int>(connected_grown_element));
        synaptic_elements.set_signal_type(neuron_id, signal_type);
    }

    std::vector<double> calcium(number_neurons, 0.0);
    std::vector<double> target_calcium(number_neurons, 0.0);
    std::vector<UpdateStatus> enable_flags(number_neurons, UpdateStatus::Enabled);
    std::vector<UpdateStatus> disable_flags(number_neurons, UpdateStatus::Disabled);

    for (auto neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
        calcium[neuron_id] = get_random_double(-100.0, 100.0);
        target_calcium[neuron_id] = get_random_double(minimum_calcium_to_grow, minimum_calcium_to_grow + 200.0);
        if (get_random_bool()) {
            disable_flags[neuron_id] = UpdateStatus::Enabled;
        }
    }

    synaptic_elements.update_number_elements_delta(calcium, target_calcium, enable_flags);
    const auto& [number_deleted_elements, deleted_element_counts] = synaptic_elements.commit_updates(disable_flags);

    const auto& deltas = synaptic_elements.get_deltas();

    for (auto neuron_id : NeuronID::range(number_neurons)) {
        const auto computed_delta = gaussian_growth_curve(calcium[neuron_id.get_neuron_id()], minimum_calcium_to_grow, target_calcium[neuron_id.get_neuron_id()], growth_factor);
        const auto delta = synaptic_elements.get_delta(neuron_id);
        if (disable_flags[neuron_id.get_neuron_id()] == UpdateStatus::Disabled) {
            ASSERT_NEAR(delta, computed_delta, eps) << ss.str() << neuron_id;
            ASSERT_NEAR(deltas[neuron_id.get_neuron_id()], computed_delta, eps) << ss.str() << neuron_id;
        } else {
            ASSERT_EQ(delta, 0.0) << ss.str() << neuron_id;
            ASSERT_EQ(deltas[neuron_id.get_neuron_id()], 0.0) << ss.str() << neuron_id;
        }
    }

    auto summed_number_deletions = 0;
    for (auto deleted_counts : deleted_element_counts) {
        summed_number_deletions += deleted_counts;
    }

    ASSERT_EQ(summed_number_deletions, number_deleted_elements) << ss.str() << summed_number_deletions << ' ' << number_deleted_elements;

    for (auto neuron_id : NeuronID::range(number_neurons)) {
        if (disable_flags[neuron_id.get_neuron_id()] == UpdateStatus::Disabled) {
            continue;
        }

        const auto previous_count = golden_counts[neuron_id.get_neuron_id()];
        const auto previous_connected = golden_connected_counts[neuron_id.get_neuron_id()];
        const auto previous_vacant = previous_count - previous_connected;

        const auto current_count = synaptic_elements.get_grown_elements(neuron_id);
        const auto current_connected = synaptic_elements.get_connected_elements(neuron_id);
        const auto current_delta = synaptic_elements.get_delta(neuron_id);

        const auto computed_delta = gaussian_growth_curve(calcium[neuron_id.get_neuron_id()], minimum_calcium_to_grow, target_calcium[neuron_id.get_neuron_id()], growth_factor);
        const auto new_vacant = previous_vacant + computed_delta;

        ASSERT_EQ(current_delta, 0.0) << ss.str() << neuron_id;

        if (new_vacant >= 0.0) {
            const auto retracted_count = (1 - retract_ratio) * new_vacant;
            const auto expected_count = retracted_count + previous_connected;

            ASSERT_NEAR(expected_count, current_count, eps) << ss.str() << neuron_id;
            ASSERT_EQ(previous_connected, current_connected) << ss.str() << neuron_id;

            continue;
        }

        const auto expected_deletions = static_cast<unsigned int>(std::ceil(std::abs(new_vacant)));

        if (expected_deletions > previous_connected) {
            ASSERT_EQ(current_count, 0.0) << ss.str() << neuron_id;
            ASSERT_EQ(current_connected, 0) << ss.str() << neuron_id;

            continue;
        }

        if (expected_deletions == previous_connected) {
            const auto expected_count = (1 - retract_ratio) * (previous_count + computed_delta);

            ASSERT_NEAR(current_count, expected_count, eps) << ss.str() << neuron_id;
            ASSERT_EQ(current_connected, 0) << ss.str() << neuron_id;

            continue;
        }

        const auto expected_connected = previous_connected - expected_deletions;

        ASSERT_EQ(expected_connected, current_connected) << ss.str() << neuron_id;
        ASSERT_EQ(expected_deletions, deleted_element_counts[neuron_id.get_neuron_id()]) << ss.str() << neuron_id;

        const auto expected_count = (1 - retract_ratio) * (previous_count + computed_delta - expected_connected) + expected_connected;
        ASSERT_NEAR(current_count, expected_count, eps) << ss.str() << neuron_id;
    }
}

TEST_F(SynapticElementsTest, testSynapticElementsCommitUpdatesException) {
    const auto minimum_calcium_to_grow = get_random_double(-100.0, 100.0);
    const auto growth_factor = get_random_double(1e-6, 100.0);

    const auto& number_neurons = get_random_number_neurons();
    const auto& element_type = get_random_element_type();

    std::stringstream ss{};
    ss << number_neurons << ' ' << element_type << ' ' << minimum_calcium_to_grow << ' ' << growth_factor << '\n';

    auto [synaptic_elements, grown_elements, connected_elements, signal_types]
        = create_random_synaptic_elements(number_neurons, element_type, minimum_calcium_to_grow, growth_factor);

    std::vector<double> calcium(number_neurons, 0.0);
    std::vector<double> target_calcium(number_neurons, 0.0);
    std::vector<UpdateStatus> disable_flags(number_neurons, UpdateStatus::Disabled);

    for (auto neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
        calcium[neuron_id] = get_random_double(-100.0, 100.0);
        target_calcium[neuron_id] = get_random_double(minimum_calcium_to_grow, minimum_calcium_to_grow + 200.0);
        if (get_random_bool()) {
            disable_flags[neuron_id] = UpdateStatus::Enabled;
        }
    }

    synaptic_elements.update_number_elements_delta(calcium, target_calcium, disable_flags);

    const auto number_too_small_disable_flags = get_random_neuron_id(number_neurons).get_neuron_id();
    const auto number_too_large_disable_flags = get_random_neuron_id(number_neurons).get_neuron_id() + number_neurons + 1;

    std::vector<UpdateStatus> disable_flags_too_small(number_too_small_disable_flags, UpdateStatus::Disabled);
    std::vector<UpdateStatus> disable_flags_too_large(number_too_large_disable_flags, UpdateStatus::Disabled);

    ASSERT_THROW(auto ret = synaptic_elements.commit_updates(disable_flags_too_small), RelearnException) << ss.str() << number_too_small_disable_flags;
    ASSERT_THROW(auto ret = synaptic_elements.commit_updates(disable_flags_too_large), RelearnException) << ss.str() << number_too_large_disable_flags;
}
