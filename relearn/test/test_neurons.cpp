#include "../googletest/include/gtest/gtest.h"

#include "commons.h"

#include "../source/neurons/Neurons.h"
#include "../source/neurons/models/NeuronModels.h"

#include "../source/neurons/NetworkGraph.h"

#include "../source/structure/Partition.h"

#include <algorithm>
#include <map>
#include <numeric>
#include <random>
#include <stack>
#include <tuple>
#include <vector>

NetworkGraph generate_random_network_graph(size_t num_neurons, size_t num_synapses, double threshold_exc) {
    std::uniform_int_distribution<size_t> uid(0, num_neurons - 1);
    std::uniform_real_distribution<double> urd(0, 1.0);

    NetworkGraph ng(num_neurons);

    for (size_t synapse_id = 0; synapse_id < num_synapses; synapse_id++) {
        const auto neuron_id_1 = uid(mt);
        auto neuron_id_2 = uid(mt);

        if (neuron_id_2 == neuron_id_1) {
            neuron_id_2 = (neuron_id_1 + 1) % num_neurons;
        }

        const auto uniform_double = urd(mt);
        const auto weight = (uniform_double < threshold_exc) ? 1 : -1;

        ng.add_edge_weight(neuron_id_1, 0, neuron_id_2, 0, weight);
    }

    return ng;
}

std::vector<size_t> generate_random_ids(size_t id_low, size_t id_high, size_t num_disables) {
    std::vector<size_t> disable_ids(num_disables);

    std::uniform_int_distribution<size_t> uid(id_low, id_high);

    for (size_t i = 0; i < num_disables; i++) {
        disable_ids[i] = uid(mt);
    }

    return disable_ids;
}

TEST(TestNeuronModels, testNeuronModelsDefaultConstructor) {
    using namespace models;

    setup();

    for (auto i = 0; i < iterations; i++) {
        const auto expected_k = NeuronModels::default_k;
        const auto expected_tau_C = NeuronModels::default_tau_C;
        const auto expected_beta = NeuronModels::default_beta;
        const auto expected_h = NeuronModels::default_h;
        const auto expected_base_background_activity = NeuronModels::default_base_background_activity;
        const auto expected_background_activity_mean = NeuronModels::default_background_activity_mean;
        const auto expected_background_activity_stddev = NeuronModels::default_background_activity_stddev;

        auto model = std::make_unique<PoissonModel>();

        EXPECT_EQ(expected_k, model->get_k());
        EXPECT_EQ(expected_tau_C, model->get_tau_C());
        EXPECT_EQ(expected_beta, model->get_beta());
        EXPECT_EQ(expected_h, model->get_h());
        EXPECT_EQ(expected_base_background_activity, model->get_base_background_activity());
        EXPECT_EQ(expected_background_activity_mean, model->get_background_activity_mean());
        EXPECT_EQ(expected_background_activity_stddev, model->get_background_activity_stddev());

        const auto expected_x0 = PoissonModel::default_x_0;
        const auto expected_refrac = PoissonModel::default_refrac_time;
        const auto expected_tau_x = PoissonModel::default_tau_x;

        EXPECT_EQ(expected_x0, model->get_x_0());
        EXPECT_EQ(expected_refrac, model->get_refrac_time());
        EXPECT_EQ(expected_tau_x, model->get_tau_x());
    }

    for (auto i = 0; i < iterations; i++) {
        const auto expected_k = NeuronModels::default_k;
        const auto expected_tau_C = NeuronModels::default_tau_C;
        const auto expected_beta = NeuronModels::default_beta;
        const auto expected_h = NeuronModels::default_h;
        const auto expected_base_background_activity = NeuronModels::default_base_background_activity;
        const auto expected_background_activity_mean = NeuronModels::default_background_activity_mean;
        const auto expected_background_activity_stddev = NeuronModels::default_background_activity_stddev;

        auto model = std::make_unique<IzhikevichModel>();

        EXPECT_EQ(expected_k, model->get_k());
        EXPECT_EQ(expected_tau_C, model->get_tau_C());
        EXPECT_EQ(expected_beta, model->get_beta());
        EXPECT_EQ(expected_h, model->get_h());
        EXPECT_EQ(expected_base_background_activity, model->get_base_background_activity());
        EXPECT_EQ(expected_background_activity_mean, model->get_background_activity_mean());
        EXPECT_EQ(expected_background_activity_stddev, model->get_background_activity_stddev());

        const auto expected_a = IzhikevichModel::default_a;
        const auto expected_b = IzhikevichModel::default_b;
        const auto expected_c = IzhikevichModel::default_c;
        const auto expected_d = IzhikevichModel::default_d;
        const auto expected_V_spike = IzhikevichModel::default_V_spike;
        const auto expected_k1 = IzhikevichModel::default_k1;
        const auto expected_k2 = IzhikevichModel::default_k2;
        const auto expected_k3 = IzhikevichModel::default_k3;

        EXPECT_EQ(expected_a, model->get_a());
        EXPECT_EQ(expected_b, model->get_b());
        EXPECT_EQ(expected_c, model->get_c());
        EXPECT_EQ(expected_d, model->get_d());
        EXPECT_EQ(expected_V_spike, model->get_V_spike());
        EXPECT_EQ(expected_k1, model->get_k1());
        EXPECT_EQ(expected_k2, model->get_k2());
        EXPECT_EQ(expected_k3, model->get_k3());
    }

    for (auto i = 0; i < iterations; i++) {
        const auto expected_k = NeuronModels::default_k;
        const auto expected_tau_C = NeuronModels::default_tau_C;
        const auto expected_beta = NeuronModels::default_beta;
        const auto expected_h = NeuronModels::default_h;
        const auto expected_base_background_activity = NeuronModels::default_base_background_activity;
        const auto expected_background_activity_mean = NeuronModels::default_background_activity_mean;
        const auto expected_background_activity_stddev = NeuronModels::default_background_activity_stddev;

        auto model = std::make_unique<FitzHughNagumoModel>();

        EXPECT_EQ(expected_k, model->get_k());
        EXPECT_EQ(expected_tau_C, model->get_tau_C());
        EXPECT_EQ(expected_beta, model->get_beta());
        EXPECT_EQ(expected_h, model->get_h());
        EXPECT_EQ(expected_base_background_activity, model->get_base_background_activity());
        EXPECT_EQ(expected_background_activity_mean, model->get_background_activity_mean());
        EXPECT_EQ(expected_background_activity_stddev, model->get_background_activity_stddev());

        const auto expected_a = FitzHughNagumoModel::default_a;
        const auto expected_b = FitzHughNagumoModel::default_b;
        const auto expected_phi = FitzHughNagumoModel::default_phi;

        EXPECT_EQ(expected_a, model->get_a());
        EXPECT_EQ(expected_b, model->get_b());
        EXPECT_EQ(expected_phi, model->get_phi());
    }

    for (auto i = 0; i < iterations; i++) {
        const auto expected_k = NeuronModels::default_k;
        const auto expected_tau_C = NeuronModels::default_tau_C;
        const auto expected_beta = NeuronModels::default_beta;
        const auto expected_h = NeuronModels::default_h;
        const auto expected_base_background_activity = NeuronModels::default_base_background_activity;
        const auto expected_background_activity_mean = NeuronModels::default_background_activity_mean;
        const auto expected_background_activity_stddev = NeuronModels::default_background_activity_stddev;

        auto model = std::make_unique<AEIFModel>();

        EXPECT_EQ(expected_k, model->get_k());
        EXPECT_EQ(expected_tau_C, model->get_tau_C());
        EXPECT_EQ(expected_beta, model->get_beta());
        EXPECT_EQ(expected_h, model->get_h());
        EXPECT_EQ(expected_base_background_activity, model->get_base_background_activity());
        EXPECT_EQ(expected_background_activity_mean, model->get_background_activity_mean());
        EXPECT_EQ(expected_background_activity_stddev, model->get_background_activity_stddev());

        const auto expected_C = AEIFModel::default_C;
        const auto expected_g_L = AEIFModel::default_g_L;
        const auto expected_E_L = AEIFModel::default_E_L;
        const auto expected_V_T = AEIFModel::default_V_T;
        const auto expected_d_T = AEIFModel::default_d_T;
        const auto expected_tau_w = AEIFModel::default_tau_w;
        const auto expected_a = AEIFModel::default_a;
        const auto expected_b = AEIFModel::default_b;
        const auto expected_V_peak = AEIFModel::default_V_peak;

        EXPECT_EQ(expected_C, model->get_C());
        EXPECT_EQ(expected_g_L, model->get_g_L());
        EXPECT_EQ(expected_E_L, model->get_E_L());
        EXPECT_EQ(expected_V_T, model->get_V_T());
        EXPECT_EQ(expected_d_T, model->get_d_T());
        EXPECT_EQ(expected_tau_w, model->get_tau_w());
        EXPECT_EQ(expected_a, model->get_a());
        EXPECT_EQ(expected_b, model->get_b());
        EXPECT_EQ(expected_V_peak, model->get_V_Peak());
    }
}

TEST(TestNeuronModels, testNeuronModelsRandomConstructor) {
    using namespace models;
    using urd = std::uniform_real_distribution<double>;
    using uid = std::uniform_int_distribution<unsigned int>;

    setup();

    for (auto i = 0; i < iterations; i++) {
        const urd urd_desired_k(NeuronModels::min_k, NeuronModels::max_k);
        const urd urd_desired_tau_C(NeuronModels::min_tau_C, NeuronModels::max_tau_C);
        const urd urd_desired_beta(NeuronModels::min_beta, NeuronModels::max_beta);
        const uid uid_desired_h(NeuronModels::min_h, NeuronModels::max_h);
        const urd urd_desired_base_background_activity(NeuronModels::min_base_background_activity, NeuronModels::max_base_background_activity);
        const urd urd_desired_background_activity_mean(NeuronModels::min_background_activity_mean, NeuronModels::max_background_activity_mean);
        const urd urd_desired_background_activity_stddev(NeuronModels::min_background_activity_stddev, NeuronModels::max_background_activity_stddev);

        const urd urd_desired_x0(PoissonModel::min_x_0, PoissonModel::max_x_0);
        const uid urd_desired_refrac(PoissonModel::min_refrac_time, PoissonModel::max_refrac_time);
        const urd urd_desired_tau_x(PoissonModel::min_tau_x, PoissonModel::max_tau_x);

        const auto expected_k = urd_desired_k(mt);
        const auto expected_tau_C = urd_desired_tau_C(mt);
        const auto expected_beta = urd_desired_beta(mt);
        const auto expected_h = uid_desired_h(mt);
        const auto expected_base_background_activity = urd_desired_base_background_activity(mt);
        const auto expected_background_activity_mean = urd_desired_background_activity_mean(mt);
        const auto expected_background_activity_stddev = urd_desired_background_activity_stddev(mt);

        const auto expected_x0 = urd_desired_x0(mt);
        const auto expected_refrac = urd_desired_refrac(mt);
        const auto expected_tau_x = urd_desired_tau_x(mt);

        auto model = std::make_unique<PoissonModel>(expected_k, expected_tau_C, expected_beta, expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
            expected_x0, expected_tau_x, expected_refrac);

        EXPECT_EQ(expected_k, model->get_k());
        EXPECT_EQ(expected_tau_C, model->get_tau_C());
        EXPECT_EQ(expected_beta, model->get_beta());
        EXPECT_EQ(expected_h, model->get_h());
        EXPECT_EQ(expected_base_background_activity, model->get_base_background_activity());
        EXPECT_EQ(expected_background_activity_mean, model->get_background_activity_mean());
        EXPECT_EQ(expected_background_activity_stddev, model->get_background_activity_stddev());

        EXPECT_EQ(expected_x0, model->get_x_0());
        EXPECT_EQ(expected_refrac, model->get_refrac_time());
        EXPECT_EQ(expected_tau_x, model->get_tau_x());
    }

    for (auto i = 0; i < iterations; i++) {
        const urd urd_desired_k(NeuronModels::min_k, NeuronModels::max_k);
        const urd urd_desired_tau_C(NeuronModels::min_tau_C, NeuronModels::max_tau_C);
        const urd urd_desired_beta(NeuronModels::min_beta, NeuronModels::max_beta);
        const uid uid_desired_h(NeuronModels::min_h, NeuronModels::max_h);
        const urd urd_desired_base_background_activity(NeuronModels::min_base_background_activity, NeuronModels::max_base_background_activity);
        const urd urd_desired_background_activity_mean(NeuronModels::min_background_activity_mean, NeuronModels::max_background_activity_mean);
        const urd urd_desired_background_activity_stddev(NeuronModels::min_background_activity_stddev, NeuronModels::max_background_activity_stddev);

        const urd urd_desired_a(IzhikevichModel::min_a, IzhikevichModel::max_a);
        const urd urd_desired_b(IzhikevichModel::min_b, IzhikevichModel::max_b);
        const urd urd_desired_c(IzhikevichModel::min_c, IzhikevichModel::max_c);
        const urd urd_desired_d(IzhikevichModel::min_d, IzhikevichModel::max_d);
        const urd urd_desired_V_spike(IzhikevichModel::min_V_spike, IzhikevichModel::max_V_spike);
        const urd urd_desired_k1(IzhikevichModel::min_k1, IzhikevichModel::max_k1);
        const urd urd_desired_k2(IzhikevichModel::min_k2, IzhikevichModel::max_k2);
        const urd urd_desired_k3(IzhikevichModel::min_k3, IzhikevichModel::max_k3);

        const auto expected_k = urd_desired_k(mt);
        const auto expected_tau_C = urd_desired_tau_C(mt);
        const auto expected_beta = urd_desired_beta(mt);
        const auto expected_h = uid_desired_h(mt);
        const auto expected_base_background_activity = urd_desired_base_background_activity(mt);
        const auto expected_background_activity_mean = urd_desired_background_activity_mean(mt);
        const auto expected_background_activity_stddev = urd_desired_background_activity_stddev(mt);

        const auto expected_a = urd_desired_a(mt);
        const auto expected_b = urd_desired_b(mt);
        const auto expected_c = urd_desired_c(mt);
        const auto expected_d = urd_desired_d(mt);
        const auto expected_V_spike = urd_desired_V_spike(mt);
        const auto expected_k1 = urd_desired_k1(mt);
        const auto expected_k2 = urd_desired_k2(mt);
        const auto expected_k3 = urd_desired_k3(mt);

        auto model = std::make_unique<IzhikevichModel>(expected_k, expected_tau_C, expected_beta, expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
            expected_a, expected_b, expected_c, expected_d, expected_V_spike, expected_k1, expected_k2, expected_k3);

        EXPECT_EQ(expected_k, model->get_k());
        EXPECT_EQ(expected_tau_C, model->get_tau_C());
        EXPECT_EQ(expected_beta, model->get_beta());
        EXPECT_EQ(expected_h, model->get_h());
        EXPECT_EQ(expected_base_background_activity, model->get_base_background_activity());
        EXPECT_EQ(expected_background_activity_mean, model->get_background_activity_mean());
        EXPECT_EQ(expected_background_activity_stddev, model->get_background_activity_stddev());

        EXPECT_EQ(expected_a, model->get_a());
        EXPECT_EQ(expected_b, model->get_b());
        EXPECT_EQ(expected_c, model->get_c());
        EXPECT_EQ(expected_d, model->get_d());
        EXPECT_EQ(expected_V_spike, model->get_V_spike());
        EXPECT_EQ(expected_k1, model->get_k1());
        EXPECT_EQ(expected_k2, model->get_k2());
        EXPECT_EQ(expected_k3, model->get_k3());
    }

    for (auto i = 0; i < iterations; i++) {
        const urd urd_desired_k(NeuronModels::min_k, NeuronModels::max_k);
        const urd urd_desired_tau_C(NeuronModels::min_tau_C, NeuronModels::max_tau_C);
        const urd urd_desired_beta(NeuronModels::min_beta, NeuronModels::max_beta);
        const uid uid_desired_h(NeuronModels::min_h, NeuronModels::max_h);
        const urd urd_desired_base_background_activity(NeuronModels::min_base_background_activity, NeuronModels::max_base_background_activity);
        const urd urd_desired_background_activity_mean(NeuronModels::min_background_activity_mean, NeuronModels::max_background_activity_mean);
        const urd urd_desired_background_activity_stddev(NeuronModels::min_background_activity_stddev, NeuronModels::max_background_activity_stddev);

        const urd urd_desired_a(FitzHughNagumoModel::min_a, FitzHughNagumoModel::max_a);
        const urd urd_desired_b(FitzHughNagumoModel::min_b, FitzHughNagumoModel::max_b);
        const urd urd_desired_phi(FitzHughNagumoModel::min_phi, FitzHughNagumoModel::max_phi);

        const auto expected_k = urd_desired_k(mt);
        const auto expected_tau_C = urd_desired_tau_C(mt);
        const auto expected_beta = urd_desired_beta(mt);
        const auto expected_h = uid_desired_h(mt);
        const auto expected_base_background_activity = urd_desired_base_background_activity(mt);
        const auto expected_background_activity_mean = urd_desired_background_activity_mean(mt);
        const auto expected_background_activity_stddev = urd_desired_background_activity_stddev(mt);

        const auto expected_a = urd_desired_a(mt);
        const auto expected_b = urd_desired_b(mt);
        const auto expected_phi = urd_desired_phi(mt);

        auto model = std::make_unique<FitzHughNagumoModel>(expected_k, expected_tau_C, expected_beta, expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
            expected_a, expected_b, expected_phi);

        EXPECT_EQ(expected_k, model->get_k());
        EXPECT_EQ(expected_tau_C, model->get_tau_C());
        EXPECT_EQ(expected_beta, model->get_beta());
        EXPECT_EQ(expected_h, model->get_h());
        EXPECT_EQ(expected_base_background_activity, model->get_base_background_activity());
        EXPECT_EQ(expected_background_activity_mean, model->get_background_activity_mean());
        EXPECT_EQ(expected_background_activity_stddev, model->get_background_activity_stddev());

        EXPECT_EQ(expected_a, model->get_a());
        EXPECT_EQ(expected_b, model->get_b());
        EXPECT_EQ(expected_phi, model->get_phi());
    }

    for (auto i = 0; i < iterations; i++) {
        const urd urd_desired_k(NeuronModels::min_k, NeuronModels::max_k);
        const urd urd_desired_tau_C(NeuronModels::min_tau_C, NeuronModels::max_tau_C);
        const urd urd_desired_beta(NeuronModels::min_beta, NeuronModels::max_beta);
        const uid uid_desired_h(NeuronModels::min_h, NeuronModels::max_h);
        const urd urd_desired_base_background_activity(NeuronModels::min_base_background_activity, NeuronModels::max_base_background_activity);
        const urd urd_desired_background_activity_mean(NeuronModels::min_background_activity_mean, NeuronModels::max_background_activity_mean);
        const urd urd_desired_background_activity_stddev(NeuronModels::min_background_activity_stddev, NeuronModels::max_background_activity_stddev);

        const urd urd_desired_C(AEIFModel::min_C, AEIFModel::max_C);
        const urd urd_desired_g_L(AEIFModel::min_g_L, AEIFModel::max_g_L);
        const urd urd_desired_E_L(AEIFModel::min_E_L, AEIFModel::max_E_L);
        const urd urd_desired_V_T(AEIFModel::min_V_T, AEIFModel::max_V_T);
        const urd urd_desired_d_T(AEIFModel::min_d_T, AEIFModel::max_d_T);
        const urd urd_desired_tau_w(AEIFModel::min_tau_w, AEIFModel::max_tau_w);
        const urd urd_desired_a(AEIFModel::min_a, AEIFModel::max_a);
        const urd urd_desired_b(AEIFModel::min_b, AEIFModel::max_b);
        const urd urd_desired_V_peak(AEIFModel::min_V_peak, AEIFModel::max_V_peak);

        const auto expected_k = urd_desired_k(mt);
        const auto expected_tau_C = urd_desired_tau_C(mt);
        const auto expected_beta = urd_desired_beta(mt);
        const auto expected_h = uid_desired_h(mt);
        const auto expected_base_background_activity = urd_desired_base_background_activity(mt);
        const auto expected_background_activity_mean = urd_desired_background_activity_mean(mt);
        const auto expected_background_activity_stddev = urd_desired_background_activity_stddev(mt);

        const auto expected_C = urd_desired_C(mt);
        const auto expected_g_L = urd_desired_g_L(mt);
        const auto expected_E_L = urd_desired_E_L(mt);
        const auto expected_V_T = urd_desired_V_T(mt);
        const auto expected_d_T = urd_desired_d_T(mt);
        const auto expected_tau_w = urd_desired_tau_w(mt);
        const auto expected_a = urd_desired_a(mt);
        const auto expected_b = urd_desired_b(mt);
        const auto expected_V_peak = urd_desired_V_peak(mt);

        auto model = std::make_unique<AEIFModel>(expected_k, expected_tau_C, expected_beta, expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
            expected_C, expected_g_L, expected_E_L, expected_V_T, expected_d_T, expected_tau_w, expected_a, expected_b, expected_V_peak);

        EXPECT_EQ(expected_k, model->get_k());
        EXPECT_EQ(expected_tau_C, model->get_tau_C());
        EXPECT_EQ(expected_beta, model->get_beta());
        EXPECT_EQ(expected_h, model->get_h());
        EXPECT_EQ(expected_base_background_activity, model->get_base_background_activity());
        EXPECT_EQ(expected_background_activity_mean, model->get_background_activity_mean());
        EXPECT_EQ(expected_background_activity_stddev, model->get_background_activity_stddev());

        EXPECT_EQ(expected_C, model->get_C());
        EXPECT_EQ(expected_g_L, model->get_g_L());
        EXPECT_EQ(expected_E_L, model->get_E_L());
        EXPECT_EQ(expected_V_T, model->get_V_T());
        EXPECT_EQ(expected_d_T, model->get_d_T());
        EXPECT_EQ(expected_tau_w, model->get_tau_w());
        EXPECT_EQ(expected_a, model->get_a());
        EXPECT_EQ(expected_b, model->get_b());
        EXPECT_EQ(expected_V_peak, model->get_V_Peak());
    }
}

TEST(TestNeuronModels, testNeuronModelsClone) {
    using namespace models;
    using urd = std::uniform_real_distribution<double>;
    using uid = std::uniform_int_distribution<unsigned int>;

    setup();

    for (auto i = 0; i < iterations; i++) {
        const urd urd_desired_k(NeuronModels::min_k, NeuronModels::max_k);
        const urd urd_desired_tau_C(NeuronModels::min_tau_C, NeuronModels::max_tau_C);
        const urd urd_desired_beta(NeuronModels::min_beta, NeuronModels::max_beta);
        const uid uid_desired_h(NeuronModels::min_h, NeuronModels::max_h);
        const urd urd_desired_base_background_activity(NeuronModels::min_base_background_activity, NeuronModels::max_base_background_activity);
        const urd urd_desired_background_activity_mean(NeuronModels::min_background_activity_mean, NeuronModels::max_background_activity_mean);
        const urd urd_desired_background_activity_stddev(NeuronModels::min_background_activity_stddev, NeuronModels::max_background_activity_stddev);

        const urd urd_desired_x0(PoissonModel::min_x_0, PoissonModel::max_x_0);
        const uid urd_desired_refrac(PoissonModel::min_refrac_time, PoissonModel::max_refrac_time);
        const urd urd_desired_tau_x(PoissonModel::min_tau_x, PoissonModel::max_tau_x);

        const auto expected_k = urd_desired_k(mt);
        const auto expected_tau_C = urd_desired_tau_C(mt);
        const auto expected_beta = urd_desired_beta(mt);
        const auto expected_h = uid_desired_h(mt);
        const auto expected_base_background_activity = urd_desired_base_background_activity(mt);
        const auto expected_background_activity_mean = urd_desired_background_activity_mean(mt);
        const auto expected_background_activity_stddev = urd_desired_background_activity_stddev(mt);

        const auto expected_x0 = urd_desired_x0(mt);
        const auto expected_refrac = urd_desired_refrac(mt);
        const auto expected_tau_x = urd_desired_tau_x(mt);

        auto model = std::make_unique<PoissonModel>(expected_k, expected_tau_C, expected_beta, expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
            expected_x0, expected_tau_x, expected_refrac);

        auto cloned_model = model->clone();
        std::shared_ptr<NeuronModels> shared_version = std::move(cloned_model);
        auto cast_cloned_model = std::dynamic_pointer_cast<PoissonModel>(shared_version);

        EXPECT_EQ(expected_k, model->get_k());
        EXPECT_EQ(expected_tau_C, model->get_tau_C());
        EXPECT_EQ(expected_beta, model->get_beta());
        EXPECT_EQ(expected_h, model->get_h());
        EXPECT_EQ(expected_base_background_activity, model->get_base_background_activity());
        EXPECT_EQ(expected_background_activity_mean, model->get_background_activity_mean());
        EXPECT_EQ(expected_background_activity_stddev, model->get_background_activity_stddev());

        EXPECT_EQ(expected_x0, model->get_x_0());
        EXPECT_EQ(expected_refrac, model->get_refrac_time());
        EXPECT_EQ(expected_tau_x, model->get_tau_x());

        EXPECT_EQ(expected_k, cast_cloned_model->get_k());
        EXPECT_EQ(expected_tau_C, cast_cloned_model->get_tau_C());
        EXPECT_EQ(expected_beta, cast_cloned_model->get_beta());
        EXPECT_EQ(expected_h, cast_cloned_model->get_h());
        EXPECT_EQ(expected_base_background_activity, cast_cloned_model->get_base_background_activity());
        EXPECT_EQ(expected_background_activity_mean, cast_cloned_model->get_background_activity_mean());
        EXPECT_EQ(expected_background_activity_stddev, cast_cloned_model->get_background_activity_stddev());

        EXPECT_EQ(expected_x0, cast_cloned_model->get_x_0());
        EXPECT_EQ(expected_refrac, cast_cloned_model->get_refrac_time());
        EXPECT_EQ(expected_tau_x, cast_cloned_model->get_tau_x());
    }

    for (auto i = 0; i < iterations; i++) {
        const urd urd_desired_k(NeuronModels::min_k, NeuronModels::max_k);
        const urd urd_desired_tau_C(NeuronModels::min_tau_C, NeuronModels::max_tau_C);
        const urd urd_desired_beta(NeuronModels::min_beta, NeuronModels::max_beta);
        const uid uid_desired_h(NeuronModels::min_h, NeuronModels::max_h);
        const urd urd_desired_base_background_activity(NeuronModels::min_base_background_activity, NeuronModels::max_base_background_activity);
        const urd urd_desired_background_activity_mean(NeuronModels::min_background_activity_mean, NeuronModels::max_background_activity_mean);
        const urd urd_desired_background_activity_stddev(NeuronModels::min_background_activity_stddev, NeuronModels::max_background_activity_stddev);

        const urd urd_desired_a(IzhikevichModel::min_a, IzhikevichModel::max_a);
        const urd urd_desired_b(IzhikevichModel::min_b, IzhikevichModel::max_b);
        const urd urd_desired_c(IzhikevichModel::min_c, IzhikevichModel::max_c);
        const urd urd_desired_d(IzhikevichModel::min_d, IzhikevichModel::max_d);
        const urd urd_desired_V_spike(IzhikevichModel::min_V_spike, IzhikevichModel::max_V_spike);
        const urd urd_desired_k1(IzhikevichModel::min_k1, IzhikevichModel::max_k1);
        const urd urd_desired_k2(IzhikevichModel::min_k2, IzhikevichModel::max_k2);
        const urd urd_desired_k3(IzhikevichModel::min_k3, IzhikevichModel::max_k3);

        const auto expected_k = urd_desired_k(mt);
        const auto expected_tau_C = urd_desired_tau_C(mt);
        const auto expected_beta = urd_desired_beta(mt);
        const auto expected_h = uid_desired_h(mt);
        const auto expected_base_background_activity = urd_desired_base_background_activity(mt);
        const auto expected_background_activity_mean = urd_desired_background_activity_mean(mt);
        const auto expected_background_activity_stddev = urd_desired_background_activity_stddev(mt);

        const auto expected_a = urd_desired_a(mt);
        const auto expected_b = urd_desired_b(mt);
        const auto expected_c = urd_desired_c(mt);
        const auto expected_d = urd_desired_d(mt);
        const auto expected_V_spike = urd_desired_V_spike(mt);
        const auto expected_k1 = urd_desired_k1(mt);
        const auto expected_k2 = urd_desired_k2(mt);
        const auto expected_k3 = urd_desired_k3(mt);

        auto model = std::make_unique<IzhikevichModel>(expected_k, expected_tau_C, expected_beta, expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
            expected_a, expected_b, expected_c, expected_d, expected_V_spike, expected_k1, expected_k2, expected_k3);

        auto cloned_model = model->clone();
        std::shared_ptr<NeuronModels> shared_version = std::move(cloned_model);
        auto cast_cloned_model = std::dynamic_pointer_cast<IzhikevichModel>(shared_version);

        EXPECT_EQ(expected_k, model->get_k());
        EXPECT_EQ(expected_tau_C, model->get_tau_C());
        EXPECT_EQ(expected_beta, model->get_beta());
        EXPECT_EQ(expected_h, model->get_h());
        EXPECT_EQ(expected_base_background_activity, model->get_base_background_activity());
        EXPECT_EQ(expected_background_activity_mean, model->get_background_activity_mean());
        EXPECT_EQ(expected_background_activity_stddev, model->get_background_activity_stddev());

        EXPECT_EQ(expected_a, model->get_a());
        EXPECT_EQ(expected_b, model->get_b());
        EXPECT_EQ(expected_c, model->get_c());
        EXPECT_EQ(expected_d, model->get_d());
        EXPECT_EQ(expected_V_spike, model->get_V_spike());
        EXPECT_EQ(expected_k1, model->get_k1());
        EXPECT_EQ(expected_k2, model->get_k2());
        EXPECT_EQ(expected_k3, model->get_k3());

        EXPECT_EQ(expected_k, cast_cloned_model->get_k());
        EXPECT_EQ(expected_tau_C, cast_cloned_model->get_tau_C());
        EXPECT_EQ(expected_beta, cast_cloned_model->get_beta());
        EXPECT_EQ(expected_h, cast_cloned_model->get_h());
        EXPECT_EQ(expected_base_background_activity, cast_cloned_model->get_base_background_activity());
        EXPECT_EQ(expected_background_activity_mean, cast_cloned_model->get_background_activity_mean());
        EXPECT_EQ(expected_background_activity_stddev, cast_cloned_model->get_background_activity_stddev());

        EXPECT_EQ(expected_a, cast_cloned_model->get_a());
        EXPECT_EQ(expected_b, cast_cloned_model->get_b());
        EXPECT_EQ(expected_c, cast_cloned_model->get_c());
        EXPECT_EQ(expected_d, cast_cloned_model->get_d());
        EXPECT_EQ(expected_V_spike, cast_cloned_model->get_V_spike());
        EXPECT_EQ(expected_k1, cast_cloned_model->get_k1());
        EXPECT_EQ(expected_k2, cast_cloned_model->get_k2());
        EXPECT_EQ(expected_k3, cast_cloned_model->get_k3());
    }

    for (auto i = 0; i < iterations; i++) {
        const urd urd_desired_k(NeuronModels::min_k, NeuronModels::max_k);
        const urd urd_desired_tau_C(NeuronModels::min_tau_C, NeuronModels::max_tau_C);
        const urd urd_desired_beta(NeuronModels::min_beta, NeuronModels::max_beta);
        const uid uid_desired_h(NeuronModels::min_h, NeuronModels::max_h);
        const urd urd_desired_base_background_activity(NeuronModels::min_base_background_activity, NeuronModels::max_base_background_activity);
        const urd urd_desired_background_activity_mean(NeuronModels::min_background_activity_mean, NeuronModels::max_background_activity_mean);
        const urd urd_desired_background_activity_stddev(NeuronModels::min_background_activity_stddev, NeuronModels::max_background_activity_stddev);

        const urd urd_desired_a(FitzHughNagumoModel::min_a, FitzHughNagumoModel::max_a);
        const urd urd_desired_b(FitzHughNagumoModel::min_b, FitzHughNagumoModel::max_b);
        const urd urd_desired_phi(FitzHughNagumoModel::min_phi, FitzHughNagumoModel::max_phi);

        const auto expected_k = urd_desired_k(mt);
        const auto expected_tau_C = urd_desired_tau_C(mt);
        const auto expected_beta = urd_desired_beta(mt);
        const auto expected_h = uid_desired_h(mt);
        const auto expected_base_background_activity = urd_desired_base_background_activity(mt);
        const auto expected_background_activity_mean = urd_desired_background_activity_mean(mt);
        const auto expected_background_activity_stddev = urd_desired_background_activity_stddev(mt);

        const auto expected_a = urd_desired_a(mt);
        const auto expected_b = urd_desired_b(mt);
        const auto expected_phi = urd_desired_phi(mt);

        auto model = std::make_unique<FitzHughNagumoModel>(expected_k, expected_tau_C, expected_beta, expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
            expected_a, expected_b, expected_phi);

        auto cloned_model = model->clone();
        std::shared_ptr<NeuronModels> shared_version = std::move(cloned_model);
        auto cast_cloned_model = std::dynamic_pointer_cast<FitzHughNagumoModel>(shared_version);

        EXPECT_EQ(expected_k, model->get_k());
        EXPECT_EQ(expected_tau_C, model->get_tau_C());
        EXPECT_EQ(expected_beta, model->get_beta());
        EXPECT_EQ(expected_h, model->get_h());
        EXPECT_EQ(expected_base_background_activity, model->get_base_background_activity());
        EXPECT_EQ(expected_background_activity_mean, model->get_background_activity_mean());
        EXPECT_EQ(expected_background_activity_stddev, model->get_background_activity_stddev());

        EXPECT_EQ(expected_a, model->get_a());
        EXPECT_EQ(expected_b, model->get_b());
        EXPECT_EQ(expected_phi, model->get_phi());

        EXPECT_EQ(expected_k, cast_cloned_model->get_k());
        EXPECT_EQ(expected_tau_C, cast_cloned_model->get_tau_C());
        EXPECT_EQ(expected_beta, cast_cloned_model->get_beta());
        EXPECT_EQ(expected_h, cast_cloned_model->get_h());
        EXPECT_EQ(expected_base_background_activity, cast_cloned_model->get_base_background_activity());
        EXPECT_EQ(expected_background_activity_mean, cast_cloned_model->get_background_activity_mean());
        EXPECT_EQ(expected_background_activity_stddev, cast_cloned_model->get_background_activity_stddev());

        EXPECT_EQ(expected_a, cast_cloned_model->get_a());
        EXPECT_EQ(expected_b, cast_cloned_model->get_b());
        EXPECT_EQ(expected_phi, cast_cloned_model->get_phi());
    }

    for (auto i = 0; i < iterations; i++) {
        const urd urd_desired_k(NeuronModels::min_k, NeuronModels::max_k);
        const urd urd_desired_tau_C(NeuronModels::min_tau_C, NeuronModels::max_tau_C);
        const urd urd_desired_beta(NeuronModels::min_beta, NeuronModels::max_beta);
        const uid uid_desired_h(NeuronModels::min_h, NeuronModels::max_h);
        const urd urd_desired_base_background_activity(NeuronModels::min_base_background_activity, NeuronModels::max_base_background_activity);
        const urd urd_desired_background_activity_mean(NeuronModels::min_background_activity_mean, NeuronModels::max_background_activity_mean);
        const urd urd_desired_background_activity_stddev(NeuronModels::min_background_activity_stddev, NeuronModels::max_background_activity_stddev);

        const urd urd_desired_C(AEIFModel::min_C, AEIFModel::max_C);
        const urd urd_desired_g_L(AEIFModel::min_g_L, AEIFModel::max_g_L);
        const urd urd_desired_E_L(AEIFModel::min_E_L, AEIFModel::max_E_L);
        const urd urd_desired_V_T(AEIFModel::min_V_T, AEIFModel::max_V_T);
        const urd urd_desired_d_T(AEIFModel::min_d_T, AEIFModel::max_d_T);
        const urd urd_desired_tau_w(AEIFModel::min_tau_w, AEIFModel::max_tau_w);
        const urd urd_desired_a(AEIFModel::min_a, AEIFModel::max_a);
        const urd urd_desired_b(AEIFModel::min_b, AEIFModel::max_b);
        const urd urd_desired_V_peak(AEIFModel::min_V_peak, AEIFModel::max_V_peak);

        const auto expected_k = urd_desired_k(mt);
        const auto expected_tau_C = urd_desired_tau_C(mt);
        const auto expected_beta = urd_desired_beta(mt);
        const auto expected_h = uid_desired_h(mt);
        const auto expected_base_background_activity = urd_desired_base_background_activity(mt);
        const auto expected_background_activity_mean = urd_desired_background_activity_mean(mt);
        const auto expected_background_activity_stddev = urd_desired_background_activity_stddev(mt);

        const auto expected_C = urd_desired_C(mt);
        const auto expected_g_L = urd_desired_g_L(mt);
        const auto expected_E_L = urd_desired_E_L(mt);
        const auto expected_V_T = urd_desired_V_T(mt);
        const auto expected_d_T = urd_desired_d_T(mt);
        const auto expected_tau_w = urd_desired_tau_w(mt);
        const auto expected_a = urd_desired_a(mt);
        const auto expected_b = urd_desired_b(mt);
        const auto expected_V_peak = urd_desired_V_peak(mt);

        auto model = std::make_unique<AEIFModel>(expected_k, expected_tau_C, expected_beta, expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
            expected_C, expected_g_L, expected_E_L, expected_V_T, expected_d_T, expected_tau_w, expected_a, expected_b, expected_V_peak);

        auto cloned_model = model->clone();
        std::shared_ptr<NeuronModels> shared_version = std::move(cloned_model);
        auto cast_cloned_model = std::dynamic_pointer_cast<AEIFModel>(shared_version);

        EXPECT_EQ(expected_k, model->get_k());
        EXPECT_EQ(expected_tau_C, model->get_tau_C());
        EXPECT_EQ(expected_beta, model->get_beta());
        EXPECT_EQ(expected_h, model->get_h());
        EXPECT_EQ(expected_base_background_activity, model->get_base_background_activity());
        EXPECT_EQ(expected_background_activity_mean, model->get_background_activity_mean());
        EXPECT_EQ(expected_background_activity_stddev, model->get_background_activity_stddev());

        EXPECT_EQ(expected_C, model->get_C());
        EXPECT_EQ(expected_g_L, model->get_g_L());
        EXPECT_EQ(expected_E_L, model->get_E_L());
        EXPECT_EQ(expected_V_T, model->get_V_T());
        EXPECT_EQ(expected_d_T, model->get_d_T());
        EXPECT_EQ(expected_tau_w, model->get_tau_w());
        EXPECT_EQ(expected_a, model->get_a());
        EXPECT_EQ(expected_b, model->get_b());
        EXPECT_EQ(expected_V_peak, model->get_V_Peak());

        EXPECT_EQ(expected_k, cast_cloned_model->get_k());
        EXPECT_EQ(expected_tau_C, cast_cloned_model->get_tau_C());
        EXPECT_EQ(expected_beta, cast_cloned_model->get_beta());
        EXPECT_EQ(expected_h, cast_cloned_model->get_h());
        EXPECT_EQ(expected_base_background_activity, cast_cloned_model->get_base_background_activity());
        EXPECT_EQ(expected_background_activity_mean, cast_cloned_model->get_background_activity_mean());
        EXPECT_EQ(expected_background_activity_stddev, cast_cloned_model->get_background_activity_stddev());

        EXPECT_EQ(expected_C, cast_cloned_model->get_C());
        EXPECT_EQ(expected_g_L, cast_cloned_model->get_g_L());
        EXPECT_EQ(expected_E_L, cast_cloned_model->get_E_L());
        EXPECT_EQ(expected_V_T, cast_cloned_model->get_V_T());
        EXPECT_EQ(expected_d_T, cast_cloned_model->get_d_T());
        EXPECT_EQ(expected_tau_w, cast_cloned_model->get_tau_w());
        EXPECT_EQ(expected_a, cast_cloned_model->get_a());
        EXPECT_EQ(expected_b, cast_cloned_model->get_b());
        EXPECT_EQ(expected_V_peak, cast_cloned_model->get_V_Peak());
    }
}

TEST(TestNeuronModels, testNeuronModelsCreate) {
    using namespace models;
    using urd = std::uniform_real_distribution<double>;
    using uid = std::uniform_int_distribution<unsigned int>;

    setup();

    for (auto i = 0; i < iterations; i++) {
        const urd urd_desired_k(NeuronModels::min_k, NeuronModels::max_k);
        const urd urd_desired_tau_C(NeuronModels::min_tau_C, NeuronModels::max_tau_C);
        const urd urd_desired_beta(NeuronModels::min_beta, NeuronModels::max_beta);
        const uid uid_desired_h(NeuronModels::min_h, NeuronModels::max_h);
        const urd urd_desired_base_background_activity(NeuronModels::min_base_background_activity, NeuronModels::max_base_background_activity);
        const urd urd_desired_background_activity_mean(NeuronModels::min_background_activity_mean, NeuronModels::max_background_activity_mean);
        const urd urd_desired_background_activity_stddev(NeuronModels::min_background_activity_stddev, NeuronModels::max_background_activity_stddev);

        const urd urd_desired_x0(PoissonModel::min_x_0, PoissonModel::max_x_0);
        const uid urd_desired_refrac(PoissonModel::min_refrac_time, PoissonModel::max_refrac_time);
        const urd urd_desired_tau_x(PoissonModel::min_tau_x, PoissonModel::max_tau_x);

        const auto expected_k = urd_desired_k(mt);
        const auto expected_tau_C = urd_desired_tau_C(mt);
        const auto expected_beta = urd_desired_beta(mt);
        const auto expected_h = uid_desired_h(mt);
        const auto expected_base_background_activity = urd_desired_base_background_activity(mt);
        const auto expected_background_activity_mean = urd_desired_background_activity_mean(mt);
        const auto expected_background_activity_stddev = urd_desired_background_activity_stddev(mt);

        const auto expected_x0 = urd_desired_x0(mt);
        const auto expected_refrac = urd_desired_refrac(mt);
        const auto expected_tau_x = urd_desired_tau_x(mt);

        auto model = NeuronModels::create<PoissonModel>(expected_k, expected_tau_C, expected_beta, expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
            expected_x0, expected_tau_x, expected_refrac);

        EXPECT_EQ(expected_k, model->get_k());
        EXPECT_EQ(expected_tau_C, model->get_tau_C());
        EXPECT_EQ(expected_beta, model->get_beta());
        EXPECT_EQ(expected_h, model->get_h());
        EXPECT_EQ(expected_base_background_activity, model->get_base_background_activity());
        EXPECT_EQ(expected_background_activity_mean, model->get_background_activity_mean());
        EXPECT_EQ(expected_background_activity_stddev, model->get_background_activity_stddev());

        EXPECT_EQ(expected_x0, model->get_x_0());
        EXPECT_EQ(expected_refrac, model->get_refrac_time());
        EXPECT_EQ(expected_tau_x, model->get_tau_x());
    }

    for (auto i = 0; i < iterations; i++) {
        const urd urd_desired_k(NeuronModels::min_k, NeuronModels::max_k);
        const urd urd_desired_tau_C(NeuronModels::min_tau_C, NeuronModels::max_tau_C);
        const urd urd_desired_beta(NeuronModels::min_beta, NeuronModels::max_beta);
        const uid uid_desired_h(NeuronModels::min_h, NeuronModels::max_h);
        const urd urd_desired_base_background_activity(NeuronModels::min_base_background_activity, NeuronModels::max_base_background_activity);
        const urd urd_desired_background_activity_mean(NeuronModels::min_background_activity_mean, NeuronModels::max_background_activity_mean);
        const urd urd_desired_background_activity_stddev(NeuronModels::min_background_activity_stddev, NeuronModels::max_background_activity_stddev);

        const urd urd_desired_a(IzhikevichModel::min_a, IzhikevichModel::max_a);
        const urd urd_desired_b(IzhikevichModel::min_b, IzhikevichModel::max_b);
        const urd urd_desired_c(IzhikevichModel::min_c, IzhikevichModel::max_c);
        const urd urd_desired_d(IzhikevichModel::min_d, IzhikevichModel::max_d);
        const urd urd_desired_V_spike(IzhikevichModel::min_V_spike, IzhikevichModel::max_V_spike);
        const urd urd_desired_k1(IzhikevichModel::min_k1, IzhikevichModel::max_k1);
        const urd urd_desired_k2(IzhikevichModel::min_k2, IzhikevichModel::max_k2);
        const urd urd_desired_k3(IzhikevichModel::min_k3, IzhikevichModel::max_k3);

        const auto expected_k = urd_desired_k(mt);
        const auto expected_tau_C = urd_desired_tau_C(mt);
        const auto expected_beta = urd_desired_beta(mt);
        const auto expected_h = uid_desired_h(mt);
        const auto expected_base_background_activity = urd_desired_base_background_activity(mt);
        const auto expected_background_activity_mean = urd_desired_background_activity_mean(mt);
        const auto expected_background_activity_stddev = urd_desired_background_activity_stddev(mt);

        const auto expected_a = urd_desired_a(mt);
        const auto expected_b = urd_desired_b(mt);
        const auto expected_c = urd_desired_c(mt);
        const auto expected_d = urd_desired_d(mt);
        const auto expected_V_spike = urd_desired_V_spike(mt);
        const auto expected_k1 = urd_desired_k1(mt);
        const auto expected_k2 = urd_desired_k2(mt);
        const auto expected_k3 = urd_desired_k3(mt);

        auto model = NeuronModels::create<IzhikevichModel>(expected_k, expected_tau_C, expected_beta, expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
            expected_a, expected_b, expected_c, expected_d, expected_V_spike, expected_k1, expected_k2, expected_k3);

        EXPECT_EQ(expected_k, model->get_k());
        EXPECT_EQ(expected_tau_C, model->get_tau_C());
        EXPECT_EQ(expected_beta, model->get_beta());
        EXPECT_EQ(expected_h, model->get_h());
        EXPECT_EQ(expected_base_background_activity, model->get_base_background_activity());
        EXPECT_EQ(expected_background_activity_mean, model->get_background_activity_mean());
        EXPECT_EQ(expected_background_activity_stddev, model->get_background_activity_stddev());

        EXPECT_EQ(expected_a, model->get_a());
        EXPECT_EQ(expected_b, model->get_b());
        EXPECT_EQ(expected_c, model->get_c());
        EXPECT_EQ(expected_d, model->get_d());
        EXPECT_EQ(expected_V_spike, model->get_V_spike());
        EXPECT_EQ(expected_k1, model->get_k1());
        EXPECT_EQ(expected_k2, model->get_k2());
        EXPECT_EQ(expected_k3, model->get_k3());
    }

    for (auto i = 0; i < iterations; i++) {
        const urd urd_desired_k(NeuronModels::min_k, NeuronModels::max_k);
        const urd urd_desired_tau_C(NeuronModels::min_tau_C, NeuronModels::max_tau_C);
        const urd urd_desired_beta(NeuronModels::min_beta, NeuronModels::max_beta);
        const uid uid_desired_h(NeuronModels::min_h, NeuronModels::max_h);
        const urd urd_desired_base_background_activity(NeuronModels::min_base_background_activity, NeuronModels::max_base_background_activity);
        const urd urd_desired_background_activity_mean(NeuronModels::min_background_activity_mean, NeuronModels::max_background_activity_mean);
        const urd urd_desired_background_activity_stddev(NeuronModels::min_background_activity_stddev, NeuronModels::max_background_activity_stddev);

        const urd urd_desired_a(FitzHughNagumoModel::min_a, FitzHughNagumoModel::max_a);
        const urd urd_desired_b(FitzHughNagumoModel::min_b, FitzHughNagumoModel::max_b);
        const urd urd_desired_phi(FitzHughNagumoModel::min_phi, FitzHughNagumoModel::max_phi);

        const auto expected_k = urd_desired_k(mt);
        const auto expected_tau_C = urd_desired_tau_C(mt);
        const auto expected_beta = urd_desired_beta(mt);
        const auto expected_h = uid_desired_h(mt);
        const auto expected_base_background_activity = urd_desired_base_background_activity(mt);
        const auto expected_background_activity_mean = urd_desired_background_activity_mean(mt);
        const auto expected_background_activity_stddev = urd_desired_background_activity_stddev(mt);

        const auto expected_a = urd_desired_a(mt);
        const auto expected_b = urd_desired_b(mt);
        const auto expected_phi = urd_desired_phi(mt);

        auto model = NeuronModels::create<FitzHughNagumoModel>(expected_k, expected_tau_C, expected_beta, expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
            expected_a, expected_b, expected_phi);

        EXPECT_EQ(expected_k, model->get_k());
        EXPECT_EQ(expected_tau_C, model->get_tau_C());
        EXPECT_EQ(expected_beta, model->get_beta());
        EXPECT_EQ(expected_h, model->get_h());
        EXPECT_EQ(expected_base_background_activity, model->get_base_background_activity());
        EXPECT_EQ(expected_background_activity_mean, model->get_background_activity_mean());
        EXPECT_EQ(expected_background_activity_stddev, model->get_background_activity_stddev());

        EXPECT_EQ(expected_a, model->get_a());
        EXPECT_EQ(expected_b, model->get_b());
        EXPECT_EQ(expected_phi, model->get_phi());
    }

    for (auto i = 0; i < iterations; i++) {
        const urd urd_desired_k(NeuronModels::min_k, NeuronModels::max_k);
        const urd urd_desired_tau_C(NeuronModels::min_tau_C, NeuronModels::max_tau_C);
        const urd urd_desired_beta(NeuronModels::min_beta, NeuronModels::max_beta);
        const uid uid_desired_h(NeuronModels::min_h, NeuronModels::max_h);
        const urd urd_desired_base_background_activity(NeuronModels::min_base_background_activity, NeuronModels::max_base_background_activity);
        const urd urd_desired_background_activity_mean(NeuronModels::min_background_activity_mean, NeuronModels::max_background_activity_mean);
        const urd urd_desired_background_activity_stddev(NeuronModels::min_background_activity_stddev, NeuronModels::max_background_activity_stddev);

        const urd urd_desired_C(AEIFModel::min_C, AEIFModel::max_C);
        const urd urd_desired_g_L(AEIFModel::min_g_L, AEIFModel::max_g_L);
        const urd urd_desired_E_L(AEIFModel::min_E_L, AEIFModel::max_E_L);
        const urd urd_desired_V_T(AEIFModel::min_V_T, AEIFModel::max_V_T);
        const urd urd_desired_d_T(AEIFModel::min_d_T, AEIFModel::max_d_T);
        const urd urd_desired_tau_w(AEIFModel::min_tau_w, AEIFModel::max_tau_w);
        const urd urd_desired_a(AEIFModel::min_a, AEIFModel::max_a);
        const urd urd_desired_b(AEIFModel::min_b, AEIFModel::max_b);
        const urd urd_desired_V_peak(AEIFModel::min_V_peak, AEIFModel::max_V_peak);

        const auto expected_k = urd_desired_k(mt);
        const auto expected_tau_C = urd_desired_tau_C(mt);
        const auto expected_beta = urd_desired_beta(mt);
        const auto expected_h = uid_desired_h(mt);
        const auto expected_base_background_activity = urd_desired_base_background_activity(mt);
        const auto expected_background_activity_mean = urd_desired_background_activity_mean(mt);
        const auto expected_background_activity_stddev = urd_desired_background_activity_stddev(mt);

        const auto expected_C = urd_desired_C(mt);
        const auto expected_g_L = urd_desired_g_L(mt);
        const auto expected_E_L = urd_desired_E_L(mt);
        const auto expected_V_T = urd_desired_V_T(mt);
        const auto expected_d_T = urd_desired_d_T(mt);
        const auto expected_tau_w = urd_desired_tau_w(mt);
        const auto expected_a = urd_desired_a(mt);
        const auto expected_b = urd_desired_b(mt);
        const auto expected_V_peak = urd_desired_V_peak(mt);

        auto model = NeuronModels::create<AEIFModel>(expected_k, expected_tau_C, expected_beta, expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
            expected_C, expected_g_L, expected_E_L, expected_V_T, expected_d_T, expected_tau_w, expected_a, expected_b, expected_V_peak);

        EXPECT_EQ(expected_k, model->get_k());
        EXPECT_EQ(expected_tau_C, model->get_tau_C());
        EXPECT_EQ(expected_beta, model->get_beta());
        EXPECT_EQ(expected_h, model->get_h());
        EXPECT_EQ(expected_base_background_activity, model->get_base_background_activity());
        EXPECT_EQ(expected_background_activity_mean, model->get_background_activity_mean());
        EXPECT_EQ(expected_background_activity_stddev, model->get_background_activity_stddev());

        EXPECT_EQ(expected_C, model->get_C());
        EXPECT_EQ(expected_g_L, model->get_g_L());
        EXPECT_EQ(expected_E_L, model->get_E_L());
        EXPECT_EQ(expected_V_T, model->get_V_T());
        EXPECT_EQ(expected_d_T, model->get_d_T());
        EXPECT_EQ(expected_tau_w, model->get_tau_w());
        EXPECT_EQ(expected_a, model->get_a());
        EXPECT_EQ(expected_b, model->get_b());
        EXPECT_EQ(expected_V_peak, model->get_V_Peak());
    }
}

TEST(TestNeuronModels, testNeuronModelsInit) {
    using namespace models;
    using urd = std::uniform_real_distribution<double>;
    using uid = std::uniform_int_distribution<unsigned int>;

    setup();

    for (auto i = 0; i < iterations; i++) {
        const urd urd_desired_k(NeuronModels::min_k, NeuronModels::max_k);
        const urd urd_desired_tau_C(NeuronModels::min_tau_C, NeuronModels::max_tau_C);
        const urd urd_desired_beta(NeuronModels::min_beta, NeuronModels::max_beta);
        const uid uid_desired_h(NeuronModels::min_h, NeuronModels::max_h);
        const urd urd_desired_base_background_activity(NeuronModels::min_base_background_activity, NeuronModels::max_base_background_activity);
        const urd urd_desired_background_activity_mean(NeuronModels::min_background_activity_mean, NeuronModels::max_background_activity_mean);
        const urd urd_desired_background_activity_stddev(NeuronModels::min_background_activity_stddev, NeuronModels::max_background_activity_stddev);

        const urd urd_desired_x0(PoissonModel::min_x_0, PoissonModel::max_x_0);
        const uid urd_desired_refrac(PoissonModel::min_refrac_time, PoissonModel::max_refrac_time);
        const urd urd_desired_tau_x(PoissonModel::min_tau_x, PoissonModel::max_tau_x);

        const auto expected_k = urd_desired_k(mt);
        const auto expected_tau_C = urd_desired_tau_C(mt);
        const auto expected_beta = urd_desired_beta(mt);
        const auto expected_h = uid_desired_h(mt);
        const auto expected_base_background_activity = urd_desired_base_background_activity(mt);
        const auto expected_background_activity_mean = urd_desired_background_activity_mean(mt);
        const auto expected_background_activity_stddev = urd_desired_background_activity_stddev(mt);

        const auto expected_x0 = urd_desired_x0(mt);
        const auto expected_refrac = urd_desired_refrac(mt);
        const auto expected_tau_x = urd_desired_tau_x(mt);

        auto model = std::make_unique<PoissonModel>(expected_k, expected_tau_C, expected_beta, expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
            expected_x0, expected_tau_x, expected_refrac);

        const uid uid_num_neurons(1, num_neurons_test);

        for (auto j = 0; j < iterations; j++) {
            const auto desired_num_neurons = uid_num_neurons(mt);

            model->init(desired_num_neurons);

            const auto num_neurons = model->get_num_neurons();

            const auto& x = model->get_x();
            const auto& fired = model->get_fired();

            EXPECT_EQ(desired_num_neurons, num_neurons);
            EXPECT_EQ(desired_num_neurons, x.size());
            EXPECT_EQ(desired_num_neurons, fired.size());

            for (size_t neuron_id = 0; neuron_id < num_neurons; neuron_id++) {
                EXPECT_NO_THROW(model->get_x(neuron_id));
                EXPECT_NO_THROW(model->get_fired(neuron_id));
                EXPECT_NO_THROW(model->get_secondary_variable(neuron_id));
                EXPECT_NO_THROW(model->get_I_syn(neuron_id));
            }

            for (size_t neuron_id = 0; neuron_id < num_neurons; neuron_id++) {
                EXPECT_THROW(model->get_x(neuron_id + num_neurons), RelearnException);
                EXPECT_THROW(model->get_fired(neuron_id + num_neurons), RelearnException);
                EXPECT_THROW(model->get_secondary_variable(neuron_id + num_neurons), RelearnException);
                EXPECT_THROW(model->get_I_syn(neuron_id + num_neurons), RelearnException);
            }
        }
    }

    for (auto i = 0; i < iterations; i++) {
        const urd urd_desired_k(NeuronModels::min_k, NeuronModels::max_k);
        const urd urd_desired_tau_C(NeuronModels::min_tau_C, NeuronModels::max_tau_C);
        const urd urd_desired_beta(NeuronModels::min_beta, NeuronModels::max_beta);
        const uid uid_desired_h(NeuronModels::min_h, NeuronModels::max_h);
        const urd urd_desired_base_background_activity(NeuronModels::min_base_background_activity, NeuronModels::max_base_background_activity);
        const urd urd_desired_background_activity_mean(NeuronModels::min_background_activity_mean, NeuronModels::max_background_activity_mean);
        const urd urd_desired_background_activity_stddev(NeuronModels::min_background_activity_stddev, NeuronModels::max_background_activity_stddev);

        const urd urd_desired_a(IzhikevichModel::min_a, IzhikevichModel::max_a);
        const urd urd_desired_b(IzhikevichModel::min_b, IzhikevichModel::max_b);
        const urd urd_desired_c(IzhikevichModel::min_c, IzhikevichModel::max_c);
        const urd urd_desired_d(IzhikevichModel::min_d, IzhikevichModel::max_d);
        const urd urd_desired_V_spike(IzhikevichModel::min_V_spike, IzhikevichModel::max_V_spike);
        const urd urd_desired_k1(IzhikevichModel::min_k1, IzhikevichModel::max_k1);
        const urd urd_desired_k2(IzhikevichModel::min_k2, IzhikevichModel::max_k2);
        const urd urd_desired_k3(IzhikevichModel::min_k3, IzhikevichModel::max_k3);

        const auto expected_k = urd_desired_k(mt);
        const auto expected_tau_C = urd_desired_tau_C(mt);
        const auto expected_beta = urd_desired_beta(mt);
        const auto expected_h = uid_desired_h(mt);
        const auto expected_base_background_activity = urd_desired_base_background_activity(mt);
        const auto expected_background_activity_mean = urd_desired_background_activity_mean(mt);
        const auto expected_background_activity_stddev = urd_desired_background_activity_stddev(mt);

        const auto expected_a = urd_desired_a(mt);
        const auto expected_b = urd_desired_b(mt);
        const auto expected_c = urd_desired_c(mt);
        const auto expected_d = urd_desired_d(mt);
        const auto expected_V_spike = urd_desired_V_spike(mt);
        const auto expected_k1 = urd_desired_k1(mt);
        const auto expected_k2 = urd_desired_k2(mt);
        const auto expected_k3 = urd_desired_k3(mt);

        auto model = std::make_unique<IzhikevichModel>(expected_k, expected_tau_C, expected_beta, expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
            expected_a, expected_b, expected_c, expected_d, expected_V_spike, expected_k1, expected_k2, expected_k3);

        const uid uid_num_neurons(1, num_neurons_test);

        for (auto j = 0; j < iterations; j++) {
            const auto desired_num_neurons = uid_num_neurons(mt);

            model->init(desired_num_neurons);

            const auto num_neurons = model->get_num_neurons();

            const auto& x = model->get_x();
            const auto& fired = model->get_fired();

            EXPECT_EQ(desired_num_neurons, num_neurons);
            EXPECT_EQ(desired_num_neurons, x.size());
            EXPECT_EQ(desired_num_neurons, fired.size());

            for (size_t neuron_id = 0; neuron_id < num_neurons; neuron_id++) {
                EXPECT_NO_THROW(model->get_x(neuron_id));
                EXPECT_NO_THROW(model->get_fired(neuron_id));
                EXPECT_NO_THROW(model->get_secondary_variable(neuron_id));
                EXPECT_NO_THROW(model->get_I_syn(neuron_id));
            }

            for (size_t neuron_id = 0; neuron_id < num_neurons; neuron_id++) {
                EXPECT_THROW(model->get_x(neuron_id + num_neurons), RelearnException);
                EXPECT_THROW(model->get_fired(neuron_id + num_neurons), RelearnException);
                EXPECT_THROW(model->get_secondary_variable(neuron_id + num_neurons), RelearnException);
                EXPECT_THROW(model->get_I_syn(neuron_id + num_neurons), RelearnException);
            }
        }
    }

    for (auto i = 0; i < iterations; i++) {
        const urd urd_desired_k(NeuronModels::min_k, NeuronModels::max_k);
        const urd urd_desired_tau_C(NeuronModels::min_tau_C, NeuronModels::max_tau_C);
        const urd urd_desired_beta(NeuronModels::min_beta, NeuronModels::max_beta);
        const uid uid_desired_h(NeuronModels::min_h, NeuronModels::max_h);
        const urd urd_desired_base_background_activity(NeuronModels::min_base_background_activity, NeuronModels::max_base_background_activity);
        const urd urd_desired_background_activity_mean(NeuronModels::min_background_activity_mean, NeuronModels::max_background_activity_mean);
        const urd urd_desired_background_activity_stddev(NeuronModels::min_background_activity_stddev, NeuronModels::max_background_activity_stddev);

        const urd urd_desired_a(FitzHughNagumoModel::min_a, FitzHughNagumoModel::max_a);
        const urd urd_desired_b(FitzHughNagumoModel::min_b, FitzHughNagumoModel::max_b);
        const urd urd_desired_phi(FitzHughNagumoModel::min_phi, FitzHughNagumoModel::max_phi);

        const auto expected_k = urd_desired_k(mt);
        const auto expected_tau_C = urd_desired_tau_C(mt);
        const auto expected_beta = urd_desired_beta(mt);
        const auto expected_h = uid_desired_h(mt);
        const auto expected_base_background_activity = urd_desired_base_background_activity(mt);
        const auto expected_background_activity_mean = urd_desired_background_activity_mean(mt);
        const auto expected_background_activity_stddev = urd_desired_background_activity_stddev(mt);

        const auto expected_a = urd_desired_a(mt);
        const auto expected_b = urd_desired_b(mt);
        const auto expected_phi = urd_desired_phi(mt);

        auto model = std::make_unique<FitzHughNagumoModel>(expected_k, expected_tau_C, expected_beta, expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
            expected_a, expected_b, expected_phi);

        const uid uid_num_neurons(1, num_neurons_test);

        for (auto j = 0; j < iterations; j++) {
            const auto desired_num_neurons = uid_num_neurons(mt);

            model->init(desired_num_neurons);

            const auto num_neurons = model->get_num_neurons();

            const auto& x = model->get_x();
            const auto& fired = model->get_fired();

            EXPECT_EQ(desired_num_neurons, num_neurons);
            EXPECT_EQ(desired_num_neurons, x.size());
            EXPECT_EQ(desired_num_neurons, fired.size());

            for (size_t neuron_id = 0; neuron_id < num_neurons; neuron_id++) {
                EXPECT_NO_THROW(model->get_x(neuron_id));
                EXPECT_NO_THROW(model->get_fired(neuron_id));
                EXPECT_NO_THROW(model->get_secondary_variable(neuron_id));
                EXPECT_NO_THROW(model->get_I_syn(neuron_id));
            }

            for (size_t neuron_id = 0; neuron_id < num_neurons; neuron_id++) {
                EXPECT_THROW(model->get_x(neuron_id + num_neurons), RelearnException);
                EXPECT_THROW(model->get_fired(neuron_id + num_neurons), RelearnException);
                EXPECT_THROW(model->get_secondary_variable(neuron_id + num_neurons), RelearnException);
                EXPECT_THROW(model->get_I_syn(neuron_id + num_neurons), RelearnException);
            }
        }
    }

    for (auto i = 0; i < iterations; i++) {
        const urd urd_desired_k(NeuronModels::min_k, NeuronModels::max_k);
        const urd urd_desired_tau_C(NeuronModels::min_tau_C, NeuronModels::max_tau_C);
        const urd urd_desired_beta(NeuronModels::min_beta, NeuronModels::max_beta);
        const uid uid_desired_h(NeuronModels::min_h, NeuronModels::max_h);
        const urd urd_desired_base_background_activity(NeuronModels::min_base_background_activity, NeuronModels::max_base_background_activity);
        const urd urd_desired_background_activity_mean(NeuronModels::min_background_activity_mean, NeuronModels::max_background_activity_mean);
        const urd urd_desired_background_activity_stddev(NeuronModels::min_background_activity_stddev, NeuronModels::max_background_activity_stddev);

        const urd urd_desired_C(AEIFModel::min_C, AEIFModel::max_C);
        const urd urd_desired_g_L(AEIFModel::min_g_L, AEIFModel::max_g_L);
        const urd urd_desired_E_L(AEIFModel::min_E_L, AEIFModel::max_E_L);
        const urd urd_desired_V_T(AEIFModel::min_V_T, AEIFModel::max_V_T);
        const urd urd_desired_d_T(AEIFModel::min_d_T, AEIFModel::max_d_T);
        const urd urd_desired_tau_w(AEIFModel::min_tau_w, AEIFModel::max_tau_w);
        const urd urd_desired_a(AEIFModel::min_a, AEIFModel::max_a);
        const urd urd_desired_b(AEIFModel::min_b, AEIFModel::max_b);
        const urd urd_desired_V_peak(AEIFModel::min_V_peak, AEIFModel::max_V_peak);

        const auto expected_k = urd_desired_k(mt);
        const auto expected_tau_C = urd_desired_tau_C(mt);
        const auto expected_beta = urd_desired_beta(mt);
        const auto expected_h = uid_desired_h(mt);
        const auto expected_base_background_activity = urd_desired_base_background_activity(mt);
        const auto expected_background_activity_mean = urd_desired_background_activity_mean(mt);
        const auto expected_background_activity_stddev = urd_desired_background_activity_stddev(mt);

        const auto expected_C = urd_desired_C(mt);
        const auto expected_g_L = urd_desired_g_L(mt);
        const auto expected_E_L = urd_desired_E_L(mt);
        const auto expected_V_T = urd_desired_V_T(mt);
        const auto expected_d_T = urd_desired_d_T(mt);
        const auto expected_tau_w = urd_desired_tau_w(mt);
        const auto expected_a = urd_desired_a(mt);
        const auto expected_b = urd_desired_b(mt);
        const auto expected_V_peak = urd_desired_V_peak(mt);

        auto model = std::make_unique<AEIFModel>(expected_k, expected_tau_C, expected_beta, expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
            expected_C, expected_g_L, expected_E_L, expected_V_T, expected_d_T, expected_tau_w, expected_a, expected_b, expected_V_peak);

        const uid uid_num_neurons(1, num_neurons_test);

        for (auto j = 0; j < iterations; j++) {
            const auto desired_num_neurons = uid_num_neurons(mt);

            model->init(desired_num_neurons);

            const auto num_neurons = model->get_num_neurons();

            const auto& x = model->get_x();
            const auto& fired = model->get_fired();

            EXPECT_EQ(desired_num_neurons, num_neurons);
            EXPECT_EQ(desired_num_neurons, x.size());
            EXPECT_EQ(desired_num_neurons, fired.size());

            for (size_t neuron_id = 0; neuron_id < num_neurons; neuron_id++) {
                EXPECT_NO_THROW(model->get_x(neuron_id));
                EXPECT_NO_THROW(model->get_fired(neuron_id));
                EXPECT_NO_THROW(model->get_secondary_variable(neuron_id));
                EXPECT_NO_THROW(model->get_I_syn(neuron_id));
            }

            for (size_t neuron_id = 0; neuron_id < num_neurons; neuron_id++) {
                EXPECT_THROW(model->get_x(neuron_id + num_neurons), RelearnException);
                EXPECT_THROW(model->get_fired(neuron_id + num_neurons), RelearnException);
                EXPECT_THROW(model->get_secondary_variable(neuron_id + num_neurons), RelearnException);
                EXPECT_THROW(model->get_I_syn(neuron_id + num_neurons), RelearnException);
            }
        }
    }
}

TEST(TestNeuronModels, testNeuronModelsCreateNeurons) {
    using namespace models;
    using urd = std::uniform_real_distribution<double>;
    using uid = std::uniform_int_distribution<unsigned int>;

    setup();

    {
        const urd urd_desired_k(NeuronModels::min_k, NeuronModels::max_k);
        const urd urd_desired_tau_C(NeuronModels::min_tau_C, NeuronModels::max_tau_C);
        const urd urd_desired_beta(NeuronModels::min_beta, NeuronModels::max_beta);
        const uid uid_desired_h(NeuronModels::min_h, NeuronModels::max_h);
        const urd urd_desired_base_background_activity(NeuronModels::min_base_background_activity, NeuronModels::max_base_background_activity);
        const urd urd_desired_background_activity_mean(NeuronModels::min_background_activity_mean, NeuronModels::max_background_activity_mean);
        const urd urd_desired_background_activity_stddev(NeuronModels::min_background_activity_stddev, NeuronModels::max_background_activity_stddev);

        const urd urd_desired_x0(PoissonModel::min_x_0, PoissonModel::max_x_0);
        const uid urd_desired_refrac(PoissonModel::min_refrac_time, PoissonModel::max_refrac_time);
        const urd urd_desired_tau_x(PoissonModel::min_tau_x, PoissonModel::max_tau_x);

        const auto expected_k = urd_desired_k(mt);
        const auto expected_tau_C = urd_desired_tau_C(mt);
        const auto expected_beta = urd_desired_beta(mt);
        const auto expected_h = uid_desired_h(mt);
        const auto expected_base_background_activity = urd_desired_base_background_activity(mt);
        const auto expected_background_activity_mean = urd_desired_background_activity_mean(mt);
        const auto expected_background_activity_stddev = urd_desired_background_activity_stddev(mt);

        const auto expected_x0 = urd_desired_x0(mt);
        const auto expected_refrac = urd_desired_refrac(mt);
        const auto expected_tau_x = urd_desired_tau_x(mt);

        auto model = std::make_unique<PoissonModel>(expected_k, expected_tau_C, expected_beta, expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
            expected_x0, expected_tau_x, expected_refrac);

        const uid uid_num_neurons(1, num_neurons_test);

        size_t current_num_neurons = 0;

        for (auto j = 0; j < 3; j++) {
            const auto desired_num_neurons = uid_num_neurons(mt);
            current_num_neurons += desired_num_neurons;

            model->create_neurons(desired_num_neurons);

            const auto num_neurons = model->get_num_neurons();

            const auto& x = model->get_x();
            const auto& fired = model->get_fired();

            EXPECT_EQ(current_num_neurons, num_neurons);
            EXPECT_EQ(current_num_neurons, x.size());
            EXPECT_EQ(current_num_neurons, fired.size());

            for (size_t neuron_id = 0; neuron_id < current_num_neurons; neuron_id++) {
                EXPECT_NO_THROW(model->get_x(neuron_id));
                EXPECT_NO_THROW(model->get_fired(neuron_id));
                EXPECT_NO_THROW(model->get_secondary_variable(neuron_id));
                EXPECT_NO_THROW(model->get_I_syn(neuron_id));
            }

            for (size_t neuron_id = 0; neuron_id < current_num_neurons; neuron_id++) {
                EXPECT_THROW(model->get_x(neuron_id + current_num_neurons), RelearnException);
                EXPECT_THROW(model->get_fired(neuron_id + current_num_neurons), RelearnException);
                EXPECT_THROW(model->get_secondary_variable(neuron_id + current_num_neurons), RelearnException);
                EXPECT_THROW(model->get_I_syn(neuron_id + current_num_neurons), RelearnException);
            }
        }
    }

    {
        const urd urd_desired_k(NeuronModels::min_k, NeuronModels::max_k);
        const urd urd_desired_tau_C(NeuronModels::min_tau_C, NeuronModels::max_tau_C);
        const urd urd_desired_beta(NeuronModels::min_beta, NeuronModels::max_beta);
        const uid uid_desired_h(NeuronModels::min_h, NeuronModels::max_h);
        const urd urd_desired_base_background_activity(NeuronModels::min_base_background_activity, NeuronModels::max_base_background_activity);
        const urd urd_desired_background_activity_mean(NeuronModels::min_background_activity_mean, NeuronModels::max_background_activity_mean);
        const urd urd_desired_background_activity_stddev(NeuronModels::min_background_activity_stddev, NeuronModels::max_background_activity_stddev);

        const urd urd_desired_a(IzhikevichModel::min_a, IzhikevichModel::max_a);
        const urd urd_desired_b(IzhikevichModel::min_b, IzhikevichModel::max_b);
        const urd urd_desired_c(IzhikevichModel::min_c, IzhikevichModel::max_c);
        const urd urd_desired_d(IzhikevichModel::min_d, IzhikevichModel::max_d);
        const urd urd_desired_V_spike(IzhikevichModel::min_V_spike, IzhikevichModel::max_V_spike);
        const urd urd_desired_k1(IzhikevichModel::min_k1, IzhikevichModel::max_k1);
        const urd urd_desired_k2(IzhikevichModel::min_k2, IzhikevichModel::max_k2);
        const urd urd_desired_k3(IzhikevichModel::min_k3, IzhikevichModel::max_k3);

        const auto expected_k = urd_desired_k(mt);
        const auto expected_tau_C = urd_desired_tau_C(mt);
        const auto expected_beta = urd_desired_beta(mt);
        const auto expected_h = uid_desired_h(mt);
        const auto expected_base_background_activity = urd_desired_base_background_activity(mt);
        const auto expected_background_activity_mean = urd_desired_background_activity_mean(mt);
        const auto expected_background_activity_stddev = urd_desired_background_activity_stddev(mt);

        const auto expected_a = urd_desired_a(mt);
        const auto expected_b = urd_desired_b(mt);
        const auto expected_c = urd_desired_c(mt);
        const auto expected_d = urd_desired_d(mt);
        const auto expected_V_spike = urd_desired_V_spike(mt);
        const auto expected_k1 = urd_desired_k1(mt);
        const auto expected_k2 = urd_desired_k2(mt);
        const auto expected_k3 = urd_desired_k3(mt);

        auto model = std::make_unique<IzhikevichModel>(expected_k, expected_tau_C, expected_beta, expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
            expected_a, expected_b, expected_c, expected_d, expected_V_spike, expected_k1, expected_k2, expected_k3);

        const uid uid_num_neurons(1, num_neurons_test);

        size_t current_num_neurons = 0;

        for (auto j = 0; j < 3; j++) {
            const auto desired_num_neurons = uid_num_neurons(mt);
            current_num_neurons += desired_num_neurons;

            model->create_neurons(desired_num_neurons);

            const auto num_neurons = model->get_num_neurons();

            const auto& x = model->get_x();
            const auto& fired = model->get_fired();

            EXPECT_EQ(current_num_neurons, num_neurons);
            EXPECT_EQ(current_num_neurons, x.size());
            EXPECT_EQ(current_num_neurons, fired.size());

            for (size_t neuron_id = 0; neuron_id < current_num_neurons; neuron_id++) {
                EXPECT_NO_THROW(model->get_x(neuron_id));
                EXPECT_NO_THROW(model->get_fired(neuron_id));
                EXPECT_NO_THROW(model->get_secondary_variable(neuron_id));
                EXPECT_NO_THROW(model->get_I_syn(neuron_id));
            }

            for (size_t neuron_id = 0; neuron_id < current_num_neurons; neuron_id++) {
                EXPECT_THROW(model->get_x(neuron_id + current_num_neurons), RelearnException);
                EXPECT_THROW(model->get_fired(neuron_id + current_num_neurons), RelearnException);
                EXPECT_THROW(model->get_secondary_variable(neuron_id + current_num_neurons), RelearnException);
                EXPECT_THROW(model->get_I_syn(neuron_id + current_num_neurons), RelearnException);
            }
        }
    }

    {
        const urd urd_desired_k(NeuronModels::min_k, NeuronModels::max_k);
        const urd urd_desired_tau_C(NeuronModels::min_tau_C, NeuronModels::max_tau_C);
        const urd urd_desired_beta(NeuronModels::min_beta, NeuronModels::max_beta);
        const uid uid_desired_h(NeuronModels::min_h, NeuronModels::max_h);
        const urd urd_desired_base_background_activity(NeuronModels::min_base_background_activity, NeuronModels::max_base_background_activity);
        const urd urd_desired_background_activity_mean(NeuronModels::min_background_activity_mean, NeuronModels::max_background_activity_mean);
        const urd urd_desired_background_activity_stddev(NeuronModels::min_background_activity_stddev, NeuronModels::max_background_activity_stddev);

        const urd urd_desired_a(FitzHughNagumoModel::min_a, FitzHughNagumoModel::max_a);
        const urd urd_desired_b(FitzHughNagumoModel::min_b, FitzHughNagumoModel::max_b);
        const urd urd_desired_phi(FitzHughNagumoModel::min_phi, FitzHughNagumoModel::max_phi);

        const auto expected_k = urd_desired_k(mt);
        const auto expected_tau_C = urd_desired_tau_C(mt);
        const auto expected_beta = urd_desired_beta(mt);
        const auto expected_h = uid_desired_h(mt);
        const auto expected_base_background_activity = urd_desired_base_background_activity(mt);
        const auto expected_background_activity_mean = urd_desired_background_activity_mean(mt);
        const auto expected_background_activity_stddev = urd_desired_background_activity_stddev(mt);

        const auto expected_a = urd_desired_a(mt);
        const auto expected_b = urd_desired_b(mt);
        const auto expected_phi = urd_desired_phi(mt);

        auto model = std::make_unique<FitzHughNagumoModel>(expected_k, expected_tau_C, expected_beta, expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
            expected_a, expected_b, expected_phi);

        const uid uid_num_neurons(1, num_neurons_test);

        size_t current_num_neurons = 0;

        for (auto j = 0; j < 3; j++) {
            const auto desired_num_neurons = uid_num_neurons(mt);
            current_num_neurons += desired_num_neurons;

            model->create_neurons(desired_num_neurons);

            const auto num_neurons = model->get_num_neurons();

            const auto& x = model->get_x();
            const auto& fired = model->get_fired();

            EXPECT_EQ(current_num_neurons, num_neurons);
            EXPECT_EQ(current_num_neurons, x.size());
            EXPECT_EQ(current_num_neurons, fired.size());

            for (size_t neuron_id = 0; neuron_id < current_num_neurons; neuron_id++) {
                EXPECT_NO_THROW(model->get_x(neuron_id));
                EXPECT_NO_THROW(model->get_fired(neuron_id));
                EXPECT_NO_THROW(model->get_secondary_variable(neuron_id));
                EXPECT_NO_THROW(model->get_I_syn(neuron_id));
            }

            for (size_t neuron_id = 0; neuron_id < current_num_neurons; neuron_id++) {
                EXPECT_THROW(model->get_x(neuron_id + current_num_neurons), RelearnException);
                EXPECT_THROW(model->get_fired(neuron_id + current_num_neurons), RelearnException);
                EXPECT_THROW(model->get_secondary_variable(neuron_id + current_num_neurons), RelearnException);
                EXPECT_THROW(model->get_I_syn(neuron_id + current_num_neurons), RelearnException);
            }
        }
    }

    {
        const urd urd_desired_k(NeuronModels::min_k, NeuronModels::max_k);
        const urd urd_desired_tau_C(NeuronModels::min_tau_C, NeuronModels::max_tau_C);
        const urd urd_desired_beta(NeuronModels::min_beta, NeuronModels::max_beta);
        const uid uid_desired_h(NeuronModels::min_h, NeuronModels::max_h);
        const urd urd_desired_base_background_activity(NeuronModels::min_base_background_activity, NeuronModels::max_base_background_activity);
        const urd urd_desired_background_activity_mean(NeuronModels::min_background_activity_mean, NeuronModels::max_background_activity_mean);
        const urd urd_desired_background_activity_stddev(NeuronModels::min_background_activity_stddev, NeuronModels::max_background_activity_stddev);

        const urd urd_desired_C(AEIFModel::min_C, AEIFModel::max_C);
        const urd urd_desired_g_L(AEIFModel::min_g_L, AEIFModel::max_g_L);
        const urd urd_desired_E_L(AEIFModel::min_E_L, AEIFModel::max_E_L);
        const urd urd_desired_V_T(AEIFModel::min_V_T, AEIFModel::max_V_T);
        const urd urd_desired_d_T(AEIFModel::min_d_T, AEIFModel::max_d_T);
        const urd urd_desired_tau_w(AEIFModel::min_tau_w, AEIFModel::max_tau_w);
        const urd urd_desired_a(AEIFModel::min_a, AEIFModel::max_a);
        const urd urd_desired_b(AEIFModel::min_b, AEIFModel::max_b);
        const urd urd_desired_V_peak(AEIFModel::min_V_peak, AEIFModel::max_V_peak);

        const auto expected_k = urd_desired_k(mt);
        const auto expected_tau_C = urd_desired_tau_C(mt);
        const auto expected_beta = urd_desired_beta(mt);
        const auto expected_h = uid_desired_h(mt);
        const auto expected_base_background_activity = urd_desired_base_background_activity(mt);
        const auto expected_background_activity_mean = urd_desired_background_activity_mean(mt);
        const auto expected_background_activity_stddev = urd_desired_background_activity_stddev(mt);

        const auto expected_C = urd_desired_C(mt);
        const auto expected_g_L = urd_desired_g_L(mt);
        const auto expected_E_L = urd_desired_E_L(mt);
        const auto expected_V_T = urd_desired_V_T(mt);
        const auto expected_d_T = urd_desired_d_T(mt);
        const auto expected_tau_w = urd_desired_tau_w(mt);
        const auto expected_a = urd_desired_a(mt);
        const auto expected_b = urd_desired_b(mt);
        const auto expected_V_peak = urd_desired_V_peak(mt);

        auto model = std::make_unique<AEIFModel>(expected_k, expected_tau_C, expected_beta, expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
            expected_C, expected_g_L, expected_E_L, expected_V_T, expected_d_T, expected_tau_w, expected_a, expected_b, expected_V_peak);

        const uid uid_num_neurons(1, num_neurons_test);

        size_t current_num_neurons = 0;

        for (auto j = 0; j < 3; j++) {
            const auto desired_num_neurons = uid_num_neurons(mt);
            current_num_neurons += desired_num_neurons;

            model->create_neurons(desired_num_neurons);

            const auto num_neurons = model->get_num_neurons();

            const auto& x = model->get_x();
            const auto& fired = model->get_fired();

            EXPECT_EQ(current_num_neurons, num_neurons);
            EXPECT_EQ(current_num_neurons, x.size());
            EXPECT_EQ(current_num_neurons, fired.size());

            for (size_t neuron_id = 0; neuron_id < current_num_neurons; neuron_id++) {
                EXPECT_NO_THROW(model->get_x(neuron_id));
                EXPECT_NO_THROW(model->get_fired(neuron_id));
                EXPECT_NO_THROW(model->get_secondary_variable(neuron_id));
                EXPECT_NO_THROW(model->get_I_syn(neuron_id));
            }

            for (size_t neuron_id = 0; neuron_id < current_num_neurons; neuron_id++) {
                EXPECT_THROW(model->get_x(neuron_id + current_num_neurons), RelearnException);
                EXPECT_THROW(model->get_fired(neuron_id + current_num_neurons), RelearnException);
                EXPECT_THROW(model->get_secondary_variable(neuron_id + current_num_neurons), RelearnException);
                EXPECT_THROW(model->get_I_syn(neuron_id + current_num_neurons), RelearnException);
            }
        }
    }
}

TEST(TestNeuronModels, testNeuronModelsDisableFired) {
    using namespace models;
    using urd = std::uniform_real_distribution<double>;
    using uid = std::uniform_int_distribution<unsigned int>;

    setup();

    for (auto i = 0; i < iterations; i++) {
        const urd urd_desired_k(NeuronModels::min_k, NeuronModels::max_k);
        const urd urd_desired_tau_C(NeuronModels::min_tau_C, NeuronModels::max_tau_C);
        const urd urd_desired_beta(NeuronModels::min_beta, NeuronModels::max_beta);
        const uid uid_desired_h(NeuronModels::min_h, NeuronModels::max_h);
        const urd urd_desired_base_background_activity(NeuronModels::min_base_background_activity, NeuronModels::max_base_background_activity);
        const urd urd_desired_background_activity_mean(NeuronModels::min_background_activity_mean, NeuronModels::max_background_activity_mean);
        const urd urd_desired_background_activity_stddev(NeuronModels::min_background_activity_stddev, NeuronModels::max_background_activity_stddev);

        const urd urd_desired_x0(PoissonModel::min_x_0, PoissonModel::max_x_0);
        const uid urd_desired_refrac(PoissonModel::min_refrac_time, PoissonModel::max_refrac_time);
        const urd urd_desired_tau_x(PoissonModel::min_tau_x, PoissonModel::max_tau_x);

        const auto expected_k = urd_desired_k(mt);
        const auto expected_tau_C = urd_desired_tau_C(mt);
        const auto expected_beta = urd_desired_beta(mt);
        const auto expected_h = uid_desired_h(mt);
        const auto expected_base_background_activity = urd_desired_base_background_activity(mt);
        const auto expected_background_activity_mean = urd_desired_background_activity_mean(mt);
        const auto expected_background_activity_stddev = urd_desired_background_activity_stddev(mt);

        const auto expected_x0 = urd_desired_x0(mt);
        const auto expected_refrac = urd_desired_refrac(mt);
        const auto expected_tau_x = urd_desired_tau_x(mt);

        auto model = std::make_unique<PoissonModel>(expected_k, expected_tau_C, expected_beta, expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
            expected_x0, expected_tau_x, expected_refrac);

        const uid uid_num_neurons(1, num_neurons_test);
        const auto num_neurons = uid_num_neurons(mt);

        const uid uid_num_neurons_disables(1, num_neurons);
        const auto num_disables = uid_num_neurons_disables(mt);

        const auto disable_ids = generate_random_ids(0, num_neurons - 1, num_disables);

        model->init(num_neurons);
        model->disable_neurons(disable_ids);

        for (auto id = 0; id < num_neurons; id++) {
            if (std::find(disable_ids.cbegin(), disable_ids.cend(), id) != disable_ids.cend()) {
                EXPECT_FALSE(model->get_fired(id));
            }
        }

        const auto disable_ids_failure = generate_random_ids(num_neurons, num_neurons + num_neurons, num_disables);

        EXPECT_THROW(model->disable_neurons(disable_ids_failure), RelearnException);
    }

    for (auto i = 0; i < iterations; i++) {
        const urd urd_desired_k(NeuronModels::min_k, NeuronModels::max_k);
        const urd urd_desired_tau_C(NeuronModels::min_tau_C, NeuronModels::max_tau_C);
        const urd urd_desired_beta(NeuronModels::min_beta, NeuronModels::max_beta);
        const uid uid_desired_h(NeuronModels::min_h, NeuronModels::max_h);
        const urd urd_desired_base_background_activity(NeuronModels::min_base_background_activity, NeuronModels::max_base_background_activity);
        const urd urd_desired_background_activity_mean(NeuronModels::min_background_activity_mean, NeuronModels::max_background_activity_mean);
        const urd urd_desired_background_activity_stddev(NeuronModels::min_background_activity_stddev, NeuronModels::max_background_activity_stddev);

        const urd urd_desired_a(IzhikevichModel::min_a, IzhikevichModel::max_a);
        const urd urd_desired_b(IzhikevichModel::min_b, IzhikevichModel::max_b);
        const urd urd_desired_c(IzhikevichModel::min_c, IzhikevichModel::max_c);
        const urd urd_desired_d(IzhikevichModel::min_d, IzhikevichModel::max_d);
        const urd urd_desired_V_spike(IzhikevichModel::min_V_spike, IzhikevichModel::max_V_spike);
        const urd urd_desired_k1(IzhikevichModel::min_k1, IzhikevichModel::max_k1);
        const urd urd_desired_k2(IzhikevichModel::min_k2, IzhikevichModel::max_k2);
        const urd urd_desired_k3(IzhikevichModel::min_k3, IzhikevichModel::max_k3);

        const auto expected_k = urd_desired_k(mt);
        const auto expected_tau_C = urd_desired_tau_C(mt);
        const auto expected_beta = urd_desired_beta(mt);
        const auto expected_h = uid_desired_h(mt);
        const auto expected_base_background_activity = urd_desired_base_background_activity(mt);
        const auto expected_background_activity_mean = urd_desired_background_activity_mean(mt);
        const auto expected_background_activity_stddev = urd_desired_background_activity_stddev(mt);

        const auto expected_a = urd_desired_a(mt);
        const auto expected_b = urd_desired_b(mt);
        const auto expected_c = urd_desired_c(mt);
        const auto expected_d = urd_desired_d(mt);
        const auto expected_V_spike = urd_desired_V_spike(mt);
        const auto expected_k1 = urd_desired_k1(mt);
        const auto expected_k2 = urd_desired_k2(mt);
        const auto expected_k3 = urd_desired_k3(mt);

        auto model = std::make_unique<IzhikevichModel>(expected_k, expected_tau_C, expected_beta, expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
            expected_a, expected_b, expected_c, expected_d, expected_V_spike, expected_k1, expected_k2, expected_k3);

        const uid uid_num_neurons(1, num_neurons_test);
        const auto num_neurons = uid_num_neurons(mt);

        const uid uid_num_neurons_disables(1, num_neurons);
        const auto num_disables = uid_num_neurons_disables(mt);

        const auto disable_ids = generate_random_ids(0, num_neurons - 1, num_disables);

        model->init(num_neurons);
        model->disable_neurons(disable_ids);

        for (auto id = 0; id < num_neurons; id++) {
            if (std::find(disable_ids.cbegin(), disable_ids.cend(), id) != disable_ids.cend()) {
                EXPECT_FALSE(model->get_fired(id));
            }
        }

        const auto disable_ids_failure = generate_random_ids(num_neurons, num_neurons + num_neurons, num_disables);

        EXPECT_THROW(model->disable_neurons(disable_ids_failure), RelearnException);
    }

    for (auto i = 0; i < iterations; i++) {
        const urd urd_desired_k(NeuronModels::min_k, NeuronModels::max_k);
        const urd urd_desired_tau_C(NeuronModels::min_tau_C, NeuronModels::max_tau_C);
        const urd urd_desired_beta(NeuronModels::min_beta, NeuronModels::max_beta);
        const uid uid_desired_h(NeuronModels::min_h, NeuronModels::max_h);
        const urd urd_desired_base_background_activity(NeuronModels::min_base_background_activity, NeuronModels::max_base_background_activity);
        const urd urd_desired_background_activity_mean(NeuronModels::min_background_activity_mean, NeuronModels::max_background_activity_mean);
        const urd urd_desired_background_activity_stddev(NeuronModels::min_background_activity_stddev, NeuronModels::max_background_activity_stddev);

        const urd urd_desired_a(FitzHughNagumoModel::min_a, FitzHughNagumoModel::max_a);
        const urd urd_desired_b(FitzHughNagumoModel::min_b, FitzHughNagumoModel::max_b);
        const urd urd_desired_phi(FitzHughNagumoModel::min_phi, FitzHughNagumoModel::max_phi);

        const auto expected_k = urd_desired_k(mt);
        const auto expected_tau_C = urd_desired_tau_C(mt);
        const auto expected_beta = urd_desired_beta(mt);
        const auto expected_h = uid_desired_h(mt);
        const auto expected_base_background_activity = urd_desired_base_background_activity(mt);
        const auto expected_background_activity_mean = urd_desired_background_activity_mean(mt);
        const auto expected_background_activity_stddev = urd_desired_background_activity_stddev(mt);

        const auto expected_a = urd_desired_a(mt);
        const auto expected_b = urd_desired_b(mt);
        const auto expected_phi = urd_desired_phi(mt);

        auto model = std::make_unique<FitzHughNagumoModel>(expected_k, expected_tau_C, expected_beta, expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
            expected_a, expected_b, expected_phi);

        const uid uid_num_neurons(1, num_neurons_test);
        const auto num_neurons = uid_num_neurons(mt);

        const uid uid_num_neurons_disables(1, num_neurons);
        const auto num_disables = uid_num_neurons_disables(mt);

        const auto disable_ids = generate_random_ids(0, num_neurons - 1, num_disables);

        model->init(num_neurons);
        model->disable_neurons(disable_ids);

        for (auto id = 0; id < num_neurons; id++) {
            if (std::find(disable_ids.cbegin(), disable_ids.cend(), id) != disable_ids.cend()) {
                EXPECT_FALSE(model->get_fired(id));
            }
        }

        const auto disable_ids_failure = generate_random_ids(num_neurons, num_neurons + num_neurons, num_disables);

        EXPECT_THROW(model->disable_neurons(disable_ids_failure), RelearnException);
    }

    for (auto i = 0; i < iterations; i++) {
        const urd urd_desired_k(NeuronModels::min_k, NeuronModels::max_k);
        const urd urd_desired_tau_C(NeuronModels::min_tau_C, NeuronModels::max_tau_C);
        const urd urd_desired_beta(NeuronModels::min_beta, NeuronModels::max_beta);
        const uid uid_desired_h(NeuronModels::min_h, NeuronModels::max_h);
        const urd urd_desired_base_background_activity(NeuronModels::min_base_background_activity, NeuronModels::max_base_background_activity);
        const urd urd_desired_background_activity_mean(NeuronModels::min_background_activity_mean, NeuronModels::max_background_activity_mean);
        const urd urd_desired_background_activity_stddev(NeuronModels::min_background_activity_stddev, NeuronModels::max_background_activity_stddev);

        const urd urd_desired_C(AEIFModel::min_C, AEIFModel::max_C);
        const urd urd_desired_g_L(AEIFModel::min_g_L, AEIFModel::max_g_L);
        const urd urd_desired_E_L(AEIFModel::min_E_L, AEIFModel::max_E_L);
        const urd urd_desired_V_T(AEIFModel::min_V_T, AEIFModel::max_V_T);
        const urd urd_desired_d_T(AEIFModel::min_d_T, AEIFModel::max_d_T);
        const urd urd_desired_tau_w(AEIFModel::min_tau_w, AEIFModel::max_tau_w);
        const urd urd_desired_a(AEIFModel::min_a, AEIFModel::max_a);
        const urd urd_desired_b(AEIFModel::min_b, AEIFModel::max_b);
        const urd urd_desired_V_peak(AEIFModel::min_V_peak, AEIFModel::max_V_peak);

        const auto expected_k = urd_desired_k(mt);
        const auto expected_tau_C = urd_desired_tau_C(mt);
        const auto expected_beta = urd_desired_beta(mt);
        const auto expected_h = uid_desired_h(mt);
        const auto expected_base_background_activity = urd_desired_base_background_activity(mt);
        const auto expected_background_activity_mean = urd_desired_background_activity_mean(mt);
        const auto expected_background_activity_stddev = urd_desired_background_activity_stddev(mt);

        const auto expected_C = urd_desired_C(mt);
        const auto expected_g_L = urd_desired_g_L(mt);
        const auto expected_E_L = urd_desired_E_L(mt);
        const auto expected_V_T = urd_desired_V_T(mt);
        const auto expected_d_T = urd_desired_d_T(mt);
        const auto expected_tau_w = urd_desired_tau_w(mt);
        const auto expected_a = urd_desired_a(mt);
        const auto expected_b = urd_desired_b(mt);
        const auto expected_V_peak = urd_desired_V_peak(mt);

        auto model = std::make_unique<AEIFModel>(expected_k, expected_tau_C, expected_beta, expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
            expected_C, expected_g_L, expected_E_L, expected_V_T, expected_d_T, expected_tau_w, expected_a, expected_b, expected_V_peak);

        const uid uid_num_neurons(1, num_neurons_test);
        const auto num_neurons = uid_num_neurons(mt);

        const uid uid_num_neurons_disables(1, num_neurons);
        const auto num_disables = uid_num_neurons_disables(mt);

        const auto disable_ids = generate_random_ids(0, num_neurons - 1, num_disables);

        model->init(num_neurons);
        model->disable_neurons(disable_ids);

        for (auto id = 0; id < num_neurons; id++) {
            if (std::find(disable_ids.cbegin(), disable_ids.cend(), id) != disable_ids.cend()) {
                EXPECT_FALSE(model->get_fired(id));
            }
        }

        const auto disable_ids_failure = generate_random_ids(num_neurons, num_neurons + num_neurons, num_disables);

        EXPECT_THROW(model->disable_neurons(disable_ids_failure), RelearnException);
    }
}

TEST(TestNeuronModels, testNeuronModelsUpdateActivityDisabled) {
    using namespace models;
    using urd = std::uniform_real_distribution<double>;
    using uid = std::uniform_int_distribution<unsigned int>;

    setup();

    for (auto i = 0; i < iterations; i++) {
        const urd urd_desired_k(NeuronModels::min_k, NeuronModels::max_k);
        const urd urd_desired_tau_C(NeuronModels::min_tau_C, NeuronModels::max_tau_C);
        const urd urd_desired_beta(NeuronModels::min_beta, NeuronModels::max_beta);
        const uid uid_desired_h(NeuronModels::min_h, NeuronModels::max_h);
        const urd urd_desired_base_background_activity(NeuronModels::min_base_background_activity, NeuronModels::max_base_background_activity);
        const urd urd_desired_background_activity_mean(NeuronModels::min_background_activity_mean, NeuronModels::max_background_activity_mean);
        const urd urd_desired_background_activity_stddev(NeuronModels::min_background_activity_stddev, NeuronModels::max_background_activity_stddev);

        const urd urd_desired_x0(PoissonModel::min_x_0, PoissonModel::max_x_0);
        const uid urd_desired_refrac(PoissonModel::min_refrac_time, PoissonModel::max_refrac_time);
        const urd urd_desired_tau_x(PoissonModel::min_tau_x, PoissonModel::max_tau_x);

        const auto expected_k = urd_desired_k(mt);
        const auto expected_tau_C = urd_desired_tau_C(mt);
        const auto expected_beta = urd_desired_beta(mt);
        const auto expected_h = uid_desired_h(mt);
        const auto expected_base_background_activity = urd_desired_base_background_activity(mt);
        const auto expected_background_activity_mean = urd_desired_background_activity_mean(mt);
        const auto expected_background_activity_stddev = urd_desired_background_activity_stddev(mt);

        const auto expected_x0 = urd_desired_x0(mt);
        const auto expected_refrac = urd_desired_refrac(mt);
        const auto expected_tau_x = urd_desired_tau_x(mt);

        auto model = std::make_unique<PoissonModel>(expected_k, expected_tau_C, expected_beta, expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
            expected_x0, expected_tau_x, expected_refrac);

        const uid uid_num_neurons(1, num_neurons_test);
        const auto desired_num_neurons = uid_num_neurons(mt);

        const auto empty_graph = generate_random_network_graph(desired_num_neurons, 0, 1.0);

        model->init(desired_num_neurons);

        std::vector<char> disable_flags(desired_num_neurons);
        std::vector<double> model_x = model->get_x();
        std::vector<double> model_secondary(desired_num_neurons);
        std::vector<bool> model_fired = model->get_fired();
        std::vector<double> model_I_sync(desired_num_neurons);

        for (auto id = 0; id < desired_num_neurons; id++) {
            model_secondary[id] = model->get_secondary_variable(id);
            model_I_sync[id] = model->get_I_syn(id);
        }

        for (auto j = 0; j < 3; j++) {
            model->update_electrical_activity(empty_graph, disable_flags);

            const auto& current_x = model->get_x();
            const auto& current_fired = model->get_fired();

            for (auto id = 0; id < desired_num_neurons; id++) {
                EXPECT_EQ(current_x[id], model_x[id]);
                EXPECT_EQ(current_fired[id], model_fired[id]);

                EXPECT_EQ(model->get_I_syn(id), model_I_sync[id]);
                EXPECT_EQ(model->get_secondary_variable(id), model_secondary[id]);

                model_I_sync[id] = model->get_I_syn(id);
                model_secondary[id] = model->get_secondary_variable(id);
            }

            model_x = current_x;
            model_fired = current_fired;
        }

        const auto random_graph = generate_random_network_graph(desired_num_neurons, desired_num_neurons, 1.0);

        for (auto j = 0; j < 3; j++) {
            model->update_electrical_activity(random_graph, disable_flags);

            const auto& current_x = model->get_x();
            const auto& current_fired = model->get_fired();

            for (auto id = 0; id < desired_num_neurons; id++) {
                EXPECT_EQ(current_x[id], model_x[id]);
                EXPECT_EQ(current_fired[id], model_fired[id]);

                EXPECT_EQ(model->get_I_syn(id), model_I_sync[id]);
                EXPECT_EQ(model->get_secondary_variable(id), model_secondary[id]);

                model_I_sync[id] = model->get_I_syn(id);
                model_secondary[id] = model->get_secondary_variable(id);
            }

            model_x = current_x;
            model_fired = current_fired;
        }
    }

    for (auto i = 0; i < iterations; i++) {
        const urd urd_desired_k(NeuronModels::min_k, NeuronModels::max_k);
        const urd urd_desired_tau_C(NeuronModels::min_tau_C, NeuronModels::max_tau_C);
        const urd urd_desired_beta(NeuronModels::min_beta, NeuronModels::max_beta);
        const uid uid_desired_h(NeuronModels::min_h, NeuronModels::max_h);
        const urd urd_desired_base_background_activity(NeuronModels::min_base_background_activity, NeuronModels::max_base_background_activity);
        const urd urd_desired_background_activity_mean(NeuronModels::min_background_activity_mean, NeuronModels::max_background_activity_mean);
        const urd urd_desired_background_activity_stddev(NeuronModels::min_background_activity_stddev, NeuronModels::max_background_activity_stddev);

        const urd urd_desired_a(IzhikevichModel::min_a, IzhikevichModel::max_a);
        const urd urd_desired_b(IzhikevichModel::min_b, IzhikevichModel::max_b);
        const urd urd_desired_c(IzhikevichModel::min_c, IzhikevichModel::max_c);
        const urd urd_desired_d(IzhikevichModel::min_d, IzhikevichModel::max_d);
        const urd urd_desired_V_spike(IzhikevichModel::min_V_spike, IzhikevichModel::max_V_spike);
        const urd urd_desired_k1(IzhikevichModel::min_k1, IzhikevichModel::max_k1);
        const urd urd_desired_k2(IzhikevichModel::min_k2, IzhikevichModel::max_k2);
        const urd urd_desired_k3(IzhikevichModel::min_k3, IzhikevichModel::max_k3);

        const auto expected_k = urd_desired_k(mt);
        const auto expected_tau_C = urd_desired_tau_C(mt);
        const auto expected_beta = urd_desired_beta(mt);
        const auto expected_h = uid_desired_h(mt);
        const auto expected_base_background_activity = urd_desired_base_background_activity(mt);
        const auto expected_background_activity_mean = urd_desired_background_activity_mean(mt);
        const auto expected_background_activity_stddev = urd_desired_background_activity_stddev(mt);

        const auto expected_a = urd_desired_a(mt);
        const auto expected_b = urd_desired_b(mt);
        const auto expected_c = urd_desired_c(mt);
        const auto expected_d = urd_desired_d(mt);
        const auto expected_V_spike = urd_desired_V_spike(mt);
        const auto expected_k1 = urd_desired_k1(mt);
        const auto expected_k2 = urd_desired_k2(mt);
        const auto expected_k3 = urd_desired_k3(mt);

        auto model = std::make_unique<IzhikevichModel>(expected_k, expected_tau_C, expected_beta, expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
            expected_a, expected_b, expected_c, expected_d, expected_V_spike, expected_k1, expected_k2, expected_k3);

        const uid uid_num_neurons(1, num_neurons_test);
        const auto desired_num_neurons = uid_num_neurons(mt);

        const auto empty_graph = generate_random_network_graph(desired_num_neurons, 0, 1.0);

        model->init(desired_num_neurons);

        std::vector<char> disable_flags(desired_num_neurons);
        std::vector<double> model_x = model->get_x();
        std::vector<double> model_secondary(desired_num_neurons);
        std::vector<bool> model_fired = model->get_fired();
        std::vector<double> model_I_sync(desired_num_neurons);

        for (auto id = 0; id < desired_num_neurons; id++) {
            model_secondary[id] = model->get_secondary_variable(id);
            model_I_sync[id] = model->get_I_syn(id);
        }

        for (auto j = 0; j < 3; j++) {
            model->update_electrical_activity(empty_graph, disable_flags);

            const auto& current_x = model->get_x();
            const auto& current_fired = model->get_fired();

            for (auto id = 0; id < desired_num_neurons; id++) {
                EXPECT_EQ(current_x[id], model_x[id]);
                EXPECT_EQ(current_fired[id], model_fired[id]);

                EXPECT_EQ(model->get_I_syn(id), model_I_sync[id]);
                EXPECT_EQ(model->get_secondary_variable(id), model_secondary[id]);

                model_I_sync[id] = model->get_I_syn(id);
                model_secondary[id] = model->get_secondary_variable(id);
            }

            model_x = current_x;
            model_fired = current_fired;
        }

        const auto random_graph = generate_random_network_graph(desired_num_neurons, desired_num_neurons, 1.0);

        for (auto j = 0; j < 3; j++) {
            model->update_electrical_activity(random_graph, disable_flags);

            const auto& current_x = model->get_x();
            const auto& current_fired = model->get_fired();

            for (auto id = 0; id < desired_num_neurons; id++) {
                EXPECT_EQ(current_x[id], model_x[id]);
                EXPECT_EQ(current_fired[id], model_fired[id]);

                EXPECT_EQ(model->get_I_syn(id), model_I_sync[id]);
                EXPECT_EQ(model->get_secondary_variable(id), model_secondary[id]);

                model_I_sync[id] = model->get_I_syn(id);
                model_secondary[id] = model->get_secondary_variable(id);
            }

            model_x = current_x;
            model_fired = current_fired;
        }
    }

    for (auto i = 0; i < iterations; i++) {
        const urd urd_desired_k(NeuronModels::min_k, NeuronModels::max_k);
        const urd urd_desired_tau_C(NeuronModels::min_tau_C, NeuronModels::max_tau_C);
        const urd urd_desired_beta(NeuronModels::min_beta, NeuronModels::max_beta);
        const uid uid_desired_h(NeuronModels::min_h, NeuronModels::max_h);
        const urd urd_desired_base_background_activity(NeuronModels::min_base_background_activity, NeuronModels::max_base_background_activity);
        const urd urd_desired_background_activity_mean(NeuronModels::min_background_activity_mean, NeuronModels::max_background_activity_mean);
        const urd urd_desired_background_activity_stddev(NeuronModels::min_background_activity_stddev, NeuronModels::max_background_activity_stddev);

        const urd urd_desired_a(FitzHughNagumoModel::min_a, FitzHughNagumoModel::max_a);
        const urd urd_desired_b(FitzHughNagumoModel::min_b, FitzHughNagumoModel::max_b);
        const urd urd_desired_phi(FitzHughNagumoModel::min_phi, FitzHughNagumoModel::max_phi);

        const auto expected_k = urd_desired_k(mt);
        const auto expected_tau_C = urd_desired_tau_C(mt);
        const auto expected_beta = urd_desired_beta(mt);
        const auto expected_h = uid_desired_h(mt);
        const auto expected_base_background_activity = urd_desired_base_background_activity(mt);
        const auto expected_background_activity_mean = urd_desired_background_activity_mean(mt);
        const auto expected_background_activity_stddev = urd_desired_background_activity_stddev(mt);

        const auto expected_a = urd_desired_a(mt);
        const auto expected_b = urd_desired_b(mt);
        const auto expected_phi = urd_desired_phi(mt);

        auto model = std::make_unique<FitzHughNagumoModel>(expected_k, expected_tau_C, expected_beta, expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
            expected_a, expected_b, expected_phi);

        const uid uid_num_neurons(1, num_neurons_test);
        const auto desired_num_neurons = uid_num_neurons(mt);

        const auto empty_graph = generate_random_network_graph(desired_num_neurons, 0, 1.0);

        model->init(desired_num_neurons);

        std::vector<char> disable_flags(desired_num_neurons);
        std::vector<double> model_x = model->get_x();
        std::vector<double> model_secondary(desired_num_neurons);
        std::vector<bool> model_fired = model->get_fired();
        std::vector<double> model_I_sync(desired_num_neurons);

        for (auto id = 0; id < desired_num_neurons; id++) {
            model_secondary[id] = model->get_secondary_variable(id);
            model_I_sync[id] = model->get_I_syn(id);
        }

        for (auto j = 0; j < 3; j++) {
            model->update_electrical_activity(empty_graph, disable_flags);

            const auto& current_x = model->get_x();
            const auto& current_fired = model->get_fired();

            for (auto id = 0; id < desired_num_neurons; id++) {
                EXPECT_EQ(current_x[id], model_x[id]);
                EXPECT_EQ(current_fired[id], model_fired[id]);

                EXPECT_EQ(model->get_I_syn(id), model_I_sync[id]);
                EXPECT_EQ(model->get_secondary_variable(id), model_secondary[id]);

                model_I_sync[id] = model->get_I_syn(id);
                model_secondary[id] = model->get_secondary_variable(id);
            }

            model_x = current_x;
            model_fired = current_fired;
        }

        const auto random_graph = generate_random_network_graph(desired_num_neurons, desired_num_neurons, 1.0);

        for (auto j = 0; j < 3; j++) {
            model->update_electrical_activity(random_graph, disable_flags);

            const auto& current_x = model->get_x();
            const auto& current_fired = model->get_fired();

            for (auto id = 0; id < desired_num_neurons; id++) {
                EXPECT_EQ(current_x[id], model_x[id]);
                EXPECT_EQ(current_fired[id], model_fired[id]);

                EXPECT_EQ(model->get_I_syn(id), model_I_sync[id]);
                EXPECT_EQ(model->get_secondary_variable(id), model_secondary[id]);

                model_I_sync[id] = model->get_I_syn(id);
                model_secondary[id] = model->get_secondary_variable(id);
            }

            model_x = current_x;
            model_fired = current_fired;
        }
    }

    for (auto i = 0; i < iterations; i++) {
        const urd urd_desired_k(NeuronModels::min_k, NeuronModels::max_k);
        const urd urd_desired_tau_C(NeuronModels::min_tau_C, NeuronModels::max_tau_C);
        const urd urd_desired_beta(NeuronModels::min_beta, NeuronModels::max_beta);
        const uid uid_desired_h(NeuronModels::min_h, NeuronModels::max_h);
        const urd urd_desired_base_background_activity(NeuronModels::min_base_background_activity, NeuronModels::max_base_background_activity);
        const urd urd_desired_background_activity_mean(NeuronModels::min_background_activity_mean, NeuronModels::max_background_activity_mean);
        const urd urd_desired_background_activity_stddev(NeuronModels::min_background_activity_stddev, NeuronModels::max_background_activity_stddev);

        const urd urd_desired_C(AEIFModel::min_C, AEIFModel::max_C);
        const urd urd_desired_g_L(AEIFModel::min_g_L, AEIFModel::max_g_L);
        const urd urd_desired_E_L(AEIFModel::min_E_L, AEIFModel::max_E_L);
        const urd urd_desired_V_T(AEIFModel::min_V_T, AEIFModel::max_V_T);
        const urd urd_desired_d_T(AEIFModel::min_d_T, AEIFModel::max_d_T);
        const urd urd_desired_tau_w(AEIFModel::min_tau_w, AEIFModel::max_tau_w);
        const urd urd_desired_a(AEIFModel::min_a, AEIFModel::max_a);
        const urd urd_desired_b(AEIFModel::min_b, AEIFModel::max_b);
        const urd urd_desired_V_peak(AEIFModel::min_V_peak, AEIFModel::max_V_peak);

        const auto expected_k = urd_desired_k(mt);
        const auto expected_tau_C = urd_desired_tau_C(mt);
        const auto expected_beta = urd_desired_beta(mt);
        const auto expected_h = uid_desired_h(mt);
        const auto expected_base_background_activity = urd_desired_base_background_activity(mt);
        const auto expected_background_activity_mean = urd_desired_background_activity_mean(mt);
        const auto expected_background_activity_stddev = urd_desired_background_activity_stddev(mt);

        const auto expected_C = urd_desired_C(mt);
        const auto expected_g_L = urd_desired_g_L(mt);
        const auto expected_E_L = urd_desired_E_L(mt);
        const auto expected_V_T = urd_desired_V_T(mt);
        const auto expected_d_T = urd_desired_d_T(mt);
        const auto expected_tau_w = urd_desired_tau_w(mt);
        const auto expected_a = urd_desired_a(mt);
        const auto expected_b = urd_desired_b(mt);
        const auto expected_V_peak = urd_desired_V_peak(mt);

        auto model = std::make_unique<AEIFModel>(expected_k, expected_tau_C, expected_beta, expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
            expected_C, expected_g_L, expected_E_L, expected_V_T, expected_d_T, expected_tau_w, expected_a, expected_b, expected_V_peak);

        const uid uid_num_neurons(1, num_neurons_test);
        const auto desired_num_neurons = uid_num_neurons(mt);

        const auto empty_graph = generate_random_network_graph(desired_num_neurons, 0, 1.0);

        model->init(desired_num_neurons);

        std::vector<char> disable_flags(desired_num_neurons);
        std::vector<double> model_x = model->get_x();
        std::vector<double> model_secondary(desired_num_neurons);
        std::vector<bool> model_fired = model->get_fired();
        std::vector<double> model_I_sync(desired_num_neurons);

        for (auto id = 0; id < desired_num_neurons; id++) {
            model_secondary[id] = model->get_secondary_variable(id);
            model_I_sync[id] = model->get_I_syn(id);
        }

        for (auto j = 0; j < 3; j++) {
            model->update_electrical_activity(empty_graph, disable_flags);

            const auto& current_x = model->get_x();
            const auto& current_fired = model->get_fired();

            for (auto id = 0; id < desired_num_neurons; id++) {
                EXPECT_EQ(current_x[id], model_x[id]);
                EXPECT_EQ(current_fired[id], model_fired[id]);

                EXPECT_EQ(model->get_I_syn(id), model_I_sync[id]);
                EXPECT_EQ(model->get_secondary_variable(id), model_secondary[id]);

                model_I_sync[id] = model->get_I_syn(id);
                model_secondary[id] = model->get_secondary_variable(id);
            }

            model_x = current_x;
            model_fired = current_fired;
        }

        const auto random_graph = generate_random_network_graph(desired_num_neurons, desired_num_neurons, 1.0);

        for (auto j = 0; j < 3; j++) {
            model->update_electrical_activity(random_graph, disable_flags);

            const auto& current_x = model->get_x();
            const auto& current_fired = model->get_fired();

            for (auto id = 0; id < desired_num_neurons; id++) {
                EXPECT_EQ(current_x[id], model_x[id]);
                EXPECT_EQ(current_fired[id], model_fired[id]);

                EXPECT_EQ(model->get_I_syn(id), model_I_sync[id]);
                EXPECT_EQ(model->get_secondary_variable(id), model_secondary[id]);

                model_I_sync[id] = model->get_I_syn(id);
                model_secondary[id] = model->get_secondary_variable(id);
            }

            model_x = current_x;
            model_fired = current_fired;
        }
    }
}


TEST(TestNeuronModels, testNeuronModelsUpdateActivityEnabled) {
    using namespace models;
    using urd = std::uniform_real_distribution<double>;
    using uid = std::uniform_int_distribution<unsigned int>;

    setup();

    for (auto i = 0; i < 1; i++) {
        const urd urd_desired_k(NeuronModels::min_k, NeuronModels::max_k);
        const urd urd_desired_tau_C(NeuronModels::min_tau_C, NeuronModels::max_tau_C);
        const urd urd_desired_beta(NeuronModels::min_beta, NeuronModels::max_beta);
        const uid uid_desired_h(NeuronModels::min_h, NeuronModels::max_h);
        const urd urd_desired_base_background_activity(NeuronModels::min_base_background_activity, NeuronModels::max_base_background_activity);
        const urd urd_desired_background_activity_mean(NeuronModels::min_background_activity_mean, NeuronModels::max_background_activity_mean);
        const urd urd_desired_background_activity_stddev(NeuronModels::min_background_activity_stddev, NeuronModels::max_background_activity_stddev);

        const urd urd_desired_x0(PoissonModel::min_x_0, PoissonModel::max_x_0);
        const uid urd_desired_refrac(PoissonModel::min_refrac_time, PoissonModel::max_refrac_time);
        const urd urd_desired_tau_x(PoissonModel::min_tau_x, PoissonModel::max_tau_x);

        const auto expected_k = urd_desired_k(mt);
        const auto expected_tau_C = urd_desired_tau_C(mt);
        const auto expected_beta = urd_desired_beta(mt);
        const auto expected_h = uid_desired_h(mt);
        const auto expected_base_background_activity = urd_desired_base_background_activity(mt);
        const auto expected_background_activity_mean = urd_desired_background_activity_mean(mt);
        const auto expected_background_activity_stddev = urd_desired_background_activity_stddev(mt);

        const auto expected_x0 = urd_desired_x0(mt);
        const auto expected_refrac = urd_desired_refrac(mt);
        const auto expected_tau_x = urd_desired_tau_x(mt);

        auto model = std::make_unique<PoissonModel>(expected_k, expected_tau_C, expected_beta, expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
            expected_x0, expected_tau_x, expected_refrac);

        const uid uid_num_neurons(1, num_neurons_test);
        const auto desired_num_neurons = uid_num_neurons(mt);

        const auto empty_graph = generate_random_network_graph(desired_num_neurons, 0, 1.0);

        model->init(desired_num_neurons);

        std::vector<char> disable_flags(desired_num_neurons, 1);
        std::vector<double> model_x = model->get_x();
        std::vector<double> model_secondary(desired_num_neurons);
        std::vector<bool> model_fired = model->get_fired();
        std::vector<double> model_I_sync(desired_num_neurons);

        for (auto id = 0; id < desired_num_neurons; id++) {
            model_secondary[id] = model->get_secondary_variable(id);
            model_I_sync[id] = model->get_I_syn(id);
        }

        for (auto j = 0; j < 3; j++) {
            model->update_electrical_activity(empty_graph, disable_flags);

            const auto& current_x = model->get_x();
            const auto& current_fired = model->get_fired();

            for (auto id = 0; id < desired_num_neurons; id++) {
                EXPECT_NE(current_x[id], model_x[id]);
                //EXPECT_EQ(current_fired[id], model_fired[id]);

                //EXPECT_EQ(model->get_I_syn(id), model_I_sync[id]);
                EXPECT_EQ(model->get_secondary_variable(id), model_secondary[id]);

                model_I_sync[id] = model->get_I_syn(id);
                model_secondary[id] = model->get_secondary_variable(id);
            }

            model_x = current_x;
            model_fired = current_fired;
        }

        const auto random_graph = generate_random_network_graph(desired_num_neurons, desired_num_neurons, 1.0);

        for (auto j = 0; j < 3; j++) {
            model->update_electrical_activity(random_graph, disable_flags);

            const auto& current_x = model->get_x();
            const auto& current_fired = model->get_fired();

            for (auto id = 0; id < desired_num_neurons; id++) {
                EXPECT_NE(current_x[id], model_x[id]);
                //EXPECT_EQ(current_fired[id], model_fired[id]);

                //EXPECT_EQ(model->get_I_syn(id), model_I_sync[id]);
                EXPECT_EQ(model->get_secondary_variable(id), model_secondary[id]);

                model_I_sync[id] = model->get_I_syn(id);
                model_secondary[id] = model->get_secondary_variable(id);
            }

            model_x = current_x;
            model_fired = current_fired;
        }
    }

    return;

    for (auto i = 0; i < iterations; i++) {
        const urd urd_desired_k(NeuronModels::min_k, NeuronModels::max_k);
        const urd urd_desired_tau_C(NeuronModels::min_tau_C, NeuronModels::max_tau_C);
        const urd urd_desired_beta(NeuronModels::min_beta, NeuronModels::max_beta);
        const uid uid_desired_h(NeuronModels::min_h, NeuronModels::max_h);
        const urd urd_desired_base_background_activity(NeuronModels::min_base_background_activity, NeuronModels::max_base_background_activity);
        const urd urd_desired_background_activity_mean(NeuronModels::min_background_activity_mean, NeuronModels::max_background_activity_mean);
        const urd urd_desired_background_activity_stddev(NeuronModels::min_background_activity_stddev, NeuronModels::max_background_activity_stddev);

        const urd urd_desired_a(IzhikevichModel::min_a, IzhikevichModel::max_a);
        const urd urd_desired_b(IzhikevichModel::min_b, IzhikevichModel::max_b);
        const urd urd_desired_c(IzhikevichModel::min_c, IzhikevichModel::max_c);
        const urd urd_desired_d(IzhikevichModel::min_d, IzhikevichModel::max_d);
        const urd urd_desired_V_spike(IzhikevichModel::min_V_spike, IzhikevichModel::max_V_spike);
        const urd urd_desired_k1(IzhikevichModel::min_k1, IzhikevichModel::max_k1);
        const urd urd_desired_k2(IzhikevichModel::min_k2, IzhikevichModel::max_k2);
        const urd urd_desired_k3(IzhikevichModel::min_k3, IzhikevichModel::max_k3);

        const auto expected_k = urd_desired_k(mt);
        const auto expected_tau_C = urd_desired_tau_C(mt);
        const auto expected_beta = urd_desired_beta(mt);
        const auto expected_h = uid_desired_h(mt);
        const auto expected_base_background_activity = urd_desired_base_background_activity(mt);
        const auto expected_background_activity_mean = urd_desired_background_activity_mean(mt);
        const auto expected_background_activity_stddev = urd_desired_background_activity_stddev(mt);

        const auto expected_a = urd_desired_a(mt);
        const auto expected_b = urd_desired_b(mt);
        const auto expected_c = urd_desired_c(mt);
        const auto expected_d = urd_desired_d(mt);
        const auto expected_V_spike = urd_desired_V_spike(mt);
        const auto expected_k1 = urd_desired_k1(mt);
        const auto expected_k2 = urd_desired_k2(mt);
        const auto expected_k3 = urd_desired_k3(mt);

        auto model = std::make_unique<IzhikevichModel>(expected_k, expected_tau_C, expected_beta, expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
            expected_a, expected_b, expected_c, expected_d, expected_V_spike, expected_k1, expected_k2, expected_k3);

        const uid uid_num_neurons(1, num_neurons_test);
        const auto desired_num_neurons = uid_num_neurons(mt);

        const auto empty_graph = generate_random_network_graph(desired_num_neurons, 0, 1.0);

        model->init(desired_num_neurons);

        std::vector<char> disable_flags(desired_num_neurons, 1);
        std::vector<double> model_x = model->get_x();
        std::vector<double> model_secondary(desired_num_neurons);
        std::vector<bool> model_fired = model->get_fired();
        std::vector<double> model_I_sync(desired_num_neurons);

        for (auto id = 0; id < desired_num_neurons; id++) {
            model_secondary[id] = model->get_secondary_variable(id);
            model_I_sync[id] = model->get_I_syn(id);
        }

        for (auto j = 0; j < 3; j++) {
            model->update_electrical_activity(empty_graph, disable_flags);

            const auto& current_x = model->get_x();
            const auto& current_fired = model->get_fired();

            for (auto id = 0; id < desired_num_neurons; id++) {
                EXPECT_NE(current_x[id], model_x[id]);
                //EXPECT_EQ(current_fired[id], model_fired[id]);

                //EXPECT_EQ(model->get_I_syn(id), model_I_sync[id]);
                EXPECT_NE(model->get_secondary_variable(id), model_secondary[id]);

                model_I_sync[id] = model->get_I_syn(id);
                model_secondary[id] = model->get_secondary_variable(id);
            }

            model_x = current_x;
            model_fired = current_fired;
        }

        const auto random_graph = generate_random_network_graph(desired_num_neurons, desired_num_neurons, 1.0);

        for (auto j = 0; j < 3; j++) {
            model->update_electrical_activity(random_graph, disable_flags);

            const auto& current_x = model->get_x();
            const auto& current_fired = model->get_fired();

            for (auto id = 0; id < desired_num_neurons; id++) {
                EXPECT_NE(current_x[id], model_x[id]);
                //EXPECT_EQ(current_fired[id], model_fired[id]);

                //EXPECT_EQ(model->get_I_syn(id), model_I_sync[id]);
                EXPECT_NE(model->get_secondary_variable(id), model_secondary[id]);

                model_I_sync[id] = model->get_I_syn(id);
                model_secondary[id] = model->get_secondary_variable(id);
            }

            model_x = current_x;
            model_fired = current_fired;
        }
    }

    for (auto i = 0; i < iterations; i++) {
        const urd urd_desired_k(NeuronModels::min_k, NeuronModels::max_k);
        const urd urd_desired_tau_C(NeuronModels::min_tau_C, NeuronModels::max_tau_C);
        const urd urd_desired_beta(NeuronModels::min_beta, NeuronModels::max_beta);
        const uid uid_desired_h(NeuronModels::min_h, NeuronModels::max_h);
        const urd urd_desired_base_background_activity(NeuronModels::min_base_background_activity, NeuronModels::max_base_background_activity);
        const urd urd_desired_background_activity_mean(NeuronModels::min_background_activity_mean, NeuronModels::max_background_activity_mean);
        const urd urd_desired_background_activity_stddev(NeuronModels::min_background_activity_stddev, NeuronModels::max_background_activity_stddev);

        const urd urd_desired_a(FitzHughNagumoModel::min_a, FitzHughNagumoModel::max_a);
        const urd urd_desired_b(FitzHughNagumoModel::min_b, FitzHughNagumoModel::max_b);
        const urd urd_desired_phi(FitzHughNagumoModel::min_phi, FitzHughNagumoModel::max_phi);

        const auto expected_k = urd_desired_k(mt);
        const auto expected_tau_C = urd_desired_tau_C(mt);
        const auto expected_beta = urd_desired_beta(mt);
        const auto expected_h = uid_desired_h(mt);
        const auto expected_base_background_activity = urd_desired_base_background_activity(mt);
        const auto expected_background_activity_mean = urd_desired_background_activity_mean(mt);
        const auto expected_background_activity_stddev = urd_desired_background_activity_stddev(mt);

        const auto expected_a = urd_desired_a(mt);
        const auto expected_b = urd_desired_b(mt);
        const auto expected_phi = urd_desired_phi(mt);

        auto model = std::make_unique<FitzHughNagumoModel>(expected_k, expected_tau_C, expected_beta, expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
            expected_a, expected_b, expected_phi);

        const uid uid_num_neurons(1, num_neurons_test);
        const auto desired_num_neurons = uid_num_neurons(mt);

        const auto empty_graph = generate_random_network_graph(desired_num_neurons, 0, 1.0);

        model->init(desired_num_neurons);

        std::vector<char> disable_flags(desired_num_neurons, 1);
        std::vector<double> model_x = model->get_x();
        std::vector<double> model_secondary(desired_num_neurons);
        std::vector<bool> model_fired = model->get_fired();
        std::vector<double> model_I_sync(desired_num_neurons);

        for (auto id = 0; id < desired_num_neurons; id++) {
            model_secondary[id] = model->get_secondary_variable(id);
            model_I_sync[id] = model->get_I_syn(id);
        }

        for (auto j = 0; j < 3; j++) {
            model->update_electrical_activity(empty_graph, disable_flags);

            const auto& current_x = model->get_x();
            const auto& current_fired = model->get_fired();

            for (auto id = 0; id < desired_num_neurons; id++) {
                EXPECT_NE(current_x[id], model_x[id]);
                //EXPECT_EQ(current_fired[id], model_fired[id]);

                //EXPECT_EQ(model->get_I_syn(id), model_I_sync[id]);
                EXPECT_NE(model->get_secondary_variable(id), model_secondary[id]);

                model_I_sync[id] = model->get_I_syn(id);
                model_secondary[id] = model->get_secondary_variable(id);
            }

            model_x = current_x;
            model_fired = current_fired;
        }

        const auto random_graph = generate_random_network_graph(desired_num_neurons, desired_num_neurons, 1.0);

        for (auto j = 0; j < 3; j++) {
            model->update_electrical_activity(random_graph, disable_flags);

            const auto& current_x = model->get_x();
            const auto& current_fired = model->get_fired();

            for (auto id = 0; id < desired_num_neurons; id++) {
                EXPECT_NE(current_x[id], model_x[id]);
                //EXPECT_EQ(current_fired[id], model_fired[id]);

                //EXPECT_EQ(model->get_I_syn(id), model_I_sync[id]);
                EXPECT_NE(model->get_secondary_variable(id), model_secondary[id]);

                model_I_sync[id] = model->get_I_syn(id);
                model_secondary[id] = model->get_secondary_variable(id);
            }

            model_x = current_x;
            model_fired = current_fired;
        }
    }

    for (auto i = 0; i < iterations; i++) {
        const urd urd_desired_k(NeuronModels::min_k, NeuronModels::max_k);
        const urd urd_desired_tau_C(NeuronModels::min_tau_C, NeuronModels::max_tau_C);
        const urd urd_desired_beta(NeuronModels::min_beta, NeuronModels::max_beta);
        const uid uid_desired_h(NeuronModels::min_h, NeuronModels::max_h);
        const urd urd_desired_base_background_activity(NeuronModels::min_base_background_activity, NeuronModels::max_base_background_activity);
        const urd urd_desired_background_activity_mean(NeuronModels::min_background_activity_mean, NeuronModels::max_background_activity_mean);
        const urd urd_desired_background_activity_stddev(NeuronModels::min_background_activity_stddev, NeuronModels::max_background_activity_stddev);

        const urd urd_desired_C(AEIFModel::min_C, AEIFModel::max_C);
        const urd urd_desired_g_L(AEIFModel::min_g_L, AEIFModel::max_g_L);
        const urd urd_desired_E_L(AEIFModel::min_E_L, AEIFModel::max_E_L);
        const urd urd_desired_V_T(AEIFModel::min_V_T, AEIFModel::max_V_T);
        const urd urd_desired_d_T(AEIFModel::min_d_T, AEIFModel::max_d_T);
        const urd urd_desired_tau_w(AEIFModel::min_tau_w, AEIFModel::max_tau_w);
        const urd urd_desired_a(AEIFModel::min_a, AEIFModel::max_a);
        const urd urd_desired_b(AEIFModel::min_b, AEIFModel::max_b);
        const urd urd_desired_V_peak(AEIFModel::min_V_peak, AEIFModel::max_V_peak);

        const auto expected_k = urd_desired_k(mt);
        const auto expected_tau_C = urd_desired_tau_C(mt);
        const auto expected_beta = urd_desired_beta(mt);
        const auto expected_h = uid_desired_h(mt);
        const auto expected_base_background_activity = urd_desired_base_background_activity(mt);
        const auto expected_background_activity_mean = urd_desired_background_activity_mean(mt);
        const auto expected_background_activity_stddev = urd_desired_background_activity_stddev(mt);

        const auto expected_C = urd_desired_C(mt);
        const auto expected_g_L = urd_desired_g_L(mt);
        const auto expected_E_L = urd_desired_E_L(mt);
        const auto expected_V_T = urd_desired_V_T(mt);
        const auto expected_d_T = urd_desired_d_T(mt);
        const auto expected_tau_w = urd_desired_tau_w(mt);
        const auto expected_a = urd_desired_a(mt);
        const auto expected_b = urd_desired_b(mt);
        const auto expected_V_peak = urd_desired_V_peak(mt);

        auto model = std::make_unique<AEIFModel>(expected_k, expected_tau_C, expected_beta, expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
            expected_C, expected_g_L, expected_E_L, expected_V_T, expected_d_T, expected_tau_w, expected_a, expected_b, expected_V_peak);

        const uid uid_num_neurons(1, num_neurons_test);
        const auto desired_num_neurons = uid_num_neurons(mt);

        const auto empty_graph = generate_random_network_graph(desired_num_neurons, 0, 1.0);

        model->init(desired_num_neurons);

        std::vector<char> disable_flags(desired_num_neurons, 1);
        std::vector<double> model_x = model->get_x();
        std::vector<double> model_secondary(desired_num_neurons);
        std::vector<bool> model_fired = model->get_fired();
        std::vector<double> model_I_sync(desired_num_neurons);

        for (auto id = 0; id < desired_num_neurons; id++) {
            model_secondary[id] = model->get_secondary_variable(id);
            model_I_sync[id] = model->get_I_syn(id);
        }

        for (auto j = 0; j < 3; j++) {
            model->update_electrical_activity(empty_graph, disable_flags);

            const auto& current_x = model->get_x();
            const auto& current_fired = model->get_fired();

            for (auto id = 0; id < desired_num_neurons; id++) {
                EXPECT_NE(current_x[id], model_x[id]);
                //EXPECT_EQ(current_fired[id], model_fired[id]);

                //EXPECT_EQ(model->get_I_syn(id), model_I_sync[id]);
                EXPECT_NE(model->get_secondary_variable(id), model_secondary[id]);

                model_I_sync[id] = model->get_I_syn(id);
                model_secondary[id] = model->get_secondary_variable(id);
            }

            model_x = current_x;
            model_fired = current_fired;
        }

        const auto random_graph = generate_random_network_graph(desired_num_neurons, desired_num_neurons, 1.0);

        for (auto j = 0; j < 3; j++) {
            model->update_electrical_activity(random_graph, disable_flags);

            const auto& current_x = model->get_x();
            const auto& current_fired = model->get_fired();

            for (auto id = 0; id < desired_num_neurons; id++) {
                EXPECT_NE(current_x[id], model_x[id]);
                //EXPECT_EQ(current_fired[id], model_fired[id]);

                //EXPECT_EQ(model->get_I_syn(id), model_I_sync[id]);
                EXPECT_NE(model->get_secondary_variable(id), model_secondary[id]);

                model_I_sync[id] = model->get_I_syn(id);
                model_secondary[id] = model->get_secondary_variable(id);
            }

            model_x = current_x;
            model_fired = current_fired;
        }
    }
}

TEST(TestNeurons, testNeuronsConstructor) {
    setup();

    auto partition = std::make_shared<Partition>(1, 0);

    auto model = std::make_unique<models::PoissonModel>();

    Neurons neurons{ partition, std::move(model) };
}
