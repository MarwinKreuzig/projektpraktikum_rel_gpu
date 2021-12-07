#include "../googletest/include/gtest/gtest.h"

#include "RelearnTest.hpp"

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

NetworkGraph generate_random_network_graph(size_t number_neurons, size_t num_synapses, double threshold_exc, std::mt19937& mt) {
    std::uniform_int_distribution<size_t> uid(0, number_neurons - 1);
    std::uniform_real_distribution<double> urd(0, 1.0);

    NetworkGraph ng(number_neurons, 0);

    for (size_t synapse_id = 0; synapse_id < num_synapses; synapse_id++) {
        const auto neuron_id_1 = uid(mt);
        auto neuron_id_2 = uid(mt);

        if (neuron_id_2 == neuron_id_1) {
            neuron_id_2 = (neuron_id_1 + 1) % number_neurons;
        }

        const auto uniform_double = urd(mt);
        const auto weight = (uniform_double < threshold_exc) ? 1 : -1;

        RankNeuronId target_id{ 0, neuron_id_1 };
        RankNeuronId source_id{ 0, neuron_id_2 };

        ng.add_edge_weight(target_id, source_id, weight);
    }

    return ng;
}

std::vector<size_t> generate_random_ids(size_t id_low, size_t id_high, size_t num_disables, std::mt19937& mt) {
    std::vector<size_t> disable_ids(num_disables);

    std::uniform_int_distribution<size_t> uid(id_low, id_high);

    for (size_t i = 0; i < num_disables; i++) {
        disable_ids[i] = uid(mt);
    }

    return disable_ids;
}

TEST_F(NeuronModelsTest, testNeuronModelsDefaultConstructorPoisson) {
    using namespace models;

    for (auto i = 0; i < iterations; i++) {
        const auto expected_k = NeuronModel::default_k;
        const auto expected_tau_C = NeuronModel::default_tau_C;
        const auto expected_beta = NeuronModel::default_beta;
        const auto expected_h = NeuronModel::default_h;
        const auto expected_base_background_activity = NeuronModel::default_base_background_activity;
        const auto expected_background_activity_mean = NeuronModel::default_background_activity_mean;
        const auto expected_background_activity_stddev = NeuronModel::default_background_activity_stddev;

        auto model = std::make_unique<PoissonModel>();

        ASSERT_EQ(expected_k, model->get_k());
        ASSERT_EQ(expected_tau_C, model->get_tau_C());
        ASSERT_EQ(expected_beta, model->get_beta());
        ASSERT_EQ(expected_h, model->get_h());
        ASSERT_EQ(expected_base_background_activity, model->get_base_background_activity());
        ASSERT_EQ(expected_background_activity_mean, model->get_background_activity_mean());
        ASSERT_EQ(expected_background_activity_stddev, model->get_background_activity_stddev());

        const auto expected_x0 = PoissonModel::default_x_0;
        const auto expected_refrac = PoissonModel::default_refrac_time;
        const auto expected_tau_x = PoissonModel::default_tau_x;

        ASSERT_EQ(expected_x0, model->get_x_0());
        ASSERT_EQ(expected_refrac, model->get_refrac_time());
        ASSERT_EQ(expected_tau_x, model->get_tau_x());
    }
}

TEST_F(NeuronModelsTest, testNeuronModelsDefaultConstructorIzhikevich) {
    using namespace models;

    for (auto i = 0; i < iterations; i++) {
        const auto expected_k = NeuronModel::default_k;
        const auto expected_tau_C = NeuronModel::default_tau_C;
        const auto expected_beta = NeuronModel::default_beta;
        const auto expected_h = NeuronModel::default_h;
        const auto expected_base_background_activity = NeuronModel::default_base_background_activity;
        const auto expected_background_activity_mean = NeuronModel::default_background_activity_mean;
        const auto expected_background_activity_stddev = NeuronModel::default_background_activity_stddev;

        auto model = std::make_unique<IzhikevichModel>();

        ASSERT_EQ(expected_k, model->get_k());
        ASSERT_EQ(expected_tau_C, model->get_tau_C());
        ASSERT_EQ(expected_beta, model->get_beta());
        ASSERT_EQ(expected_h, model->get_h());
        ASSERT_EQ(expected_base_background_activity, model->get_base_background_activity());
        ASSERT_EQ(expected_background_activity_mean, model->get_background_activity_mean());
        ASSERT_EQ(expected_background_activity_stddev, model->get_background_activity_stddev());

        const auto expected_a = IzhikevichModel::default_a;
        const auto expected_b = IzhikevichModel::default_b;
        const auto expected_c = IzhikevichModel::default_c;
        const auto expected_d = IzhikevichModel::default_d;
        const auto expected_V_spike = IzhikevichModel::default_V_spike;
        const auto expected_k1 = IzhikevichModel::default_k1;
        const auto expected_k2 = IzhikevichModel::default_k2;
        const auto expected_k3 = IzhikevichModel::default_k3;

        ASSERT_EQ(expected_a, model->get_a());
        ASSERT_EQ(expected_b, model->get_b());
        ASSERT_EQ(expected_c, model->get_c());
        ASSERT_EQ(expected_d, model->get_d());
        ASSERT_EQ(expected_V_spike, model->get_V_spike());
        ASSERT_EQ(expected_k1, model->get_k1());
        ASSERT_EQ(expected_k2, model->get_k2());
        ASSERT_EQ(expected_k3, model->get_k3());
    }
}

TEST_F(NeuronModelsTest, testNeuronModelsDefaultConstructorFitzHughNagumo) {
    using namespace models;

    for (auto i = 0; i < iterations; i++) {
        const auto expected_k = NeuronModel::default_k;
        const auto expected_tau_C = NeuronModel::default_tau_C;
        const auto expected_beta = NeuronModel::default_beta;
        const auto expected_h = NeuronModel::default_h;
        const auto expected_base_background_activity = NeuronModel::default_base_background_activity;
        const auto expected_background_activity_mean = NeuronModel::default_background_activity_mean;
        const auto expected_background_activity_stddev = NeuronModel::default_background_activity_stddev;

        auto model = std::make_unique<FitzHughNagumoModel>();

        ASSERT_EQ(expected_k, model->get_k());
        ASSERT_EQ(expected_tau_C, model->get_tau_C());
        ASSERT_EQ(expected_beta, model->get_beta());
        ASSERT_EQ(expected_h, model->get_h());
        ASSERT_EQ(expected_base_background_activity, model->get_base_background_activity());
        ASSERT_EQ(expected_background_activity_mean, model->get_background_activity_mean());
        ASSERT_EQ(expected_background_activity_stddev, model->get_background_activity_stddev());

        const auto expected_a = FitzHughNagumoModel::default_a;
        const auto expected_b = FitzHughNagumoModel::default_b;
        const auto expected_phi = FitzHughNagumoModel::default_phi;

        ASSERT_EQ(expected_a, model->get_a());
        ASSERT_EQ(expected_b, model->get_b());
        ASSERT_EQ(expected_phi, model->get_phi());
    }
}

TEST_F(NeuronModelsTest, testNeuronModelsDefaultConstructorAEIF) {
    using namespace models;

    for (auto i = 0; i < iterations; i++) {
        const auto expected_k = NeuronModel::default_k;
        const auto expected_tau_C = NeuronModel::default_tau_C;
        const auto expected_beta = NeuronModel::default_beta;
        const auto expected_h = NeuronModel::default_h;
        const auto expected_base_background_activity = NeuronModel::default_base_background_activity;
        const auto expected_background_activity_mean = NeuronModel::default_background_activity_mean;
        const auto expected_background_activity_stddev = NeuronModel::default_background_activity_stddev;

        auto model = std::make_unique<AEIFModel>();

        ASSERT_EQ(expected_k, model->get_k());
        ASSERT_EQ(expected_tau_C, model->get_tau_C());
        ASSERT_EQ(expected_beta, model->get_beta());
        ASSERT_EQ(expected_h, model->get_h());
        ASSERT_EQ(expected_base_background_activity, model->get_base_background_activity());
        ASSERT_EQ(expected_background_activity_mean, model->get_background_activity_mean());
        ASSERT_EQ(expected_background_activity_stddev, model->get_background_activity_stddev());

        const auto expected_C = AEIFModel::default_C;
        const auto expected_g_L = AEIFModel::default_g_L;
        const auto expected_E_L = AEIFModel::default_E_L;
        const auto expected_V_T = AEIFModel::default_V_T;
        const auto expected_d_T = AEIFModel::default_d_T;
        const auto expected_tau_w = AEIFModel::default_tau_w;
        const auto expected_a = AEIFModel::default_a;
        const auto expected_b = AEIFModel::default_b;
        const auto expected_V_spike = AEIFModel::default_V_spike;

        ASSERT_EQ(expected_C, model->get_C());
        ASSERT_EQ(expected_g_L, model->get_g_L());
        ASSERT_EQ(expected_E_L, model->get_E_L());
        ASSERT_EQ(expected_V_T, model->get_V_T());
        ASSERT_EQ(expected_d_T, model->get_d_T());
        ASSERT_EQ(expected_tau_w, model->get_tau_w());
        ASSERT_EQ(expected_a, model->get_a());
        ASSERT_EQ(expected_b, model->get_b());
        ASSERT_EQ(expected_V_spike, model->get_V_spike());
    }
}

TEST_F(NeuronModelsTest, testNeuronModelsRandomConstructorPoisson) {
    using namespace models;
    using urd = std::uniform_real_distribution<double>;
    using uid = std::uniform_int_distribution<unsigned int>;

    for (auto i = 0; i < iterations; i++) {
        std::uniform_real_distribution<double> urd_desired_k(NeuronModel::min_k, NeuronModel::max_k);
        std::uniform_real_distribution<double> urd_desired_tau_C(NeuronModel::min_tau_C, NeuronModel::max_tau_C);
        std::uniform_real_distribution<double> urd_desired_beta(NeuronModel::min_beta, NeuronModel::max_beta);
        std::uniform_int_distribution<unsigned int> uid_desired_h(NeuronModel::min_h, NeuronModel::max_h);
        std::uniform_real_distribution<double> urd_desired_base_background_activity(NeuronModel::min_base_background_activity, NeuronModel::max_base_background_activity);
        std::uniform_real_distribution<double> urd_desired_background_activity_mean(NeuronModel::min_background_activity_mean, NeuronModel::max_background_activity_mean);
        std::uniform_real_distribution<double> urd_desired_background_activity_stddev(NeuronModel::min_background_activity_stddev, NeuronModel::max_background_activity_stddev);

        std::uniform_real_distribution<double> urd_desired_x0(PoissonModel::min_x_0, PoissonModel::max_x_0);
        std::uniform_int_distribution<unsigned int> urd_desired_refrac(PoissonModel::min_refrac_time, PoissonModel::max_refrac_time);
        std::uniform_real_distribution<double> urd_desired_tau_x(PoissonModel::min_tau_x, PoissonModel::max_tau_x);

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

        ASSERT_EQ(expected_k, model->get_k());
        ASSERT_EQ(expected_tau_C, model->get_tau_C());
        ASSERT_EQ(expected_beta, model->get_beta());
        ASSERT_EQ(expected_h, model->get_h());
        ASSERT_EQ(expected_base_background_activity, model->get_base_background_activity());
        ASSERT_EQ(expected_background_activity_mean, model->get_background_activity_mean());
        ASSERT_EQ(expected_background_activity_stddev, model->get_background_activity_stddev());

        ASSERT_EQ(expected_x0, model->get_x_0());
        ASSERT_EQ(expected_refrac, model->get_refrac_time());
        ASSERT_EQ(expected_tau_x, model->get_tau_x());
    }
}

TEST_F(NeuronModelsTest, testNeuronModelsRandomConstructorIzhikevich) {
    using namespace models;
    using urd = std::uniform_real_distribution<double>;
    using uid = std::uniform_int_distribution<unsigned int>;

    for (auto i = 0; i < iterations; i++) {
        std::uniform_real_distribution<double> urd_desired_k(NeuronModel::min_k, NeuronModel::max_k);
        std::uniform_real_distribution<double> urd_desired_tau_C(NeuronModel::min_tau_C, NeuronModel::max_tau_C);
        std::uniform_real_distribution<double> urd_desired_beta(NeuronModel::min_beta, NeuronModel::max_beta);
        std::uniform_int_distribution<unsigned int> uid_desired_h(NeuronModel::min_h, NeuronModel::max_h);
        std::uniform_real_distribution<double> urd_desired_base_background_activity(NeuronModel::min_base_background_activity, NeuronModel::max_base_background_activity);
        std::uniform_real_distribution<double> urd_desired_background_activity_mean(NeuronModel::min_background_activity_mean, NeuronModel::max_background_activity_mean);
        std::uniform_real_distribution<double> urd_desired_background_activity_stddev(NeuronModel::min_background_activity_stddev, NeuronModel::max_background_activity_stddev);

        std::uniform_real_distribution<double> urd_desired_a(IzhikevichModel::min_a, IzhikevichModel::max_a);
        std::uniform_real_distribution<double> urd_desired_b(IzhikevichModel::min_b, IzhikevichModel::max_b);
        std::uniform_real_distribution<double> urd_desired_c(IzhikevichModel::min_c, IzhikevichModel::max_c);
        std::uniform_real_distribution<double> urd_desired_d(IzhikevichModel::min_d, IzhikevichModel::max_d);
        std::uniform_real_distribution<double> urd_desired_V_spike(IzhikevichModel::min_V_spike, IzhikevichModel::max_V_spike);
        std::uniform_real_distribution<double> urd_desired_k1(IzhikevichModel::min_k1, IzhikevichModel::max_k1);
        std::uniform_real_distribution<double> urd_desired_k2(IzhikevichModel::min_k2, IzhikevichModel::max_k2);
        std::uniform_real_distribution<double> urd_desired_k3(IzhikevichModel::min_k3, IzhikevichModel::max_k3);

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

        ASSERT_EQ(expected_k, model->get_k());
        ASSERT_EQ(expected_tau_C, model->get_tau_C());
        ASSERT_EQ(expected_beta, model->get_beta());
        ASSERT_EQ(expected_h, model->get_h());
        ASSERT_EQ(expected_base_background_activity, model->get_base_background_activity());
        ASSERT_EQ(expected_background_activity_mean, model->get_background_activity_mean());
        ASSERT_EQ(expected_background_activity_stddev, model->get_background_activity_stddev());

        ASSERT_EQ(expected_a, model->get_a());
        ASSERT_EQ(expected_b, model->get_b());
        ASSERT_EQ(expected_c, model->get_c());
        ASSERT_EQ(expected_d, model->get_d());
        ASSERT_EQ(expected_V_spike, model->get_V_spike());
        ASSERT_EQ(expected_k1, model->get_k1());
        ASSERT_EQ(expected_k2, model->get_k2());
        ASSERT_EQ(expected_k3, model->get_k3());
    }
}

TEST_F(NeuronModelsTest, testNeuronModelsRandomConstructorFitzHughNagumo) {
    using namespace models;
    using urd = std::uniform_real_distribution<double>;
    using uid = std::uniform_int_distribution<unsigned int>;

    for (auto i = 0; i < iterations; i++) {
        std::uniform_real_distribution<double> urd_desired_k(NeuronModel::min_k, NeuronModel::max_k);
        std::uniform_real_distribution<double> urd_desired_tau_C(NeuronModel::min_tau_C, NeuronModel::max_tau_C);
        std::uniform_real_distribution<double> urd_desired_beta(NeuronModel::min_beta, NeuronModel::max_beta);
        std::uniform_int_distribution<unsigned int> uid_desired_h(NeuronModel::min_h, NeuronModel::max_h);
        std::uniform_real_distribution<double> urd_desired_base_background_activity(NeuronModel::min_base_background_activity, NeuronModel::max_base_background_activity);
        std::uniform_real_distribution<double> urd_desired_background_activity_mean(NeuronModel::min_background_activity_mean, NeuronModel::max_background_activity_mean);
        std::uniform_real_distribution<double> urd_desired_background_activity_stddev(NeuronModel::min_background_activity_stddev, NeuronModel::max_background_activity_stddev);

        std::uniform_real_distribution<double> urd_desired_a(FitzHughNagumoModel::min_a, FitzHughNagumoModel::max_a);
        std::uniform_real_distribution<double> urd_desired_b(FitzHughNagumoModel::min_b, FitzHughNagumoModel::max_b);
        std::uniform_real_distribution<double> urd_desired_phi(FitzHughNagumoModel::min_phi, FitzHughNagumoModel::max_phi);

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

        ASSERT_EQ(expected_k, model->get_k());
        ASSERT_EQ(expected_tau_C, model->get_tau_C());
        ASSERT_EQ(expected_beta, model->get_beta());
        ASSERT_EQ(expected_h, model->get_h());
        ASSERT_EQ(expected_base_background_activity, model->get_base_background_activity());
        ASSERT_EQ(expected_background_activity_mean, model->get_background_activity_mean());
        ASSERT_EQ(expected_background_activity_stddev, model->get_background_activity_stddev());

        ASSERT_EQ(expected_a, model->get_a());
        ASSERT_EQ(expected_b, model->get_b());
        ASSERT_EQ(expected_phi, model->get_phi());
    }
}

TEST_F(NeuronModelsTest, testNeuronModelsRandomConstructorAEIF) {
    using namespace models;
    using urd = std::uniform_real_distribution<double>;
    using uid = std::uniform_int_distribution<unsigned int>;

    for (auto i = 0; i < iterations; i++) {
        std::uniform_real_distribution<double> urd_desired_k(NeuronModel::min_k, NeuronModel::max_k);
        std::uniform_real_distribution<double> urd_desired_tau_C(NeuronModel::min_tau_C, NeuronModel::max_tau_C);
        std::uniform_real_distribution<double> urd_desired_beta(NeuronModel::min_beta, NeuronModel::max_beta);
        std::uniform_int_distribution<unsigned int> uid_desired_h(NeuronModel::min_h, NeuronModel::max_h);
        std::uniform_real_distribution<double> urd_desired_base_background_activity(NeuronModel::min_base_background_activity, NeuronModel::max_base_background_activity);
        std::uniform_real_distribution<double> urd_desired_background_activity_mean(NeuronModel::min_background_activity_mean, NeuronModel::max_background_activity_mean);
        std::uniform_real_distribution<double> urd_desired_background_activity_stddev(NeuronModel::min_background_activity_stddev, NeuronModel::max_background_activity_stddev);

        std::uniform_real_distribution<double> urd_desired_C(AEIFModel::min_C, AEIFModel::max_C);
        std::uniform_real_distribution<double> urd_desired_g_L(AEIFModel::min_g_L, AEIFModel::max_g_L);
        std::uniform_real_distribution<double> urd_desired_E_L(AEIFModel::min_E_L, AEIFModel::max_E_L);
        std::uniform_real_distribution<double> urd_desired_V_T(AEIFModel::min_V_T, AEIFModel::max_V_T);
        std::uniform_real_distribution<double> urd_desired_d_T(AEIFModel::min_d_T, AEIFModel::max_d_T);
        std::uniform_real_distribution<double> urd_desired_tau_w(AEIFModel::min_tau_w, AEIFModel::max_tau_w);
        std::uniform_real_distribution<double> urd_desired_a(AEIFModel::min_a, AEIFModel::max_a);
        std::uniform_real_distribution<double> urd_desired_b(AEIFModel::min_b, AEIFModel::max_b);
        std::uniform_real_distribution<double> urd_desired_V_spike(AEIFModel::min_V_spike, AEIFModel::max_V_spike);

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
        const auto expected_V_spike = urd_desired_V_spike(mt);

        auto model = std::make_unique<AEIFModel>(expected_k, expected_tau_C, expected_beta, expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
            expected_C, expected_g_L, expected_E_L, expected_V_T, expected_d_T, expected_tau_w, expected_a, expected_b, expected_V_spike);

        ASSERT_EQ(expected_k, model->get_k());
        ASSERT_EQ(expected_tau_C, model->get_tau_C());
        ASSERT_EQ(expected_beta, model->get_beta());
        ASSERT_EQ(expected_h, model->get_h());
        ASSERT_EQ(expected_base_background_activity, model->get_base_background_activity());
        ASSERT_EQ(expected_background_activity_mean, model->get_background_activity_mean());
        ASSERT_EQ(expected_background_activity_stddev, model->get_background_activity_stddev());

        ASSERT_EQ(expected_C, model->get_C());
        ASSERT_EQ(expected_g_L, model->get_g_L());
        ASSERT_EQ(expected_E_L, model->get_E_L());
        ASSERT_EQ(expected_V_T, model->get_V_T());
        ASSERT_EQ(expected_d_T, model->get_d_T());
        ASSERT_EQ(expected_tau_w, model->get_tau_w());
        ASSERT_EQ(expected_a, model->get_a());
        ASSERT_EQ(expected_b, model->get_b());
        ASSERT_EQ(expected_V_spike, model->get_V_spike());
    }
}

TEST_F(NeuronModelsTest, testNeuronModelsClonePoisson) {
    using namespace models;
    using urd = std::uniform_real_distribution<double>;
    using uid = std::uniform_int_distribution<unsigned int>;

    for (auto i = 0; i < iterations; i++) {
        std::uniform_real_distribution<double> urd_desired_k(NeuronModel::min_k, NeuronModel::max_k);
        std::uniform_real_distribution<double> urd_desired_tau_C(NeuronModel::min_tau_C, NeuronModel::max_tau_C);
        std::uniform_real_distribution<double> urd_desired_beta(NeuronModel::min_beta, NeuronModel::max_beta);
        std::uniform_int_distribution<unsigned int> uid_desired_h(NeuronModel::min_h, NeuronModel::max_h);
        std::uniform_real_distribution<double> urd_desired_base_background_activity(NeuronModel::min_base_background_activity, NeuronModel::max_base_background_activity);
        std::uniform_real_distribution<double> urd_desired_background_activity_mean(NeuronModel::min_background_activity_mean, NeuronModel::max_background_activity_mean);
        std::uniform_real_distribution<double> urd_desired_background_activity_stddev(NeuronModel::min_background_activity_stddev, NeuronModel::max_background_activity_stddev);

        std::uniform_real_distribution<double> urd_desired_x0(PoissonModel::min_x_0, PoissonModel::max_x_0);
        std::uniform_int_distribution<unsigned int> urd_desired_refrac(PoissonModel::min_refrac_time, PoissonModel::max_refrac_time);
        std::uniform_real_distribution<double> urd_desired_tau_x(PoissonModel::min_tau_x, PoissonModel::max_tau_x);

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
        std::shared_ptr<NeuronModel> shared_version = std::move(cloned_model);
        auto cast_cloned_model = std::dynamic_pointer_cast<PoissonModel>(shared_version);

        ASSERT_EQ(expected_k, model->get_k());
        ASSERT_EQ(expected_tau_C, model->get_tau_C());
        ASSERT_EQ(expected_beta, model->get_beta());
        ASSERT_EQ(expected_h, model->get_h());
        ASSERT_EQ(expected_base_background_activity, model->get_base_background_activity());
        ASSERT_EQ(expected_background_activity_mean, model->get_background_activity_mean());
        ASSERT_EQ(expected_background_activity_stddev, model->get_background_activity_stddev());

        ASSERT_EQ(expected_x0, model->get_x_0());
        ASSERT_EQ(expected_refrac, model->get_refrac_time());
        ASSERT_EQ(expected_tau_x, model->get_tau_x());

        ASSERT_EQ(expected_k, cast_cloned_model->get_k());
        ASSERT_EQ(expected_tau_C, cast_cloned_model->get_tau_C());
        ASSERT_EQ(expected_beta, cast_cloned_model->get_beta());
        ASSERT_EQ(expected_h, cast_cloned_model->get_h());
        ASSERT_EQ(expected_base_background_activity, cast_cloned_model->get_base_background_activity());
        ASSERT_EQ(expected_background_activity_mean, cast_cloned_model->get_background_activity_mean());
        ASSERT_EQ(expected_background_activity_stddev, cast_cloned_model->get_background_activity_stddev());

        ASSERT_EQ(expected_x0, cast_cloned_model->get_x_0());
        ASSERT_EQ(expected_refrac, cast_cloned_model->get_refrac_time());
        ASSERT_EQ(expected_tau_x, cast_cloned_model->get_tau_x());
    }
}

TEST_F(NeuronModelsTest, testNeuronModelsCloneIzhikevich) {
    using namespace models;
    using urd = std::uniform_real_distribution<double>;
    using uid = std::uniform_int_distribution<unsigned int>;

    for (auto i = 0; i < iterations; i++) {
        std::uniform_real_distribution<double> urd_desired_k(NeuronModel::min_k, NeuronModel::max_k);
        std::uniform_real_distribution<double> urd_desired_tau_C(NeuronModel::min_tau_C, NeuronModel::max_tau_C);
        std::uniform_real_distribution<double> urd_desired_beta(NeuronModel::min_beta, NeuronModel::max_beta);
        std::uniform_int_distribution<unsigned int> uid_desired_h(NeuronModel::min_h, NeuronModel::max_h);
        std::uniform_real_distribution<double> urd_desired_base_background_activity(NeuronModel::min_base_background_activity, NeuronModel::max_base_background_activity);
        std::uniform_real_distribution<double> urd_desired_background_activity_mean(NeuronModel::min_background_activity_mean, NeuronModel::max_background_activity_mean);
        std::uniform_real_distribution<double> urd_desired_background_activity_stddev(NeuronModel::min_background_activity_stddev, NeuronModel::max_background_activity_stddev);

        std::uniform_real_distribution<double> urd_desired_a(IzhikevichModel::min_a, IzhikevichModel::max_a);
        std::uniform_real_distribution<double> urd_desired_b(IzhikevichModel::min_b, IzhikevichModel::max_b);
        std::uniform_real_distribution<double> urd_desired_c(IzhikevichModel::min_c, IzhikevichModel::max_c);
        std::uniform_real_distribution<double> urd_desired_d(IzhikevichModel::min_d, IzhikevichModel::max_d);
        std::uniform_real_distribution<double> urd_desired_V_spike(IzhikevichModel::min_V_spike, IzhikevichModel::max_V_spike);
        std::uniform_real_distribution<double> urd_desired_k1(IzhikevichModel::min_k1, IzhikevichModel::max_k1);
        std::uniform_real_distribution<double> urd_desired_k2(IzhikevichModel::min_k2, IzhikevichModel::max_k2);
        std::uniform_real_distribution<double> urd_desired_k3(IzhikevichModel::min_k3, IzhikevichModel::max_k3);

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
        std::shared_ptr<NeuronModel> shared_version = std::move(cloned_model);
        auto cast_cloned_model = std::dynamic_pointer_cast<IzhikevichModel>(shared_version);

        ASSERT_EQ(expected_k, model->get_k());
        ASSERT_EQ(expected_tau_C, model->get_tau_C());
        ASSERT_EQ(expected_beta, model->get_beta());
        ASSERT_EQ(expected_h, model->get_h());
        ASSERT_EQ(expected_base_background_activity, model->get_base_background_activity());
        ASSERT_EQ(expected_background_activity_mean, model->get_background_activity_mean());
        ASSERT_EQ(expected_background_activity_stddev, model->get_background_activity_stddev());

        ASSERT_EQ(expected_a, model->get_a());
        ASSERT_EQ(expected_b, model->get_b());
        ASSERT_EQ(expected_c, model->get_c());
        ASSERT_EQ(expected_d, model->get_d());
        ASSERT_EQ(expected_V_spike, model->get_V_spike());
        ASSERT_EQ(expected_k1, model->get_k1());
        ASSERT_EQ(expected_k2, model->get_k2());
        ASSERT_EQ(expected_k3, model->get_k3());

        ASSERT_EQ(expected_k, cast_cloned_model->get_k());
        ASSERT_EQ(expected_tau_C, cast_cloned_model->get_tau_C());
        ASSERT_EQ(expected_beta, cast_cloned_model->get_beta());
        ASSERT_EQ(expected_h, cast_cloned_model->get_h());
        ASSERT_EQ(expected_base_background_activity, cast_cloned_model->get_base_background_activity());
        ASSERT_EQ(expected_background_activity_mean, cast_cloned_model->get_background_activity_mean());
        ASSERT_EQ(expected_background_activity_stddev, cast_cloned_model->get_background_activity_stddev());

        ASSERT_EQ(expected_a, cast_cloned_model->get_a());
        ASSERT_EQ(expected_b, cast_cloned_model->get_b());
        ASSERT_EQ(expected_c, cast_cloned_model->get_c());
        ASSERT_EQ(expected_d, cast_cloned_model->get_d());
        ASSERT_EQ(expected_V_spike, cast_cloned_model->get_V_spike());
        ASSERT_EQ(expected_k1, cast_cloned_model->get_k1());
        ASSERT_EQ(expected_k2, cast_cloned_model->get_k2());
        ASSERT_EQ(expected_k3, cast_cloned_model->get_k3());
    }
}

TEST_F(NeuronModelsTest, testNeuronModelsCloneFitzHughNagumo) {
    using namespace models;
    using urd = std::uniform_real_distribution<double>;
    using uid = std::uniform_int_distribution<unsigned int>;

    for (auto i = 0; i < iterations; i++) {
        std::uniform_real_distribution<double> urd_desired_k(NeuronModel::min_k, NeuronModel::max_k);
        std::uniform_real_distribution<double> urd_desired_tau_C(NeuronModel::min_tau_C, NeuronModel::max_tau_C);
        std::uniform_real_distribution<double> urd_desired_beta(NeuronModel::min_beta, NeuronModel::max_beta);
        std::uniform_int_distribution<unsigned int> uid_desired_h(NeuronModel::min_h, NeuronModel::max_h);
        std::uniform_real_distribution<double> urd_desired_base_background_activity(NeuronModel::min_base_background_activity, NeuronModel::max_base_background_activity);
        std::uniform_real_distribution<double> urd_desired_background_activity_mean(NeuronModel::min_background_activity_mean, NeuronModel::max_background_activity_mean);
        std::uniform_real_distribution<double> urd_desired_background_activity_stddev(NeuronModel::min_background_activity_stddev, NeuronModel::max_background_activity_stddev);

        std::uniform_real_distribution<double> urd_desired_a(FitzHughNagumoModel::min_a, FitzHughNagumoModel::max_a);
        std::uniform_real_distribution<double> urd_desired_b(FitzHughNagumoModel::min_b, FitzHughNagumoModel::max_b);
        std::uniform_real_distribution<double> urd_desired_phi(FitzHughNagumoModel::min_phi, FitzHughNagumoModel::max_phi);

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
        std::shared_ptr<NeuronModel> shared_version = std::move(cloned_model);
        auto cast_cloned_model = std::dynamic_pointer_cast<FitzHughNagumoModel>(shared_version);

        ASSERT_EQ(expected_k, model->get_k());
        ASSERT_EQ(expected_tau_C, model->get_tau_C());
        ASSERT_EQ(expected_beta, model->get_beta());
        ASSERT_EQ(expected_h, model->get_h());
        ASSERT_EQ(expected_base_background_activity, model->get_base_background_activity());
        ASSERT_EQ(expected_background_activity_mean, model->get_background_activity_mean());
        ASSERT_EQ(expected_background_activity_stddev, model->get_background_activity_stddev());

        ASSERT_EQ(expected_a, model->get_a());
        ASSERT_EQ(expected_b, model->get_b());
        ASSERT_EQ(expected_phi, model->get_phi());

        ASSERT_EQ(expected_k, cast_cloned_model->get_k());
        ASSERT_EQ(expected_tau_C, cast_cloned_model->get_tau_C());
        ASSERT_EQ(expected_beta, cast_cloned_model->get_beta());
        ASSERT_EQ(expected_h, cast_cloned_model->get_h());
        ASSERT_EQ(expected_base_background_activity, cast_cloned_model->get_base_background_activity());
        ASSERT_EQ(expected_background_activity_mean, cast_cloned_model->get_background_activity_mean());
        ASSERT_EQ(expected_background_activity_stddev, cast_cloned_model->get_background_activity_stddev());

        ASSERT_EQ(expected_a, cast_cloned_model->get_a());
        ASSERT_EQ(expected_b, cast_cloned_model->get_b());
        ASSERT_EQ(expected_phi, cast_cloned_model->get_phi());
    }
}

TEST_F(NeuronModelsTest, testNeuronModelsCloneAEIF) {
    using namespace models;
    using urd = std::uniform_real_distribution<double>;
    using uid = std::uniform_int_distribution<unsigned int>;

    for (auto i = 0; i < iterations; i++) {
        std::uniform_real_distribution<double> urd_desired_k(NeuronModel::min_k, NeuronModel::max_k);
        std::uniform_real_distribution<double> urd_desired_tau_C(NeuronModel::min_tau_C, NeuronModel::max_tau_C);
        std::uniform_real_distribution<double> urd_desired_beta(NeuronModel::min_beta, NeuronModel::max_beta);
        std::uniform_int_distribution<unsigned int> uid_desired_h(NeuronModel::min_h, NeuronModel::max_h);
        std::uniform_real_distribution<double> urd_desired_base_background_activity(NeuronModel::min_base_background_activity, NeuronModel::max_base_background_activity);
        std::uniform_real_distribution<double> urd_desired_background_activity_mean(NeuronModel::min_background_activity_mean, NeuronModel::max_background_activity_mean);
        std::uniform_real_distribution<double> urd_desired_background_activity_stddev(NeuronModel::min_background_activity_stddev, NeuronModel::max_background_activity_stddev);

        std::uniform_real_distribution<double> urd_desired_C(AEIFModel::min_C, AEIFModel::max_C);
        std::uniform_real_distribution<double> urd_desired_g_L(AEIFModel::min_g_L, AEIFModel::max_g_L);
        std::uniform_real_distribution<double> urd_desired_E_L(AEIFModel::min_E_L, AEIFModel::max_E_L);
        std::uniform_real_distribution<double> urd_desired_V_T(AEIFModel::min_V_T, AEIFModel::max_V_T);
        std::uniform_real_distribution<double> urd_desired_d_T(AEIFModel::min_d_T, AEIFModel::max_d_T);
        std::uniform_real_distribution<double> urd_desired_tau_w(AEIFModel::min_tau_w, AEIFModel::max_tau_w);
        std::uniform_real_distribution<double> urd_desired_a(AEIFModel::min_a, AEIFModel::max_a);
        std::uniform_real_distribution<double> urd_desired_b(AEIFModel::min_b, AEIFModel::max_b);
        std::uniform_real_distribution<double> urd_desired_V_spike(AEIFModel::min_V_spike, AEIFModel::max_V_spike);

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
        const auto expected_V_spike = urd_desired_V_spike(mt);

        auto model = std::make_unique<AEIFModel>(expected_k, expected_tau_C, expected_beta, expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
            expected_C, expected_g_L, expected_E_L, expected_V_T, expected_d_T, expected_tau_w, expected_a, expected_b, expected_V_spike);

        auto cloned_model = model->clone();
        std::shared_ptr<NeuronModel> shared_version = std::move(cloned_model);
        auto cast_cloned_model = std::dynamic_pointer_cast<AEIFModel>(shared_version);

        ASSERT_EQ(expected_k, model->get_k());
        ASSERT_EQ(expected_tau_C, model->get_tau_C());
        ASSERT_EQ(expected_beta, model->get_beta());
        ASSERT_EQ(expected_h, model->get_h());
        ASSERT_EQ(expected_base_background_activity, model->get_base_background_activity());
        ASSERT_EQ(expected_background_activity_mean, model->get_background_activity_mean());
        ASSERT_EQ(expected_background_activity_stddev, model->get_background_activity_stddev());

        ASSERT_EQ(expected_C, model->get_C());
        ASSERT_EQ(expected_g_L, model->get_g_L());
        ASSERT_EQ(expected_E_L, model->get_E_L());
        ASSERT_EQ(expected_V_T, model->get_V_T());
        ASSERT_EQ(expected_d_T, model->get_d_T());
        ASSERT_EQ(expected_tau_w, model->get_tau_w());
        ASSERT_EQ(expected_a, model->get_a());
        ASSERT_EQ(expected_b, model->get_b());
        ASSERT_EQ(expected_V_spike, model->get_V_spike());

        ASSERT_EQ(expected_k, cast_cloned_model->get_k());
        ASSERT_EQ(expected_tau_C, cast_cloned_model->get_tau_C());
        ASSERT_EQ(expected_beta, cast_cloned_model->get_beta());
        ASSERT_EQ(expected_h, cast_cloned_model->get_h());
        ASSERT_EQ(expected_base_background_activity, cast_cloned_model->get_base_background_activity());
        ASSERT_EQ(expected_background_activity_mean, cast_cloned_model->get_background_activity_mean());
        ASSERT_EQ(expected_background_activity_stddev, cast_cloned_model->get_background_activity_stddev());

        ASSERT_EQ(expected_C, cast_cloned_model->get_C());
        ASSERT_EQ(expected_g_L, cast_cloned_model->get_g_L());
        ASSERT_EQ(expected_E_L, cast_cloned_model->get_E_L());
        ASSERT_EQ(expected_V_T, cast_cloned_model->get_V_T());
        ASSERT_EQ(expected_d_T, cast_cloned_model->get_d_T());
        ASSERT_EQ(expected_tau_w, cast_cloned_model->get_tau_w());
        ASSERT_EQ(expected_a, cast_cloned_model->get_a());
        ASSERT_EQ(expected_b, cast_cloned_model->get_b());
        ASSERT_EQ(expected_V_spike, cast_cloned_model->get_V_spike());
    }
}

TEST_F(NeuronModelsTest, testNeuronModelsCreatePoisson) {
    using namespace models;
    using urd = std::uniform_real_distribution<double>;
    using uid = std::uniform_int_distribution<unsigned int>;

    for (auto i = 0; i < iterations; i++) {
        std::uniform_real_distribution<double> urd_desired_k(NeuronModel::min_k, NeuronModel::max_k);
        std::uniform_real_distribution<double> urd_desired_tau_C(NeuronModel::min_tau_C, NeuronModel::max_tau_C);
        std::uniform_real_distribution<double> urd_desired_beta(NeuronModel::min_beta, NeuronModel::max_beta);
        std::uniform_int_distribution<unsigned int> uid_desired_h(NeuronModel::min_h, NeuronModel::max_h);
        std::uniform_real_distribution<double> urd_desired_base_background_activity(NeuronModel::min_base_background_activity, NeuronModel::max_base_background_activity);
        std::uniform_real_distribution<double> urd_desired_background_activity_mean(NeuronModel::min_background_activity_mean, NeuronModel::max_background_activity_mean);
        std::uniform_real_distribution<double> urd_desired_background_activity_stddev(NeuronModel::min_background_activity_stddev, NeuronModel::max_background_activity_stddev);

        std::uniform_real_distribution<double> urd_desired_x0(PoissonModel::min_x_0, PoissonModel::max_x_0);
        std::uniform_int_distribution<unsigned int> urd_desired_refrac(PoissonModel::min_refrac_time, PoissonModel::max_refrac_time);
        std::uniform_real_distribution<double> urd_desired_tau_x(PoissonModel::min_tau_x, PoissonModel::max_tau_x);

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

        auto model = NeuronModel::create<PoissonModel>(expected_k, expected_tau_C, expected_beta, expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
            expected_x0, expected_tau_x, expected_refrac);

        ASSERT_EQ(expected_k, model->get_k());
        ASSERT_EQ(expected_tau_C, model->get_tau_C());
        ASSERT_EQ(expected_beta, model->get_beta());
        ASSERT_EQ(expected_h, model->get_h());
        ASSERT_EQ(expected_base_background_activity, model->get_base_background_activity());
        ASSERT_EQ(expected_background_activity_mean, model->get_background_activity_mean());
        ASSERT_EQ(expected_background_activity_stddev, model->get_background_activity_stddev());

        ASSERT_EQ(expected_x0, model->get_x_0());
        ASSERT_EQ(expected_refrac, model->get_refrac_time());
        ASSERT_EQ(expected_tau_x, model->get_tau_x());
    }
}

TEST_F(NeuronModelsTest, testNeuronModelsCreateIzhikevich) {
    using namespace models;
    using urd = std::uniform_real_distribution<double>;
    using uid = std::uniform_int_distribution<unsigned int>;

    for (auto i = 0; i < iterations; i++) {
        std::uniform_real_distribution<double> urd_desired_k(NeuronModel::min_k, NeuronModel::max_k);
        std::uniform_real_distribution<double> urd_desired_tau_C(NeuronModel::min_tau_C, NeuronModel::max_tau_C);
        std::uniform_real_distribution<double> urd_desired_beta(NeuronModel::min_beta, NeuronModel::max_beta);
        std::uniform_int_distribution<unsigned int> uid_desired_h(NeuronModel::min_h, NeuronModel::max_h);
        std::uniform_real_distribution<double> urd_desired_base_background_activity(NeuronModel::min_base_background_activity, NeuronModel::max_base_background_activity);
        std::uniform_real_distribution<double> urd_desired_background_activity_mean(NeuronModel::min_background_activity_mean, NeuronModel::max_background_activity_mean);
        std::uniform_real_distribution<double> urd_desired_background_activity_stddev(NeuronModel::min_background_activity_stddev, NeuronModel::max_background_activity_stddev);

        std::uniform_real_distribution<double> urd_desired_a(IzhikevichModel::min_a, IzhikevichModel::max_a);
        std::uniform_real_distribution<double> urd_desired_b(IzhikevichModel::min_b, IzhikevichModel::max_b);
        std::uniform_real_distribution<double> urd_desired_c(IzhikevichModel::min_c, IzhikevichModel::max_c);
        std::uniform_real_distribution<double> urd_desired_d(IzhikevichModel::min_d, IzhikevichModel::max_d);
        std::uniform_real_distribution<double> urd_desired_V_spike(IzhikevichModel::min_V_spike, IzhikevichModel::max_V_spike);
        std::uniform_real_distribution<double> urd_desired_k1(IzhikevichModel::min_k1, IzhikevichModel::max_k1);
        std::uniform_real_distribution<double> urd_desired_k2(IzhikevichModel::min_k2, IzhikevichModel::max_k2);
        std::uniform_real_distribution<double> urd_desired_k3(IzhikevichModel::min_k3, IzhikevichModel::max_k3);

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

        auto model = NeuronModel::create<IzhikevichModel>(expected_k, expected_tau_C, expected_beta, expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
            expected_a, expected_b, expected_c, expected_d, expected_V_spike, expected_k1, expected_k2, expected_k3);

        ASSERT_EQ(expected_k, model->get_k());
        ASSERT_EQ(expected_tau_C, model->get_tau_C());
        ASSERT_EQ(expected_beta, model->get_beta());
        ASSERT_EQ(expected_h, model->get_h());
        ASSERT_EQ(expected_base_background_activity, model->get_base_background_activity());
        ASSERT_EQ(expected_background_activity_mean, model->get_background_activity_mean());
        ASSERT_EQ(expected_background_activity_stddev, model->get_background_activity_stddev());

        ASSERT_EQ(expected_a, model->get_a());
        ASSERT_EQ(expected_b, model->get_b());
        ASSERT_EQ(expected_c, model->get_c());
        ASSERT_EQ(expected_d, model->get_d());
        ASSERT_EQ(expected_V_spike, model->get_V_spike());
        ASSERT_EQ(expected_k1, model->get_k1());
        ASSERT_EQ(expected_k2, model->get_k2());
        ASSERT_EQ(expected_k3, model->get_k3());
    }
}

TEST_F(NeuronModelsTest, testNeuronModelsCreateFitzHughNagumo) {
    using namespace models;
    using urd = std::uniform_real_distribution<double>;
    using uid = std::uniform_int_distribution<unsigned int>;

    for (auto i = 0; i < iterations; i++) {
        std::uniform_real_distribution<double> urd_desired_k(NeuronModel::min_k, NeuronModel::max_k);
        std::uniform_real_distribution<double> urd_desired_tau_C(NeuronModel::min_tau_C, NeuronModel::max_tau_C);
        std::uniform_real_distribution<double> urd_desired_beta(NeuronModel::min_beta, NeuronModel::max_beta);
        std::uniform_int_distribution<unsigned int> uid_desired_h(NeuronModel::min_h, NeuronModel::max_h);
        std::uniform_real_distribution<double> urd_desired_base_background_activity(NeuronModel::min_base_background_activity, NeuronModel::max_base_background_activity);
        std::uniform_real_distribution<double> urd_desired_background_activity_mean(NeuronModel::min_background_activity_mean, NeuronModel::max_background_activity_mean);
        std::uniform_real_distribution<double> urd_desired_background_activity_stddev(NeuronModel::min_background_activity_stddev, NeuronModel::max_background_activity_stddev);

        std::uniform_real_distribution<double> urd_desired_a(FitzHughNagumoModel::min_a, FitzHughNagumoModel::max_a);
        std::uniform_real_distribution<double> urd_desired_b(FitzHughNagumoModel::min_b, FitzHughNagumoModel::max_b);
        std::uniform_real_distribution<double> urd_desired_phi(FitzHughNagumoModel::min_phi, FitzHughNagumoModel::max_phi);

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

        auto model = NeuronModel::create<FitzHughNagumoModel>(expected_k, expected_tau_C, expected_beta, expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
            expected_a, expected_b, expected_phi);

        ASSERT_EQ(expected_k, model->get_k());
        ASSERT_EQ(expected_tau_C, model->get_tau_C());
        ASSERT_EQ(expected_beta, model->get_beta());
        ASSERT_EQ(expected_h, model->get_h());
        ASSERT_EQ(expected_base_background_activity, model->get_base_background_activity());
        ASSERT_EQ(expected_background_activity_mean, model->get_background_activity_mean());
        ASSERT_EQ(expected_background_activity_stddev, model->get_background_activity_stddev());

        ASSERT_EQ(expected_a, model->get_a());
        ASSERT_EQ(expected_b, model->get_b());
        ASSERT_EQ(expected_phi, model->get_phi());
    }
}

TEST_F(NeuronModelsTest, testNeuronModelsCreateAEIF) {
    using namespace models;
    using urd = std::uniform_real_distribution<double>;
    using uid = std::uniform_int_distribution<unsigned int>;

    for (auto i = 0; i < iterations; i++) {
        std::uniform_real_distribution<double> urd_desired_k(NeuronModel::min_k, NeuronModel::max_k);
        std::uniform_real_distribution<double> urd_desired_tau_C(NeuronModel::min_tau_C, NeuronModel::max_tau_C);
        std::uniform_real_distribution<double> urd_desired_beta(NeuronModel::min_beta, NeuronModel::max_beta);
        std::uniform_int_distribution<unsigned int> uid_desired_h(NeuronModel::min_h, NeuronModel::max_h);
        std::uniform_real_distribution<double> urd_desired_base_background_activity(NeuronModel::min_base_background_activity, NeuronModel::max_base_background_activity);
        std::uniform_real_distribution<double> urd_desired_background_activity_mean(NeuronModel::min_background_activity_mean, NeuronModel::max_background_activity_mean);
        std::uniform_real_distribution<double> urd_desired_background_activity_stddev(NeuronModel::min_background_activity_stddev, NeuronModel::max_background_activity_stddev);

        std::uniform_real_distribution<double> urd_desired_C(AEIFModel::min_C, AEIFModel::max_C);
        std::uniform_real_distribution<double> urd_desired_g_L(AEIFModel::min_g_L, AEIFModel::max_g_L);
        std::uniform_real_distribution<double> urd_desired_E_L(AEIFModel::min_E_L, AEIFModel::max_E_L);
        std::uniform_real_distribution<double> urd_desired_V_T(AEIFModel::min_V_T, AEIFModel::max_V_T);
        std::uniform_real_distribution<double> urd_desired_d_T(AEIFModel::min_d_T, AEIFModel::max_d_T);
        std::uniform_real_distribution<double> urd_desired_tau_w(AEIFModel::min_tau_w, AEIFModel::max_tau_w);
        std::uniform_real_distribution<double> urd_desired_a(AEIFModel::min_a, AEIFModel::max_a);
        std::uniform_real_distribution<double> urd_desired_b(AEIFModel::min_b, AEIFModel::max_b);
        std::uniform_real_distribution<double> urd_desired_V_spike(AEIFModel::min_V_spike, AEIFModel::max_V_spike);

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
        const auto expected_V_spike = urd_desired_V_spike(mt);

        auto model = NeuronModel::create<AEIFModel>(expected_k, expected_tau_C, expected_beta, expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
            expected_C, expected_g_L, expected_E_L, expected_V_T, expected_d_T, expected_tau_w, expected_a, expected_b, expected_V_spike);

        ASSERT_EQ(expected_k, model->get_k());
        ASSERT_EQ(expected_tau_C, model->get_tau_C());
        ASSERT_EQ(expected_beta, model->get_beta());
        ASSERT_EQ(expected_h, model->get_h());
        ASSERT_EQ(expected_base_background_activity, model->get_base_background_activity());
        ASSERT_EQ(expected_background_activity_mean, model->get_background_activity_mean());
        ASSERT_EQ(expected_background_activity_stddev, model->get_background_activity_stddev());

        ASSERT_EQ(expected_C, model->get_C());
        ASSERT_EQ(expected_g_L, model->get_g_L());
        ASSERT_EQ(expected_E_L, model->get_E_L());
        ASSERT_EQ(expected_V_T, model->get_V_T());
        ASSERT_EQ(expected_d_T, model->get_d_T());
        ASSERT_EQ(expected_tau_w, model->get_tau_w());
        ASSERT_EQ(expected_a, model->get_a());
        ASSERT_EQ(expected_b, model->get_b());
        ASSERT_EQ(expected_V_spike, model->get_V_spike());
    }
}

TEST_F(NeuronModelsTest, testNeuronModelsInitPoisson) {
    using namespace models;
    using urd = std::uniform_real_distribution<double>;
    using uid = std::uniform_int_distribution<unsigned int>;

    for (auto i = 0; i < iterations; i++) {
        std::uniform_real_distribution<double> urd_desired_k(NeuronModel::min_k, NeuronModel::max_k);
        std::uniform_real_distribution<double> urd_desired_tau_C(NeuronModel::min_tau_C, NeuronModel::max_tau_C);
        std::uniform_real_distribution<double> urd_desired_beta(NeuronModel::min_beta, NeuronModel::max_beta);
        std::uniform_int_distribution<unsigned int> uid_desired_h(NeuronModel::min_h, NeuronModel::max_h);
        std::uniform_real_distribution<double> urd_desired_base_background_activity(NeuronModel::min_base_background_activity, NeuronModel::max_base_background_activity);
        std::uniform_real_distribution<double> urd_desired_background_activity_mean(NeuronModel::min_background_activity_mean, NeuronModel::max_background_activity_mean);
        std::uniform_real_distribution<double> urd_desired_background_activity_stddev(NeuronModel::min_background_activity_stddev, NeuronModel::max_background_activity_stddev);

        std::uniform_real_distribution<double> urd_desired_x0(PoissonModel::min_x_0, PoissonModel::max_x_0);
        std::uniform_int_distribution<unsigned int> urd_desired_refrac(PoissonModel::min_refrac_time, PoissonModel::max_refrac_time);
        std::uniform_real_distribution<double> urd_desired_tau_x(PoissonModel::min_tau_x, PoissonModel::max_tau_x);

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

        for (auto j = 0; j < iterations; j++) {
            const auto get_requested_number_neurons = get_random_number_neurons();

            model->init(get_requested_number_neurons);

            const auto number_neurons = model->get_num_neurons();

            const auto& x = model->get_x();
            const auto& fired = model->get_fired();

            ASSERT_EQ(get_requested_number_neurons, number_neurons);
            ASSERT_EQ(get_requested_number_neurons, x.size());
            ASSERT_EQ(get_requested_number_neurons, fired.size());

            for (size_t neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
                ASSERT_NO_THROW(auto tmp = model->get_x(neuron_id));
                ASSERT_NO_THROW(auto tmp = model->get_fired(neuron_id));
                ASSERT_NO_THROW(auto tmp = model->get_secondary_variable(neuron_id));
                ASSERT_NO_THROW(auto tmp = model->get_I_syn(neuron_id));
            }

            for (size_t neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
                ASSERT_THROW(auto tmp = model->get_x(neuron_id + number_neurons), RelearnException);
                ASSERT_THROW(auto tmp = model->get_fired(neuron_id + number_neurons), RelearnException);
                ASSERT_THROW(auto tmp = model->get_secondary_variable(neuron_id + number_neurons), RelearnException);
                ASSERT_THROW(auto tmp = model->get_I_syn(neuron_id + number_neurons), RelearnException);
            }
        }
    }
}

TEST_F(NeuronModelsTest, testNeuronModelsInitIzhikevich) {
    using namespace models;
    using urd = std::uniform_real_distribution<double>;
    using uid = std::uniform_int_distribution<unsigned int>;

    for (auto i = 0; i < iterations; i++) {
        std::uniform_real_distribution<double> urd_desired_k(NeuronModel::min_k, NeuronModel::max_k);
        std::uniform_real_distribution<double> urd_desired_tau_C(NeuronModel::min_tau_C, NeuronModel::max_tau_C);
        std::uniform_real_distribution<double> urd_desired_beta(NeuronModel::min_beta, NeuronModel::max_beta);
        std::uniform_int_distribution<unsigned int> uid_desired_h(NeuronModel::min_h, NeuronModel::max_h);
        std::uniform_real_distribution<double> urd_desired_base_background_activity(NeuronModel::min_base_background_activity, NeuronModel::max_base_background_activity);
        std::uniform_real_distribution<double> urd_desired_background_activity_mean(NeuronModel::min_background_activity_mean, NeuronModel::max_background_activity_mean);
        std::uniform_real_distribution<double> urd_desired_background_activity_stddev(NeuronModel::min_background_activity_stddev, NeuronModel::max_background_activity_stddev);

        std::uniform_real_distribution<double> urd_desired_a(IzhikevichModel::min_a, IzhikevichModel::max_a);
        std::uniform_real_distribution<double> urd_desired_b(IzhikevichModel::min_b, IzhikevichModel::max_b);
        std::uniform_real_distribution<double> urd_desired_c(IzhikevichModel::min_c, IzhikevichModel::max_c);
        std::uniform_real_distribution<double> urd_desired_d(IzhikevichModel::min_d, IzhikevichModel::max_d);
        std::uniform_real_distribution<double> urd_desired_V_spike(IzhikevichModel::min_V_spike, IzhikevichModel::max_V_spike);
        std::uniform_real_distribution<double> urd_desired_k1(IzhikevichModel::min_k1, IzhikevichModel::max_k1);
        std::uniform_real_distribution<double> urd_desired_k2(IzhikevichModel::min_k2, IzhikevichModel::max_k2);
        std::uniform_real_distribution<double> urd_desired_k3(IzhikevichModel::min_k3, IzhikevichModel::max_k3);

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

        for (auto j = 0; j < iterations; j++) {
            const auto get_requested_number_neurons = get_random_number_neurons();

            model->init(get_requested_number_neurons);

            const auto number_neurons = model->get_num_neurons();

            const auto& x = model->get_x();
            const auto& fired = model->get_fired();

            ASSERT_EQ(get_requested_number_neurons, number_neurons);
            ASSERT_EQ(get_requested_number_neurons, x.size());
            ASSERT_EQ(get_requested_number_neurons, fired.size());

            for (size_t neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
                ASSERT_NO_THROW(auto tmp = model->get_x(neuron_id));
                ASSERT_NO_THROW(auto tmp = model->get_fired(neuron_id));
                ASSERT_NO_THROW(auto tmp = model->get_secondary_variable(neuron_id));
                ASSERT_NO_THROW(auto tmp = model->get_I_syn(neuron_id));
            }

            for (size_t neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
                ASSERT_THROW(auto tmp = model->get_x(neuron_id + number_neurons), RelearnException);
                ASSERT_THROW(auto tmp = model->get_fired(neuron_id + number_neurons), RelearnException);
                ASSERT_THROW(auto tmp = model->get_secondary_variable(neuron_id + number_neurons), RelearnException);
                ASSERT_THROW(auto tmp = model->get_I_syn(neuron_id + number_neurons), RelearnException);
            }
        }
    }
}

TEST_F(NeuronModelsTest, testNeuronModelsInitFitzHughNagumo) {
    using namespace models;
    using urd = std::uniform_real_distribution<double>;
    using uid = std::uniform_int_distribution<unsigned int>;

    for (auto i = 0; i < iterations; i++) {
        std::uniform_real_distribution<double> urd_desired_k(NeuronModel::min_k, NeuronModel::max_k);
        std::uniform_real_distribution<double> urd_desired_tau_C(NeuronModel::min_tau_C, NeuronModel::max_tau_C);
        std::uniform_real_distribution<double> urd_desired_beta(NeuronModel::min_beta, NeuronModel::max_beta);
        std::uniform_int_distribution<unsigned int> uid_desired_h(NeuronModel::min_h, NeuronModel::max_h);
        std::uniform_real_distribution<double> urd_desired_base_background_activity(NeuronModel::min_base_background_activity, NeuronModel::max_base_background_activity);
        std::uniform_real_distribution<double> urd_desired_background_activity_mean(NeuronModel::min_background_activity_mean, NeuronModel::max_background_activity_mean);
        std::uniform_real_distribution<double> urd_desired_background_activity_stddev(NeuronModel::min_background_activity_stddev, NeuronModel::max_background_activity_stddev);

        std::uniform_real_distribution<double> urd_desired_a(FitzHughNagumoModel::min_a, FitzHughNagumoModel::max_a);
        std::uniform_real_distribution<double> urd_desired_b(FitzHughNagumoModel::min_b, FitzHughNagumoModel::max_b);
        std::uniform_real_distribution<double> urd_desired_phi(FitzHughNagumoModel::min_phi, FitzHughNagumoModel::max_phi);

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

        for (auto j = 0; j < iterations; j++) {
            const auto get_requested_number_neurons = get_random_number_neurons();

            model->init(get_requested_number_neurons);

            const auto number_neurons = model->get_num_neurons();

            const auto& x = model->get_x();
            const auto& fired = model->get_fired();

            ASSERT_EQ(get_requested_number_neurons, number_neurons);
            ASSERT_EQ(get_requested_number_neurons, x.size());
            ASSERT_EQ(get_requested_number_neurons, fired.size());

            for (size_t neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
                ASSERT_NO_THROW(auto tmp = model->get_x(neuron_id));
                ASSERT_NO_THROW(auto tmp = model->get_fired(neuron_id));
                ASSERT_NO_THROW(auto tmp = model->get_secondary_variable(neuron_id));
                ASSERT_NO_THROW(auto tmp = model->get_I_syn(neuron_id));
            }

            for (size_t neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
                ASSERT_THROW(auto tmp = model->get_x(neuron_id + number_neurons), RelearnException);
                ASSERT_THROW(auto tmp = model->get_fired(neuron_id + number_neurons), RelearnException);
                ASSERT_THROW(auto tmp = model->get_secondary_variable(neuron_id + number_neurons), RelearnException);
                ASSERT_THROW(auto tmp = model->get_I_syn(neuron_id + number_neurons), RelearnException);
            }
        }
    }
}

TEST_F(NeuronModelsTest, testNeuronModelsInitAEIF) {
    using namespace models;
    using urd = std::uniform_real_distribution<double>;
    using uid = std::uniform_int_distribution<unsigned int>;

    for (auto i = 0; i < iterations; i++) {
        std::uniform_real_distribution<double> urd_desired_k(NeuronModel::min_k, NeuronModel::max_k);
        std::uniform_real_distribution<double> urd_desired_tau_C(NeuronModel::min_tau_C, NeuronModel::max_tau_C);
        std::uniform_real_distribution<double> urd_desired_beta(NeuronModel::min_beta, NeuronModel::max_beta);
        std::uniform_int_distribution<unsigned int> uid_desired_h(NeuronModel::min_h, NeuronModel::max_h);
        std::uniform_real_distribution<double> urd_desired_base_background_activity(NeuronModel::min_base_background_activity, NeuronModel::max_base_background_activity);
        std::uniform_real_distribution<double> urd_desired_background_activity_mean(NeuronModel::min_background_activity_mean, NeuronModel::max_background_activity_mean);
        std::uniform_real_distribution<double> urd_desired_background_activity_stddev(NeuronModel::min_background_activity_stddev, NeuronModel::max_background_activity_stddev);

        std::uniform_real_distribution<double> urd_desired_C(AEIFModel::min_C, AEIFModel::max_C);
        std::uniform_real_distribution<double> urd_desired_g_L(AEIFModel::min_g_L, AEIFModel::max_g_L);
        std::uniform_real_distribution<double> urd_desired_E_L(AEIFModel::min_E_L, AEIFModel::max_E_L);
        std::uniform_real_distribution<double> urd_desired_V_T(AEIFModel::min_V_T, AEIFModel::max_V_T);
        std::uniform_real_distribution<double> urd_desired_d_T(AEIFModel::min_d_T, AEIFModel::max_d_T);
        std::uniform_real_distribution<double> urd_desired_tau_w(AEIFModel::min_tau_w, AEIFModel::max_tau_w);
        std::uniform_real_distribution<double> urd_desired_a(AEIFModel::min_a, AEIFModel::max_a);
        std::uniform_real_distribution<double> urd_desired_b(AEIFModel::min_b, AEIFModel::max_b);
        std::uniform_real_distribution<double> urd_desired_V_spike(AEIFModel::min_V_spike, AEIFModel::max_V_spike);

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
        const auto expected_V_spike = urd_desired_V_spike(mt);

        auto model = std::make_unique<AEIFModel>(expected_k, expected_tau_C, expected_beta, expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
            expected_C, expected_g_L, expected_E_L, expected_V_T, expected_d_T, expected_tau_w, expected_a, expected_b, expected_V_spike);

        for (auto j = 0; j < iterations; j++) {
            const auto get_requested_number_neurons = get_random_number_neurons();

            model->init(get_requested_number_neurons);

            const auto number_neurons = model->get_num_neurons();

            const auto& x = model->get_x();
            const auto& fired = model->get_fired();

            ASSERT_EQ(get_requested_number_neurons, number_neurons);
            ASSERT_EQ(get_requested_number_neurons, x.size());
            ASSERT_EQ(get_requested_number_neurons, fired.size());

            for (size_t neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
                ASSERT_NO_THROW(auto tmp = model->get_x(neuron_id));
                ASSERT_NO_THROW(auto tmp = model->get_fired(neuron_id));
                ASSERT_NO_THROW(auto tmp = model->get_secondary_variable(neuron_id));
                ASSERT_NO_THROW(auto tmp = model->get_I_syn(neuron_id));
            }

            for (size_t neuron_id = 0; neuron_id < number_neurons; neuron_id++) {
                ASSERT_THROW(auto tmp = model->get_x(neuron_id + number_neurons), RelearnException);
                ASSERT_THROW(auto tmp = model->get_fired(neuron_id + number_neurons), RelearnException);
                ASSERT_THROW(auto tmp = model->get_secondary_variable(neuron_id + number_neurons), RelearnException);
                ASSERT_THROW(auto tmp = model->get_I_syn(neuron_id + number_neurons), RelearnException);
            }
        }
    }
}

TEST_F(NeuronModelsTest, testNeuronModelsCreateNeuronsPoisson) {
    using namespace models;

    {
        std::uniform_real_distribution<double> urd_desired_k(NeuronModel::min_k, NeuronModel::max_k);
        std::uniform_real_distribution<double> urd_desired_tau_C(NeuronModel::min_tau_C, NeuronModel::max_tau_C);
        std::uniform_real_distribution<double> urd_desired_beta(NeuronModel::min_beta, NeuronModel::max_beta);
        std::uniform_int_distribution<unsigned int> uid_desired_h(NeuronModel::min_h, NeuronModel::max_h);
        std::uniform_real_distribution<double> urd_desired_base_background_activity(NeuronModel::min_base_background_activity, NeuronModel::max_base_background_activity);
        std::uniform_real_distribution<double> urd_desired_background_activity_mean(NeuronModel::min_background_activity_mean, NeuronModel::max_background_activity_mean);
        std::uniform_real_distribution<double> urd_desired_background_activity_stddev(NeuronModel::min_background_activity_stddev, NeuronModel::max_background_activity_stddev);

        std::uniform_real_distribution<double> urd_desired_x0(PoissonModel::min_x_0, PoissonModel::max_x_0);
        std::uniform_int_distribution<unsigned int> urd_desired_refrac(PoissonModel::min_refrac_time, PoissonModel::max_refrac_time);
        std::uniform_real_distribution<double> urd_desired_tau_x(PoissonModel::min_tau_x, PoissonModel::max_tau_x);

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

        size_t current_num_neurons = 0;

        for (auto j = 0; j < 3; j++) {
            const auto get_requested_number_neurons = get_random_number_neurons();
            current_num_neurons += get_requested_number_neurons;

            model->create_neurons(get_requested_number_neurons);

            const auto number_neurons = model->get_num_neurons();

            const auto& x = model->get_x();
            const auto& fired = model->get_fired();

            ASSERT_EQ(current_num_neurons, number_neurons);
            ASSERT_EQ(current_num_neurons, x.size());
            ASSERT_EQ(current_num_neurons, fired.size());

            for (size_t neuron_id = 0; neuron_id < current_num_neurons; neuron_id++) {
                ASSERT_NO_THROW(auto tmp = model->get_x(neuron_id));
                ASSERT_NO_THROW(auto tmp = model->get_fired(neuron_id));
                ASSERT_NO_THROW(auto tmp = model->get_secondary_variable(neuron_id));
                ASSERT_NO_THROW(auto tmp = model->get_I_syn(neuron_id));
            }

            for (size_t neuron_id = 0; neuron_id < current_num_neurons; neuron_id++) {
                ASSERT_THROW(auto tmp = model->get_x(neuron_id + current_num_neurons), RelearnException);
                ASSERT_THROW(auto tmp = model->get_fired(neuron_id + current_num_neurons), RelearnException);
                ASSERT_THROW(auto tmp = model->get_secondary_variable(neuron_id + current_num_neurons), RelearnException);
                ASSERT_THROW(auto tmp = model->get_I_syn(neuron_id + current_num_neurons), RelearnException);
            }
        }
    }
}

TEST_F(NeuronModelsTest, testNeuronModelsCreateNeuronsIzhikevich) {
    using namespace models;

    {
        std::uniform_real_distribution<double> urd_desired_k(NeuronModel::min_k, NeuronModel::max_k);
        std::uniform_real_distribution<double> urd_desired_tau_C(NeuronModel::min_tau_C, NeuronModel::max_tau_C);
        std::uniform_real_distribution<double> urd_desired_beta(NeuronModel::min_beta, NeuronModel::max_beta);
        std::uniform_int_distribution<unsigned int> uid_desired_h(NeuronModel::min_h, NeuronModel::max_h);
        std::uniform_real_distribution<double> urd_desired_base_background_activity(NeuronModel::min_base_background_activity, NeuronModel::max_base_background_activity);
        std::uniform_real_distribution<double> urd_desired_background_activity_mean(NeuronModel::min_background_activity_mean, NeuronModel::max_background_activity_mean);
        std::uniform_real_distribution<double> urd_desired_background_activity_stddev(NeuronModel::min_background_activity_stddev, NeuronModel::max_background_activity_stddev);

        std::uniform_real_distribution<double> urd_desired_a(IzhikevichModel::min_a, IzhikevichModel::max_a);
        std::uniform_real_distribution<double> urd_desired_b(IzhikevichModel::min_b, IzhikevichModel::max_b);
        std::uniform_real_distribution<double> urd_desired_c(IzhikevichModel::min_c, IzhikevichModel::max_c);
        std::uniform_real_distribution<double> urd_desired_d(IzhikevichModel::min_d, IzhikevichModel::max_d);
        std::uniform_real_distribution<double> urd_desired_V_spike(IzhikevichModel::min_V_spike, IzhikevichModel::max_V_spike);
        std::uniform_real_distribution<double> urd_desired_k1(IzhikevichModel::min_k1, IzhikevichModel::max_k1);
        std::uniform_real_distribution<double> urd_desired_k2(IzhikevichModel::min_k2, IzhikevichModel::max_k2);
        std::uniform_real_distribution<double> urd_desired_k3(IzhikevichModel::min_k3, IzhikevichModel::max_k3);

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

        size_t current_num_neurons = 0;

        for (auto j = 0; j < 3; j++) {
            const auto get_requested_number_neurons = get_random_number_neurons();
            current_num_neurons += get_requested_number_neurons;

            model->create_neurons(get_requested_number_neurons);

            const auto number_neurons = model->get_num_neurons();

            const auto& x = model->get_x();
            const auto& fired = model->get_fired();

            ASSERT_EQ(current_num_neurons, number_neurons);
            ASSERT_EQ(current_num_neurons, x.size());
            ASSERT_EQ(current_num_neurons, fired.size());

            for (size_t neuron_id = 0; neuron_id < current_num_neurons; neuron_id++) {
                ASSERT_NO_THROW(auto tmp = model->get_x(neuron_id));
                ASSERT_NO_THROW(auto tmp = model->get_fired(neuron_id));
                ASSERT_NO_THROW(auto tmp = model->get_secondary_variable(neuron_id));
                ASSERT_NO_THROW(auto tmp = model->get_I_syn(neuron_id));
            }

            for (size_t neuron_id = 0; neuron_id < current_num_neurons; neuron_id++) {
                ASSERT_THROW(auto tmp = model->get_x(neuron_id + current_num_neurons), RelearnException);
                ASSERT_THROW(auto tmp = model->get_fired(neuron_id + current_num_neurons), RelearnException);
                ASSERT_THROW(auto tmp = model->get_secondary_variable(neuron_id + current_num_neurons), RelearnException);
                ASSERT_THROW(auto tmp = model->get_I_syn(neuron_id + current_num_neurons), RelearnException);
            }
        }
    }
}

TEST_F(NeuronModelsTest, testNeuronModelsCreateNeuronsFitzHughNagumo) {
    using namespace models;

    {
        std::uniform_real_distribution<double> urd_desired_k(NeuronModel::min_k, NeuronModel::max_k);
        std::uniform_real_distribution<double> urd_desired_tau_C(NeuronModel::min_tau_C, NeuronModel::max_tau_C);
        std::uniform_real_distribution<double> urd_desired_beta(NeuronModel::min_beta, NeuronModel::max_beta);
        std::uniform_int_distribution<unsigned int> uid_desired_h(NeuronModel::min_h, NeuronModel::max_h);
        std::uniform_real_distribution<double> urd_desired_base_background_activity(NeuronModel::min_base_background_activity, NeuronModel::max_base_background_activity);
        std::uniform_real_distribution<double> urd_desired_background_activity_mean(NeuronModel::min_background_activity_mean, NeuronModel::max_background_activity_mean);
        std::uniform_real_distribution<double> urd_desired_background_activity_stddev(NeuronModel::min_background_activity_stddev, NeuronModel::max_background_activity_stddev);

        std::uniform_real_distribution<double> urd_desired_a(FitzHughNagumoModel::min_a, FitzHughNagumoModel::max_a);
        std::uniform_real_distribution<double> urd_desired_b(FitzHughNagumoModel::min_b, FitzHughNagumoModel::max_b);
        std::uniform_real_distribution<double> urd_desired_phi(FitzHughNagumoModel::min_phi, FitzHughNagumoModel::max_phi);

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

        size_t current_num_neurons = 0;

        for (auto j = 0; j < 3; j++) {
            const auto get_requested_number_neurons = get_random_number_neurons();
            current_num_neurons += get_requested_number_neurons;

            model->create_neurons(get_requested_number_neurons);

            const auto number_neurons = model->get_num_neurons();

            const auto& x = model->get_x();
            const auto& fired = model->get_fired();

            ASSERT_EQ(current_num_neurons, number_neurons);
            ASSERT_EQ(current_num_neurons, x.size());
            ASSERT_EQ(current_num_neurons, fired.size());

            for (size_t neuron_id = 0; neuron_id < current_num_neurons; neuron_id++) {
                ASSERT_NO_THROW(auto tmp = model->get_x(neuron_id));
                ASSERT_NO_THROW(auto tmp = model->get_fired(neuron_id));
                ASSERT_NO_THROW(auto tmp = model->get_secondary_variable(neuron_id));
                ASSERT_NO_THROW(auto tmp = model->get_I_syn(neuron_id));
            }

            for (size_t neuron_id = 0; neuron_id < current_num_neurons; neuron_id++) {
                ASSERT_THROW(auto tmp = model->get_x(neuron_id + current_num_neurons), RelearnException);
                ASSERT_THROW(auto tmp = model->get_fired(neuron_id + current_num_neurons), RelearnException);
                ASSERT_THROW(auto tmp = model->get_secondary_variable(neuron_id + current_num_neurons), RelearnException);
                ASSERT_THROW(auto tmp = model->get_I_syn(neuron_id + current_num_neurons), RelearnException);
            }
        }
    }
}

TEST_F(NeuronModelsTest, testNeuronModelsCreateNeuronsAEIF) {
    using namespace models;

    {
        std::uniform_real_distribution<double> urd_desired_k(NeuronModel::min_k, NeuronModel::max_k);
        std::uniform_real_distribution<double> urd_desired_tau_C(NeuronModel::min_tau_C, NeuronModel::max_tau_C);
        std::uniform_real_distribution<double> urd_desired_beta(NeuronModel::min_beta, NeuronModel::max_beta);
        std::uniform_int_distribution<unsigned int> uid_desired_h(NeuronModel::min_h, NeuronModel::max_h);
        std::uniform_real_distribution<double> urd_desired_base_background_activity(NeuronModel::min_base_background_activity, NeuronModel::max_base_background_activity);
        std::uniform_real_distribution<double> urd_desired_background_activity_mean(NeuronModel::min_background_activity_mean, NeuronModel::max_background_activity_mean);
        std::uniform_real_distribution<double> urd_desired_background_activity_stddev(NeuronModel::min_background_activity_stddev, NeuronModel::max_background_activity_stddev);

        std::uniform_real_distribution<double> urd_desired_C(AEIFModel::min_C, AEIFModel::max_C);
        std::uniform_real_distribution<double> urd_desired_g_L(AEIFModel::min_g_L, AEIFModel::max_g_L);
        std::uniform_real_distribution<double> urd_desired_E_L(AEIFModel::min_E_L, AEIFModel::max_E_L);
        std::uniform_real_distribution<double> urd_desired_V_T(AEIFModel::min_V_T, AEIFModel::max_V_T);
        std::uniform_real_distribution<double> urd_desired_d_T(AEIFModel::min_d_T, AEIFModel::max_d_T);
        std::uniform_real_distribution<double> urd_desired_tau_w(AEIFModel::min_tau_w, AEIFModel::max_tau_w);
        std::uniform_real_distribution<double> urd_desired_a(AEIFModel::min_a, AEIFModel::max_a);
        std::uniform_real_distribution<double> urd_desired_b(AEIFModel::min_b, AEIFModel::max_b);
        std::uniform_real_distribution<double> urd_desired_V_spike(AEIFModel::min_V_spike, AEIFModel::max_V_spike);

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
        const auto expected_V_spike = urd_desired_V_spike(mt);

        auto model = std::make_unique<AEIFModel>(expected_k, expected_tau_C, expected_beta, expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
            expected_C, expected_g_L, expected_E_L, expected_V_T, expected_d_T, expected_tau_w, expected_a, expected_b, expected_V_spike);

        size_t current_num_neurons = 0;

        for (auto j = 0; j < 3; j++) {
            const auto get_requested_number_neurons = get_random_number_neurons();
            current_num_neurons += get_requested_number_neurons;

            model->create_neurons(get_requested_number_neurons);

            const auto number_neurons = model->get_num_neurons();

            const auto& x = model->get_x();
            const auto& fired = model->get_fired();

            ASSERT_EQ(current_num_neurons, number_neurons);
            ASSERT_EQ(current_num_neurons, x.size());
            ASSERT_EQ(current_num_neurons, fired.size());

            for (size_t neuron_id = 0; neuron_id < current_num_neurons; neuron_id++) {
                ASSERT_NO_THROW(auto tmp = model->get_x(neuron_id));
                ASSERT_NO_THROW(auto tmp = model->get_fired(neuron_id));
                ASSERT_NO_THROW(auto tmp = model->get_secondary_variable(neuron_id));
                ASSERT_NO_THROW(auto tmp = model->get_I_syn(neuron_id));
            }

            for (size_t neuron_id = 0; neuron_id < current_num_neurons; neuron_id++) {
                ASSERT_THROW(auto tmp = model->get_x(neuron_id + current_num_neurons), RelearnException);
                ASSERT_THROW(auto tmp = model->get_fired(neuron_id + current_num_neurons), RelearnException);
                ASSERT_THROW(auto tmp = model->get_secondary_variable(neuron_id + current_num_neurons), RelearnException);
                ASSERT_THROW(auto tmp = model->get_I_syn(neuron_id + current_num_neurons), RelearnException);
            }
        }
    }
}

TEST_F(NeuronModelsTest, testNeuronModelsDisableFiredPoisson) {
    using namespace models;

    for (auto i = 0; i < iterations; i++) {
        std::uniform_real_distribution<double> urd_desired_k(NeuronModel::min_k, NeuronModel::max_k);
        std::uniform_real_distribution<double> urd_desired_tau_C(NeuronModel::min_tau_C, NeuronModel::max_tau_C);
        std::uniform_real_distribution<double> urd_desired_beta(NeuronModel::min_beta, NeuronModel::max_beta);
        std::uniform_int_distribution<unsigned int> uid_desired_h(NeuronModel::min_h, NeuronModel::max_h);
        std::uniform_real_distribution<double> urd_desired_base_background_activity(NeuronModel::min_base_background_activity, NeuronModel::max_base_background_activity);
        std::uniform_real_distribution<double> urd_desired_background_activity_mean(NeuronModel::min_background_activity_mean, NeuronModel::max_background_activity_mean);
        std::uniform_real_distribution<double> urd_desired_background_activity_stddev(NeuronModel::min_background_activity_stddev, NeuronModel::max_background_activity_stddev);

        std::uniform_real_distribution<double> urd_desired_x0(PoissonModel::min_x_0, PoissonModel::max_x_0);
        std::uniform_int_distribution<unsigned int> urd_desired_refrac(PoissonModel::min_refrac_time, PoissonModel::max_refrac_time);
        std::uniform_real_distribution<double> urd_desired_tau_x(PoissonModel::min_tau_x, PoissonModel::max_tau_x);

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

        const auto number_neurons = get_random_number_neurons();

        std::uniform_int_distribution<unsigned int> uid_num_neurons_disables(1, number_neurons);
        const auto num_disables = uid_num_neurons_disables(mt);

        const auto disable_ids = generate_random_ids(0, number_neurons - 1, num_disables, mt);

        model->init(number_neurons);
        model->disable_neurons(disable_ids);

        for (unsigned int id = 0; id < number_neurons; id++) {
            if (std::find(disable_ids.cbegin(), disable_ids.cend(), id) != disable_ids.cend()) {
                ASSERT_FALSE(model->get_fired(id));
            }
        }

        const auto disable_ids_failure = generate_random_ids(number_neurons, number_neurons + number_neurons, num_disables, mt);

        ASSERT_THROW(model->disable_neurons(disable_ids_failure), RelearnException);
    }
}

TEST_F(NeuronModelsTest, testNeuronModelsDisableFiredIzhikevich) {
    using namespace models;

    for (auto i = 0; i < iterations; i++) {
        std::uniform_real_distribution<double> urd_desired_k(NeuronModel::min_k, NeuronModel::max_k);
        std::uniform_real_distribution<double> urd_desired_tau_C(NeuronModel::min_tau_C, NeuronModel::max_tau_C);
        std::uniform_real_distribution<double> urd_desired_beta(NeuronModel::min_beta, NeuronModel::max_beta);
        std::uniform_int_distribution<unsigned int> uid_desired_h(NeuronModel::min_h, NeuronModel::max_h);
        std::uniform_real_distribution<double> urd_desired_base_background_activity(NeuronModel::min_base_background_activity, NeuronModel::max_base_background_activity);
        std::uniform_real_distribution<double> urd_desired_background_activity_mean(NeuronModel::min_background_activity_mean, NeuronModel::max_background_activity_mean);
        std::uniform_real_distribution<double> urd_desired_background_activity_stddev(NeuronModel::min_background_activity_stddev, NeuronModel::max_background_activity_stddev);

        std::uniform_real_distribution<double> urd_desired_a(IzhikevichModel::min_a, IzhikevichModel::max_a);
        std::uniform_real_distribution<double> urd_desired_b(IzhikevichModel::min_b, IzhikevichModel::max_b);
        std::uniform_real_distribution<double> urd_desired_c(IzhikevichModel::min_c, IzhikevichModel::max_c);
        std::uniform_real_distribution<double> urd_desired_d(IzhikevichModel::min_d, IzhikevichModel::max_d);
        std::uniform_real_distribution<double> urd_desired_V_spike(IzhikevichModel::min_V_spike, IzhikevichModel::max_V_spike);
        std::uniform_real_distribution<double> urd_desired_k1(IzhikevichModel::min_k1, IzhikevichModel::max_k1);
        std::uniform_real_distribution<double> urd_desired_k2(IzhikevichModel::min_k2, IzhikevichModel::max_k2);
        std::uniform_real_distribution<double> urd_desired_k3(IzhikevichModel::min_k3, IzhikevichModel::max_k3);

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

        const auto number_neurons = get_random_number_neurons();

        std::uniform_int_distribution<unsigned int> uid_num_neurons_disables(1, number_neurons);
        const auto num_disables = uid_num_neurons_disables(mt);

        const auto disable_ids = generate_random_ids(0, number_neurons - 1, num_disables, mt);

        model->init(number_neurons);
        model->disable_neurons(disable_ids);

        for (unsigned int id = 0; id < number_neurons; id++) {
            if (std::find(disable_ids.cbegin(), disable_ids.cend(), id) != disable_ids.cend()) {
                ASSERT_FALSE(model->get_fired(id));
            }
        }

        const auto disable_ids_failure = generate_random_ids(number_neurons, number_neurons + number_neurons, num_disables, mt);

        ASSERT_THROW(model->disable_neurons(disable_ids_failure), RelearnException);
    }
}

TEST_F(NeuronModelsTest, testNeuronModelsDisableFiredFitzHughNagumo) {
    using namespace models;

    for (auto i = 0; i < iterations; i++) {
        std::uniform_real_distribution<double> urd_desired_k(NeuronModel::min_k, NeuronModel::max_k);
        std::uniform_real_distribution<double> urd_desired_tau_C(NeuronModel::min_tau_C, NeuronModel::max_tau_C);
        std::uniform_real_distribution<double> urd_desired_beta(NeuronModel::min_beta, NeuronModel::max_beta);
        std::uniform_int_distribution<unsigned int> uid_desired_h(NeuronModel::min_h, NeuronModel::max_h);
        std::uniform_real_distribution<double> urd_desired_base_background_activity(NeuronModel::min_base_background_activity, NeuronModel::max_base_background_activity);
        std::uniform_real_distribution<double> urd_desired_background_activity_mean(NeuronModel::min_background_activity_mean, NeuronModel::max_background_activity_mean);
        std::uniform_real_distribution<double> urd_desired_background_activity_stddev(NeuronModel::min_background_activity_stddev, NeuronModel::max_background_activity_stddev);

        std::uniform_real_distribution<double> urd_desired_a(FitzHughNagumoModel::min_a, FitzHughNagumoModel::max_a);
        std::uniform_real_distribution<double> urd_desired_b(FitzHughNagumoModel::min_b, FitzHughNagumoModel::max_b);
        std::uniform_real_distribution<double> urd_desired_phi(FitzHughNagumoModel::min_phi, FitzHughNagumoModel::max_phi);

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

        const auto number_neurons = get_random_number_neurons();

        std::uniform_int_distribution<unsigned int> uid_num_neurons_disables(1, number_neurons);
        const auto num_disables = uid_num_neurons_disables(mt);

        const auto disable_ids = generate_random_ids(0, number_neurons - 1, num_disables, mt);

        model->init(number_neurons);
        model->disable_neurons(disable_ids);

        for (unsigned int id = 0; id < number_neurons; id++) {
            if (std::find(disable_ids.cbegin(), disable_ids.cend(), id) != disable_ids.cend()) {
                ASSERT_FALSE(model->get_fired(id));
            }
        }

        const auto disable_ids_failure = generate_random_ids(number_neurons, number_neurons + number_neurons, num_disables, mt);

        ASSERT_THROW(model->disable_neurons(disable_ids_failure), RelearnException);
    }
}

TEST_F(NeuronModelsTest, testNeuronModelsDisableFiredAEIF) {
    using namespace models;

    for (auto i = 0; i < iterations; i++) {
        std::uniform_real_distribution<double> urd_desired_k(NeuronModel::min_k, NeuronModel::max_k);
        std::uniform_real_distribution<double> urd_desired_tau_C(NeuronModel::min_tau_C, NeuronModel::max_tau_C);
        std::uniform_real_distribution<double> urd_desired_beta(NeuronModel::min_beta, NeuronModel::max_beta);
        std::uniform_int_distribution<unsigned int> uid_desired_h(NeuronModel::min_h, NeuronModel::max_h);
        std::uniform_real_distribution<double> urd_desired_base_background_activity(NeuronModel::min_base_background_activity, NeuronModel::max_base_background_activity);
        std::uniform_real_distribution<double> urd_desired_background_activity_mean(NeuronModel::min_background_activity_mean, NeuronModel::max_background_activity_mean);
        std::uniform_real_distribution<double> urd_desired_background_activity_stddev(NeuronModel::min_background_activity_stddev, NeuronModel::max_background_activity_stddev);

        std::uniform_real_distribution<double> urd_desired_C(AEIFModel::min_C, AEIFModel::max_C);
        std::uniform_real_distribution<double> urd_desired_g_L(AEIFModel::min_g_L, AEIFModel::max_g_L);
        std::uniform_real_distribution<double> urd_desired_E_L(AEIFModel::min_E_L, AEIFModel::max_E_L);
        std::uniform_real_distribution<double> urd_desired_V_T(AEIFModel::min_V_T, AEIFModel::max_V_T);
        std::uniform_real_distribution<double> urd_desired_d_T(AEIFModel::min_d_T, AEIFModel::max_d_T);
        std::uniform_real_distribution<double> urd_desired_tau_w(AEIFModel::min_tau_w, AEIFModel::max_tau_w);
        std::uniform_real_distribution<double> urd_desired_a(AEIFModel::min_a, AEIFModel::max_a);
        std::uniform_real_distribution<double> urd_desired_b(AEIFModel::min_b, AEIFModel::max_b);
        std::uniform_real_distribution<double> urd_desired_V_spike(AEIFModel::min_V_spike, AEIFModel::max_V_spike);

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
        const auto expected_V_spike = urd_desired_V_spike(mt);

        auto model = std::make_unique<AEIFModel>(expected_k, expected_tau_C, expected_beta, expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
            expected_C, expected_g_L, expected_E_L, expected_V_T, expected_d_T, expected_tau_w, expected_a, expected_b, expected_V_spike);

        const auto number_neurons = get_random_number_neurons();

        std::uniform_int_distribution<unsigned int> uid_num_neurons_disables(1, number_neurons);
        const auto num_disables = uid_num_neurons_disables(mt);

        const auto disable_ids = generate_random_ids(0, number_neurons - 1, num_disables, mt);

        model->init(number_neurons);
        model->disable_neurons(disable_ids);

        for (unsigned int id = 0; id < number_neurons; id++) {
            if (std::find(disable_ids.cbegin(), disable_ids.cend(), id) != disable_ids.cend()) {
                ASSERT_FALSE(model->get_fired(id));
            }
        }

        const auto disable_ids_failure = generate_random_ids(number_neurons, number_neurons + number_neurons, num_disables, mt);

        ASSERT_THROW(model->disable_neurons(disable_ids_failure), RelearnException);
    }
}

TEST_F(NeuronModelsTest, testNeuronModelsUpdateActivityDisabledPoisson) {
    using namespace models;

    for (auto i = 0; i < iterations; i++) {
        std::uniform_real_distribution<double> urd_desired_k(NeuronModel::min_k, NeuronModel::max_k);
        std::uniform_real_distribution<double> urd_desired_tau_C(NeuronModel::min_tau_C, NeuronModel::max_tau_C);
        std::uniform_real_distribution<double> urd_desired_beta(NeuronModel::min_beta, NeuronModel::max_beta);
        std::uniform_int_distribution<unsigned int> uid_desired_h(NeuronModel::min_h, NeuronModel::max_h);
        std::uniform_real_distribution<double> urd_desired_base_background_activity(NeuronModel::min_base_background_activity, NeuronModel::max_base_background_activity);
        std::uniform_real_distribution<double> urd_desired_background_activity_mean(NeuronModel::min_background_activity_mean, NeuronModel::max_background_activity_mean);
        std::uniform_real_distribution<double> urd_desired_background_activity_stddev(NeuronModel::min_background_activity_stddev, NeuronModel::max_background_activity_stddev);

        std::uniform_real_distribution<double> urd_desired_x0(PoissonModel::min_x_0, PoissonModel::max_x_0);
        std::uniform_int_distribution<unsigned int> urd_desired_refrac(PoissonModel::min_refrac_time, PoissonModel::max_refrac_time);
        std::uniform_real_distribution<double> urd_desired_tau_x(PoissonModel::min_tau_x, PoissonModel::max_tau_x);

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

        const auto get_requested_number_neurons = get_random_number_neurons();

        const auto empty_graph = generate_random_network_graph(get_requested_number_neurons, 0, 1.0, mt);

        model->init(get_requested_number_neurons);

        std::vector<char> disable_flags(get_requested_number_neurons);
        std::vector<double> model_x = model->get_x();
        std::vector<double> model_secondary(get_requested_number_neurons);
        std::vector<char> model_fired = model->get_fired();
        std::vector<double> model_I_sync(get_requested_number_neurons);

        for (unsigned int id = 0; id < get_requested_number_neurons; id++) {
            model_secondary[id] = model->get_secondary_variable(id);
            model_I_sync[id] = model->get_I_syn(id);
        }

        for (auto j = 0; j < 3; j++) {
            model->update_electrical_activity(empty_graph, disable_flags);

            const auto& current_x = model->get_x();
            const auto& current_fired = model->get_fired();

            for (unsigned int id = 0; id < get_requested_number_neurons; id++) {
                ASSERT_EQ(current_x[id], model_x[id]);
                ASSERT_EQ(current_fired[id], model_fired[id]);

                ASSERT_EQ(model->get_I_syn(id), model_I_sync[id]);
                ASSERT_EQ(model->get_secondary_variable(id), model_secondary[id]);

                model_I_sync[id] = model->get_I_syn(id);
                model_secondary[id] = model->get_secondary_variable(id);
            }

            model_x = current_x;
            model_fired = current_fired;
        }

        const auto random_graph = generate_random_network_graph(get_requested_number_neurons, get_requested_number_neurons, 1.0, mt);

        for (auto j = 0; j < 3; j++) {
            model->update_electrical_activity(random_graph, disable_flags);

            const auto& current_x = model->get_x();
            const auto& current_fired = model->get_fired();

            for (unsigned int id = 0; id < get_requested_number_neurons; id++) {
                ASSERT_EQ(current_x[id], model_x[id]);
                ASSERT_EQ(current_fired[id], model_fired[id]);

                ASSERT_EQ(model->get_I_syn(id), model_I_sync[id]);
                ASSERT_EQ(model->get_secondary_variable(id), model_secondary[id]);

                model_I_sync[id] = model->get_I_syn(id);
                model_secondary[id] = model->get_secondary_variable(id);
            }

            model_x = current_x;
            model_fired = current_fired;
        }
    }
}

TEST_F(NeuronModelsTest, testNeuronModelsUpdateActivityDisabledIzhikevich) {
    using namespace models;

    for (auto i = 0; i < iterations; i++) {
        std::uniform_real_distribution<double> urd_desired_k(NeuronModel::min_k, NeuronModel::max_k);
        std::uniform_real_distribution<double> urd_desired_tau_C(NeuronModel::min_tau_C, NeuronModel::max_tau_C);
        std::uniform_real_distribution<double> urd_desired_beta(NeuronModel::min_beta, NeuronModel::max_beta);
        std::uniform_int_distribution<unsigned int> uid_desired_h(NeuronModel::min_h, NeuronModel::max_h);
        std::uniform_real_distribution<double> urd_desired_base_background_activity(NeuronModel::min_base_background_activity, NeuronModel::max_base_background_activity);
        std::uniform_real_distribution<double> urd_desired_background_activity_mean(NeuronModel::min_background_activity_mean, NeuronModel::max_background_activity_mean);
        std::uniform_real_distribution<double> urd_desired_background_activity_stddev(NeuronModel::min_background_activity_stddev, NeuronModel::max_background_activity_stddev);

        std::uniform_real_distribution<double> urd_desired_a(IzhikevichModel::min_a, IzhikevichModel::max_a);
        std::uniform_real_distribution<double> urd_desired_b(IzhikevichModel::min_b, IzhikevichModel::max_b);
        std::uniform_real_distribution<double> urd_desired_c(IzhikevichModel::min_c, IzhikevichModel::max_c);
        std::uniform_real_distribution<double> urd_desired_d(IzhikevichModel::min_d, IzhikevichModel::max_d);
        std::uniform_real_distribution<double> urd_desired_V_spike(IzhikevichModel::min_V_spike, IzhikevichModel::max_V_spike);
        std::uniform_real_distribution<double> urd_desired_k1(IzhikevichModel::min_k1, IzhikevichModel::max_k1);
        std::uniform_real_distribution<double> urd_desired_k2(IzhikevichModel::min_k2, IzhikevichModel::max_k2);
        std::uniform_real_distribution<double> urd_desired_k3(IzhikevichModel::min_k3, IzhikevichModel::max_k3);

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

        const auto get_requested_number_neurons = get_random_number_neurons();

        const auto empty_graph = generate_random_network_graph(get_requested_number_neurons, 0, 1.0, mt);

        model->init(get_requested_number_neurons);

        std::vector<char> disable_flags(get_requested_number_neurons);
        std::vector<double> model_x = model->get_x();
        std::vector<double> model_secondary(get_requested_number_neurons);
        std::vector<char> model_fired = model->get_fired();
        std::vector<double> model_I_sync(get_requested_number_neurons);

        for (unsigned int id = 0; id < get_requested_number_neurons; id++) {
            model_secondary[id] = model->get_secondary_variable(id);
            model_I_sync[id] = model->get_I_syn(id);
        }

        for (auto j = 0; j < 3; j++) {
            model->update_electrical_activity(empty_graph, disable_flags);

            const auto& current_x = model->get_x();
            const auto& current_fired = model->get_fired();

            for (unsigned int id = 0; id < get_requested_number_neurons; id++) {
                ASSERT_EQ(current_x[id], model_x[id]);
                ASSERT_EQ(current_fired[id], model_fired[id]);

                ASSERT_EQ(model->get_I_syn(id), model_I_sync[id]);
                ASSERT_EQ(model->get_secondary_variable(id), model_secondary[id]);

                model_I_sync[id] = model->get_I_syn(id);
                model_secondary[id] = model->get_secondary_variable(id);
            }

            model_x = current_x;
            model_fired = current_fired;
        }

        const auto random_graph = generate_random_network_graph(get_requested_number_neurons, get_requested_number_neurons, 1.0, mt);

        for (auto j = 0; j < 3; j++) {
            model->update_electrical_activity(random_graph, disable_flags);

            const auto& current_x = model->get_x();
            const auto& current_fired = model->get_fired();

            for (unsigned int id = 0; id < get_requested_number_neurons; id++) {
                ASSERT_EQ(current_x[id], model_x[id]);
                ASSERT_EQ(current_fired[id], model_fired[id]);

                ASSERT_EQ(model->get_I_syn(id), model_I_sync[id]);
                ASSERT_EQ(model->get_secondary_variable(id), model_secondary[id]);

                model_I_sync[id] = model->get_I_syn(id);
                model_secondary[id] = model->get_secondary_variable(id);
            }

            model_x = current_x;
            model_fired = current_fired;
        }
    }
}

TEST_F(NeuronModelsTest, testNeuronModelsUpdateActivityDisabledFitzHughNagumo) {
    using namespace models;

    for (auto i = 0; i < iterations; i++) {
        std::uniform_real_distribution<double> urd_desired_k(NeuronModel::min_k, NeuronModel::max_k);
        std::uniform_real_distribution<double> urd_desired_tau_C(NeuronModel::min_tau_C, NeuronModel::max_tau_C);
        std::uniform_real_distribution<double> urd_desired_beta(NeuronModel::min_beta, NeuronModel::max_beta);
        std::uniform_int_distribution<unsigned int> uid_desired_h(NeuronModel::min_h, NeuronModel::max_h);
        std::uniform_real_distribution<double> urd_desired_base_background_activity(NeuronModel::min_base_background_activity, NeuronModel::max_base_background_activity);
        std::uniform_real_distribution<double> urd_desired_background_activity_mean(NeuronModel::min_background_activity_mean, NeuronModel::max_background_activity_mean);
        std::uniform_real_distribution<double> urd_desired_background_activity_stddev(NeuronModel::min_background_activity_stddev, NeuronModel::max_background_activity_stddev);

        std::uniform_real_distribution<double> urd_desired_a(FitzHughNagumoModel::min_a, FitzHughNagumoModel::max_a);
        std::uniform_real_distribution<double> urd_desired_b(FitzHughNagumoModel::min_b, FitzHughNagumoModel::max_b);
        std::uniform_real_distribution<double> urd_desired_phi(FitzHughNagumoModel::min_phi, FitzHughNagumoModel::max_phi);

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

        const auto get_requested_number_neurons = get_random_number_neurons();

        const auto empty_graph = generate_random_network_graph(get_requested_number_neurons, 0, 1.0, mt);

        model->init(get_requested_number_neurons);

        std::vector<char> disable_flags(get_requested_number_neurons);
        std::vector<double> model_x = model->get_x();
        std::vector<double> model_secondary(get_requested_number_neurons);
        std::vector<char> model_fired = model->get_fired();
        std::vector<double> model_I_sync(get_requested_number_neurons);

        for (unsigned int id = 0; id < get_requested_number_neurons; id++) {
            model_secondary[id] = model->get_secondary_variable(id);
            model_I_sync[id] = model->get_I_syn(id);
        }

        for (auto j = 0; j < 3; j++) {
            model->update_electrical_activity(empty_graph, disable_flags);

            const auto& current_x = model->get_x();
            const auto& current_fired = model->get_fired();

            for (unsigned int id = 0; id < get_requested_number_neurons; id++) {
                ASSERT_EQ(current_x[id], model_x[id]);
                ASSERT_EQ(current_fired[id], model_fired[id]);

                ASSERT_EQ(model->get_I_syn(id), model_I_sync[id]);
                ASSERT_EQ(model->get_secondary_variable(id), model_secondary[id]);

                model_I_sync[id] = model->get_I_syn(id);
                model_secondary[id] = model->get_secondary_variable(id);
            }

            model_x = current_x;
            model_fired = current_fired;
        }

        const auto random_graph = generate_random_network_graph(get_requested_number_neurons, get_requested_number_neurons, 1.0, mt);

        for (auto j = 0; j < 3; j++) {
            model->update_electrical_activity(random_graph, disable_flags);

            const auto& current_x = model->get_x();
            const auto& current_fired = model->get_fired();

            for (unsigned int id = 0; id < get_requested_number_neurons; id++) {
                ASSERT_EQ(current_x[id], model_x[id]);
                ASSERT_EQ(current_fired[id], model_fired[id]);

                ASSERT_EQ(model->get_I_syn(id), model_I_sync[id]);
                ASSERT_EQ(model->get_secondary_variable(id), model_secondary[id]);

                model_I_sync[id] = model->get_I_syn(id);
                model_secondary[id] = model->get_secondary_variable(id);
            }

            model_x = current_x;
            model_fired = current_fired;
        }
    }
}

TEST_F(NeuronModelsTest, testNeuronModelsUpdateActivityDisabledAEIF) {
    using namespace models;

    for (auto i = 0; i < iterations; i++) {
        std::uniform_real_distribution<double> urd_desired_k(NeuronModel::min_k, NeuronModel::max_k);
        std::uniform_real_distribution<double> urd_desired_tau_C(NeuronModel::min_tau_C, NeuronModel::max_tau_C);
        std::uniform_real_distribution<double> urd_desired_beta(NeuronModel::min_beta, NeuronModel::max_beta);
        std::uniform_int_distribution<unsigned int> uid_desired_h(NeuronModel::min_h, NeuronModel::max_h);
        std::uniform_real_distribution<double> urd_desired_base_background_activity(NeuronModel::min_base_background_activity, NeuronModel::max_base_background_activity);
        std::uniform_real_distribution<double> urd_desired_background_activity_mean(NeuronModel::min_background_activity_mean, NeuronModel::max_background_activity_mean);
        std::uniform_real_distribution<double> urd_desired_background_activity_stddev(NeuronModel::min_background_activity_stddev, NeuronModel::max_background_activity_stddev);

        std::uniform_real_distribution<double> urd_desired_C(AEIFModel::min_C, AEIFModel::max_C);
        std::uniform_real_distribution<double> urd_desired_g_L(AEIFModel::min_g_L, AEIFModel::max_g_L);
        std::uniform_real_distribution<double> urd_desired_E_L(AEIFModel::min_E_L, AEIFModel::max_E_L);
        std::uniform_real_distribution<double> urd_desired_V_T(AEIFModel::min_V_T, AEIFModel::max_V_T);
        std::uniform_real_distribution<double> urd_desired_d_T(AEIFModel::min_d_T, AEIFModel::max_d_T);
        std::uniform_real_distribution<double> urd_desired_tau_w(AEIFModel::min_tau_w, AEIFModel::max_tau_w);
        std::uniform_real_distribution<double> urd_desired_a(AEIFModel::min_a, AEIFModel::max_a);
        std::uniform_real_distribution<double> urd_desired_b(AEIFModel::min_b, AEIFModel::max_b);
        std::uniform_real_distribution<double> urd_desired_V_spike(AEIFModel::min_V_spike, AEIFModel::max_V_spike);

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
        const auto expected_V_spike = urd_desired_V_spike(mt);

        auto model = std::make_unique<AEIFModel>(expected_k, expected_tau_C, expected_beta, expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
            expected_C, expected_g_L, expected_E_L, expected_V_T, expected_d_T, expected_tau_w, expected_a, expected_b, expected_V_spike);

        const auto get_requested_number_neurons = get_random_number_neurons();

        const auto empty_graph = generate_random_network_graph(get_requested_number_neurons, 0, 1.0, mt);

        model->init(get_requested_number_neurons);

        std::vector<char> disable_flags(get_requested_number_neurons);
        std::vector<double> model_x = model->get_x();
        std::vector<double> model_secondary(get_requested_number_neurons);
        std::vector<char> model_fired = model->get_fired();
        std::vector<double> model_I_sync(get_requested_number_neurons);

        for (unsigned int id = 0; id < get_requested_number_neurons; id++) {
            model_secondary[id] = model->get_secondary_variable(id);
            model_I_sync[id] = model->get_I_syn(id);
        }

        for (auto j = 0; j < 3; j++) {
            model->update_electrical_activity(empty_graph, disable_flags);

            const auto& current_x = model->get_x();
            const auto& current_fired = model->get_fired();

            for (unsigned int id = 0; id < get_requested_number_neurons; id++) {
                ASSERT_EQ(current_x[id], model_x[id]);
                ASSERT_EQ(current_fired[id], model_fired[id]);

                ASSERT_EQ(model->get_I_syn(id), model_I_sync[id]);
                ASSERT_EQ(model->get_secondary_variable(id), model_secondary[id]);

                model_I_sync[id] = model->get_I_syn(id);
                model_secondary[id] = model->get_secondary_variable(id);
            }

            model_x = current_x;
            model_fired = current_fired;
        }

        const auto random_graph = generate_random_network_graph(get_requested_number_neurons, get_requested_number_neurons, 1.0, mt);

        for (auto j = 0; j < 3; j++) {
            model->update_electrical_activity(random_graph, disable_flags);

            const auto& current_x = model->get_x();
            const auto& current_fired = model->get_fired();

            for (unsigned int id = 0; id < get_requested_number_neurons; id++) {
                ASSERT_EQ(current_x[id], model_x[id]);
                ASSERT_EQ(current_fired[id], model_fired[id]);

                ASSERT_EQ(model->get_I_syn(id), model_I_sync[id]);
                ASSERT_EQ(model->get_secondary_variable(id), model_secondary[id]);

                model_I_sync[id] = model->get_I_syn(id);
                model_secondary[id] = model->get_secondary_variable(id);
            }

            model_x = current_x;
            model_fired = current_fired;
        }
    }
}

TEST_F(NeuronModelsTest, testNeuronModelsUpdateActivityEnabledNoBackgroundPoisson) {
    using namespace models;

    for (auto i = 0; i < iterations; i++) {
        std::uniform_real_distribution<double> urd_desired_k(NeuronModel::min_k, NeuronModel::max_k);
        std::uniform_real_distribution<double> urd_desired_tau_C(NeuronModel::min_tau_C, NeuronModel::max_tau_C);
        std::uniform_real_distribution<double> urd_desired_beta(NeuronModel::min_beta, NeuronModel::max_beta);
        std::uniform_int_distribution<unsigned int> uid_desired_h(NeuronModel::min_h, NeuronModel::max_h);
        std::uniform_real_distribution<double> urd_desired_base_background_activity(NeuronModel::min_base_background_activity, NeuronModel::max_base_background_activity);
        std::uniform_real_distribution<double> urd_desired_background_activity_mean(NeuronModel::min_background_activity_mean, NeuronModel::max_background_activity_mean);

        std::uniform_real_distribution<double> urd_desired_x0(PoissonModel::min_x_0, PoissonModel::max_x_0);
        std::uniform_int_distribution<unsigned int> urd_desired_refrac(PoissonModel::min_refrac_time, PoissonModel::max_refrac_time);
        std::uniform_real_distribution<double> urd_desired_tau_x(PoissonModel::min_tau_x, PoissonModel::max_tau_x);

        const auto expected_k = urd_desired_k(mt);
        const auto expected_tau_C = urd_desired_tau_C(mt);
        const auto expected_beta = urd_desired_beta(mt);
        const auto expected_h = uid_desired_h(mt);
        const auto expected_base_background_activity = urd_desired_base_background_activity(mt);
        const auto expected_background_activity_mean = urd_desired_background_activity_mean(mt);
        const auto expected_background_activity_stddev = 0.0;

        const auto expected_x0 = urd_desired_x0(mt);
        const auto expected_refrac = urd_desired_refrac(mt);
        const auto expected_tau_x = urd_desired_tau_x(mt);

        auto model = std::make_unique<PoissonModel>(expected_k, expected_tau_C, expected_beta, expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
            expected_x0, expected_tau_x, expected_refrac);

        const auto get_requested_number_neurons = get_random_number_neurons();

        const auto empty_graph = generate_random_network_graph(get_requested_number_neurons, 0, 1.0, mt);

        model->init(get_requested_number_neurons);

        std::vector<char> disable_flags(get_requested_number_neurons, 1);
        std::vector<double> model_x = model->get_x();
        std::vector<double> model_secondary(get_requested_number_neurons);
        std::vector<char> model_fired = model->get_fired();
        std::vector<double> model_I_syn = model->get_I_syn();

        for (unsigned int id = 0; id < get_requested_number_neurons; id++) {
            model_secondary[id] = model->get_secondary_variable(id);
        }

        for (auto j = 0; j < 3; j++) {
            model->update_electrical_activity(empty_graph, disable_flags);

            const auto& current_x = model->get_x();
            const auto& current_fired = model->get_fired();
            const auto& current_I_syn = model->get_I_syn();

            for (unsigned int id = 0; id < get_requested_number_neurons; id++) {
                ASSERT_NE(current_x[id], model_x[id]) << i << ' ' << j << ' ' << id;

                if (j == 0) {
                    ASSERT_EQ(0.0, model_I_syn[id]) << id << ' ' << j << ' ' << i;
                } else {
                    ASSERT_EQ(expected_base_background_activity, model_I_syn[id]) << id << ' ' << j << ' ' << i;
                }

                const auto current_refrac = model->get_secondary_variable(id);
                if (model->get_fired(id)) {
                    ASSERT_EQ(current_refrac, expected_refrac) << i << ' ' << j << ' ' << id;
                } else if (model_secondary[id] == 0) {
                    ASSERT_EQ(current_refrac, 0) << i << ' ' << j << ' ' << id;
                } else {
                    ASSERT_EQ(current_refrac, model_secondary[id] - 1) << i << ' ' << j << ' ' << id;
                }

                model_secondary[id] = model->get_secondary_variable(id);
            }

            model_x = current_x;
            model_fired = current_fired;
            model_I_syn = current_I_syn;
        }

        const auto random_graph = generate_random_network_graph(get_requested_number_neurons, get_requested_number_neurons, 1.0, mt);

        for (auto j = 0; j < 3; j++) {
            model->update_electrical_activity(random_graph, disable_flags);

            const auto& current_x = model->get_x();
            const auto& current_fired = model->get_fired();
            const auto& current_I_syn = model->get_I_syn();

            for (unsigned int id = 0; id < get_requested_number_neurons; id++) {
                ASSERT_NE(current_x[id], model_x[id]);

                const auto current_refrac = model->get_secondary_variable(id);
                if (model->get_fired(id)) {
                    ASSERT_EQ(current_refrac, expected_refrac);
                } else if (model_secondary[id] == 0) {
                    ASSERT_EQ(current_refrac, 0);
                } else {
                    ASSERT_EQ(current_refrac, model_secondary[id] - 1);
                }

                model_secondary[id] = model->get_secondary_variable(id);
            }

            model_x = current_x;
            model_fired = current_fired;
            model_I_syn = current_I_syn;
        }
    }
}

TEST_F(NeuronModelsTest, testNeuronModelsUpdateActivityEnabledNoBackgroundIzhikevich) {
    using namespace models;

    for (auto i = 0; i < iterations; i++) {
        std::uniform_real_distribution<double> urd_desired_k(NeuronModel::min_k, NeuronModel::max_k);
        std::uniform_real_distribution<double> urd_desired_tau_C(NeuronModel::min_tau_C, NeuronModel::max_tau_C);
        std::uniform_real_distribution<double> urd_desired_beta(NeuronModel::min_beta, NeuronModel::max_beta);
        std::uniform_int_distribution<unsigned int> uid_desired_h(NeuronModel::min_h, NeuronModel::max_h);
        std::uniform_real_distribution<double> urd_desired_base_background_activity(NeuronModel::min_base_background_activity, NeuronModel::max_base_background_activity);
        std::uniform_real_distribution<double> urd_desired_background_activity_mean(NeuronModel::min_background_activity_mean, NeuronModel::max_background_activity_mean);

        std::uniform_real_distribution<double> urd_desired_a(IzhikevichModel::min_a, IzhikevichModel::max_a);
        std::uniform_real_distribution<double> urd_desired_b(IzhikevichModel::min_b, IzhikevichModel::max_b);
        std::uniform_real_distribution<double> urd_desired_c(IzhikevichModel::min_c, IzhikevichModel::max_c);
        std::uniform_real_distribution<double> urd_desired_d(IzhikevichModel::min_d, IzhikevichModel::max_d);
        std::uniform_real_distribution<double> urd_desired_V_spike(IzhikevichModel::min_V_spike, IzhikevichModel::max_V_spike);
        std::uniform_real_distribution<double> urd_desired_k1(IzhikevichModel::min_k1, IzhikevichModel::max_k1);
        std::uniform_real_distribution<double> urd_desired_k2(IzhikevichModel::min_k2, IzhikevichModel::max_k2);
        std::uniform_real_distribution<double> urd_desired_k3(IzhikevichModel::min_k3, IzhikevichModel::max_k3);

        const auto expected_k = urd_desired_k(mt);
        const auto expected_tau_C = urd_desired_tau_C(mt);
        const auto expected_beta = urd_desired_beta(mt);
        const auto expected_h = uid_desired_h(mt);
        const auto expected_base_background_activity = urd_desired_base_background_activity(mt);
        const auto expected_background_activity_mean = urd_desired_background_activity_mean(mt);
        const auto expected_background_activity_stddev = 0.0;

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

        const auto get_requested_number_neurons = get_random_number_neurons();

        const auto empty_graph = generate_random_network_graph(get_requested_number_neurons, 0, 1.0, mt);

        model->init(get_requested_number_neurons);

        std::vector<char> disable_flags(get_requested_number_neurons, 1);
        std::vector<double> model_x = model->get_x();
        std::vector<double> model_secondary(get_requested_number_neurons);
        std::vector<char> model_fired = model->get_fired();
        std::vector<double> model_I_syn = model->get_I_syn();

        for (unsigned int id = 0; id < get_requested_number_neurons; id++) {
            model_secondary[id] = model->get_secondary_variable(id);
        }

        for (auto j = 0; j < 3; j++) {
            model->update_electrical_activity(empty_graph, disable_flags);

            const auto& current_x = model->get_x();
            const auto& current_fired = model->get_fired();
            const auto& current_I_syn = model->get_I_syn();

            for (unsigned int id = 0; id < get_requested_number_neurons; id++) {
                if (current_fired[id]) {
                    ASSERT_EQ(current_x[id], expected_c) << i << ' ' << j << ' ' << id;
                }

                if (j == 0) {
                    ASSERT_EQ(0.0, model_I_syn[id]) << id << ' ' << j << ' ' << i;
                } else {
                    ASSERT_EQ(expected_base_background_activity, model_I_syn[id]) << id << ' ' << j << ' ' << i;
                }
                ASSERT_NE(model->get_secondary_variable(id), model_secondary[id]) << i << ' ' << j << ' ' << id;

                model_secondary[id] = model->get_secondary_variable(id);
            }

            model_x = current_x;
            model_fired = current_fired;
            model_I_syn = current_I_syn;
        }

        const auto random_graph = generate_random_network_graph(get_requested_number_neurons, get_requested_number_neurons, 1.0, mt);

        for (auto j = 0; j < 3; j++) {
            model->update_electrical_activity(random_graph, disable_flags);

            const auto& current_x = model->get_x();
            const auto& current_fired = model->get_fired();
            const auto& current_I_syn = model->get_I_syn();

            for (unsigned int id = 0; id < get_requested_number_neurons; id++) {
                if (current_fired[id]) {
                    ASSERT_EQ(current_x[id], expected_c) << i << ' ' << j << ' ' << id;
                }

                ASSERT_NE(model->get_secondary_variable(id), model_secondary[id]);

                model_secondary[id] = model->get_secondary_variable(id);
            }

            model_x = current_x;
            model_fired = current_fired;
            model_I_syn = current_I_syn;
        }
    }
}

TEST_F(NeuronModelsTest, testNeuronModelsUpdateActivityEnabledNoBackgroundFitzHughNagumo) {
    using namespace models;

    for (auto i = 0; i < iterations; i++) {
        std::uniform_real_distribution<double> urd_desired_k(NeuronModel::min_k, NeuronModel::max_k);
        std::uniform_real_distribution<double> urd_desired_tau_C(NeuronModel::min_tau_C, NeuronModel::max_tau_C);
        std::uniform_real_distribution<double> urd_desired_beta(NeuronModel::min_beta, NeuronModel::max_beta);
        std::uniform_int_distribution<unsigned int> uid_desired_h(NeuronModel::min_h, NeuronModel::max_h);
        std::uniform_real_distribution<double> urd_desired_base_background_activity(NeuronModel::min_base_background_activity, NeuronModel::max_base_background_activity);
        std::uniform_real_distribution<double> urd_desired_background_activity_mean(NeuronModel::min_background_activity_mean, NeuronModel::max_background_activity_mean);

        std::uniform_real_distribution<double> urd_desired_a(FitzHughNagumoModel::min_a, FitzHughNagumoModel::max_a);
        std::uniform_real_distribution<double> urd_desired_b(FitzHughNagumoModel::min_b, FitzHughNagumoModel::max_b);
        std::uniform_real_distribution<double> urd_desired_phi(FitzHughNagumoModel::min_phi, FitzHughNagumoModel::max_phi);

        const auto expected_k = urd_desired_k(mt);
        const auto expected_tau_C = urd_desired_tau_C(mt);
        const auto expected_beta = urd_desired_beta(mt);
        const auto expected_h = uid_desired_h(mt);
        const auto expected_base_background_activity = urd_desired_base_background_activity(mt);
        const auto expected_background_activity_mean = urd_desired_background_activity_mean(mt);
        const auto expected_background_activity_stddev = 0.0;

        const auto expected_a = urd_desired_a(mt);
        const auto expected_b = urd_desired_b(mt);
        const auto expected_phi = urd_desired_phi(mt);

        auto model = std::make_unique<FitzHughNagumoModel>(expected_k, expected_tau_C, expected_beta, expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
            expected_a, expected_b, expected_phi);

        const auto get_requested_number_neurons = get_random_number_neurons();

        const auto empty_graph = generate_random_network_graph(get_requested_number_neurons, 0, 1.0, mt);

        model->init(get_requested_number_neurons);

        std::vector<char> disable_flags(get_requested_number_neurons, 1);
        std::vector<double> model_x = model->get_x();
        std::vector<double> model_secondary(get_requested_number_neurons);
        std::vector<double> model_I_syn = model->get_I_syn();

        for (unsigned int id = 0; id < get_requested_number_neurons; id++) {
            model_secondary[id] = model->get_secondary_variable(id);
        }

        for (auto j = 0; j < 3; j++) {
            model->update_electrical_activity(empty_graph, disable_flags);

            const auto& current_x = model->get_x();
            const auto& current_I_syn = model->get_I_syn();

            for (unsigned int id = 0; id < get_requested_number_neurons; id++) {
                ASSERT_NE(current_x[id], model_x[id]);

                if (j == 0) {
                    ASSERT_EQ(0.0, model_I_syn[id]) << id << ' ' << j << ' ' << i;
                } else {
                    ASSERT_EQ(expected_base_background_activity, model_I_syn[id]) << id << ' ' << j << ' ' << i;
                }

                ASSERT_NE(model->get_secondary_variable(id), model_secondary[id]);

                model_secondary[id] = model->get_secondary_variable(id);
            }

            model_x = current_x;
            model_I_syn = current_I_syn;
        }

        const auto random_graph = generate_random_network_graph(get_requested_number_neurons, get_requested_number_neurons, 1.0, mt);

        for (auto j = 0; j < 3; j++) {
            model->update_electrical_activity(random_graph, disable_flags);

            const auto& current_x = model->get_x();
            const auto& current_fired = model->get_fired();
            const auto& current_I_syn = model->get_I_syn();

            for (unsigned int id = 0; id < get_requested_number_neurons; id++) {
                ASSERT_NE(current_x[id], model_x[id]);

                ASSERT_NE(model->get_secondary_variable(id), model_secondary[id]);

                model_secondary[id] = model->get_secondary_variable(id);
            }

            model_x = current_x;
            model_I_syn = current_I_syn;
        }
    }
}

TEST_F(NeuronModelsTest, testNeuronModelsUpdateActivityEnabledNoBackgroundAEIF) {
    using namespace models;

    for (auto i = 0; i < iterations; i++) {
        std::uniform_real_distribution<double> urd_desired_k(NeuronModel::min_k, NeuronModel::max_k);
        std::uniform_real_distribution<double> urd_desired_tau_C(NeuronModel::min_tau_C, NeuronModel::max_tau_C);
        std::uniform_real_distribution<double> urd_desired_beta(NeuronModel::min_beta, NeuronModel::max_beta);
        std::uniform_int_distribution<unsigned int> uid_desired_h(NeuronModel::min_h, NeuronModel::max_h);
        std::uniform_real_distribution<double> urd_desired_base_background_activity(NeuronModel::min_base_background_activity, NeuronModel::max_base_background_activity);
        std::uniform_real_distribution<double> urd_desired_background_activity_mean(NeuronModel::min_background_activity_mean, NeuronModel::max_background_activity_mean);

        std::uniform_real_distribution<double> urd_desired_C(AEIFModel::min_C, AEIFModel::max_C);
        std::uniform_real_distribution<double> urd_desired_g_L(AEIFModel::min_g_L, AEIFModel::max_g_L);
        std::uniform_real_distribution<double> urd_desired_E_L(AEIFModel::min_E_L, AEIFModel::max_E_L);
        std::uniform_real_distribution<double> urd_desired_V_T(AEIFModel::min_V_T, AEIFModel::max_V_T);
        std::uniform_real_distribution<double> urd_desired_d_T(AEIFModel::min_d_T, AEIFModel::max_d_T);
        std::uniform_real_distribution<double> urd_desired_tau_w(AEIFModel::min_tau_w, AEIFModel::max_tau_w);
        std::uniform_real_distribution<double> urd_desired_a(AEIFModel::min_a, AEIFModel::max_a);
        std::uniform_real_distribution<double> urd_desired_b(AEIFModel::min_b, AEIFModel::max_b);
        std::uniform_real_distribution<double> urd_desired_V_spike(AEIFModel::min_V_spike, AEIFModel::max_V_spike);

        const auto expected_k = urd_desired_k(mt);
        const auto expected_tau_C = urd_desired_tau_C(mt);
        const auto expected_beta = urd_desired_beta(mt);
        const auto expected_h = uid_desired_h(mt);
        const auto expected_base_background_activity = urd_desired_base_background_activity(mt);
        const auto expected_background_activity_mean = urd_desired_background_activity_mean(mt);
        const auto expected_background_activity_stddev = 0.0;

        const auto expected_C = urd_desired_C(mt);
        const auto expected_g_L = urd_desired_g_L(mt);
        const auto expected_E_L = urd_desired_E_L(mt);
        const auto expected_V_T = urd_desired_V_T(mt);
        const auto expected_d_T = urd_desired_d_T(mt);
        const auto expected_tau_w = urd_desired_tau_w(mt);
        const auto expected_a = urd_desired_a(mt);
        const auto expected_b = urd_desired_b(mt);
        const auto expected_V_spike = urd_desired_V_spike(mt);

        auto model = std::make_unique<AEIFModel>(expected_k, expected_tau_C, expected_beta, expected_h, expected_base_background_activity, expected_background_activity_mean, expected_background_activity_stddev,
            expected_C, expected_g_L, expected_E_L, expected_V_T, expected_d_T, expected_tau_w, expected_a, expected_b, expected_V_spike);

        const auto get_requested_number_neurons = get_random_number_neurons();

        const auto empty_graph = generate_random_network_graph(get_requested_number_neurons, 0, 1.0, mt);

        model->init(get_requested_number_neurons);

        std::vector<char> disable_flags(get_requested_number_neurons, 1);
        std::vector<double> model_x = model->get_x();
        std::vector<double> model_secondary(get_requested_number_neurons);
        std::vector<char> model_fired = model->get_fired();
        std::vector<double> model_I_syn = model->get_I_syn();

        for (unsigned int id = 0; id < get_requested_number_neurons; id++) {
            model_secondary[id] = model->get_secondary_variable(id);
        }

        for (auto j = 0; j < 3; j++) {
            model->update_electrical_activity(empty_graph, disable_flags);

            const auto& current_x = model->get_x();
            const auto& current_fired = model->get_fired();
            const auto& current_I_syn = model->get_I_syn();

            for (unsigned int id = 0; id < get_requested_number_neurons; id++) {
                if (current_fired[id]) {
                    EXPECT_EQ(current_x[id], expected_E_L) << id << ' ' << j << ' ' << i;
                }

                if (j == 0) {
                    ASSERT_EQ(0.0, model_I_syn[id]) << id << ' ' << j << ' ' << i;
                } else {
                    ASSERT_EQ(expected_base_background_activity, model_I_syn[id]) << id << ' ' << j << ' ' << i;
                }

                model_secondary[id] = model->get_secondary_variable(id);
            }

            model_x = current_x;
            model_fired = current_fired;
            model_I_syn = current_I_syn;
        }

        const auto random_graph = generate_random_network_graph(get_requested_number_neurons, get_requested_number_neurons, 1.0, mt);

        for (auto j = 0; j < 3; j++) {
            model->update_electrical_activity(random_graph, disable_flags);

            const auto& current_x = model->get_x();
            const auto& current_fired = model->get_fired();
            const auto& current_I_syn = model->get_I_syn();

            for (unsigned int id = 0; id < get_requested_number_neurons; id++) {
                if (current_fired[id]) {
                    EXPECT_EQ(current_x[id], expected_E_L) << id;
                }

                model_secondary[id] = model->get_secondary_variable(id);
            }

            model_x = current_x;
            model_fired = current_fired;
            model_I_syn = current_I_syn;
        }
    }
}
