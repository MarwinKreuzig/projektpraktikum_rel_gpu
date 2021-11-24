#include "../googletest/include/gtest/gtest.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "RelearnTest.hpp"

#define protected public
#include "../source/sim/NeuronToSubdomainAssignment.h"
#include "../source/structure/Partition.h"
#include "../source/sim/SubdomainFromNeuronDensity.h"
#include "../source/sim/SubdomainFromFile.h"
#include "../source/util/RelearnException.h"

void sort_indices(const std::vector<Vec3d>& vectors, std::vector<int>& sequence) {
    sequence = std::vector<int>(vectors.size());
    std::iota(sequence.begin(), sequence.end(), 0);

    std::stable_sort(sequence.begin(), sequence.end(),
        [&vectors](int i1, int i2) { return vectors[i1] < vectors[i2]; });
}

void check_types_fraction(std::vector<SignalType>& types, double& frac_ex, unsigned long long total_subdomains, const size_t& num_neurons) {
    size_t neurons_ex = 0;
    size_t neurons_in = 0;

    for (auto& type : types) {
        if (type == SignalType::EXCITATORY) {
            neurons_ex++;
        } else if (type == SignalType::INHIBITORY) {
            neurons_in++;
        }
    }

    auto frac_ex_ = (static_cast<double>(neurons_ex) / static_cast<double>(neurons_ex + neurons_in));
    ASSERT_NEAR(frac_ex, frac_ex_, static_cast<double>(total_subdomains) / static_cast<double>(num_neurons));
}

void check_positions(std::vector<NeuronToSubdomainAssignment::position_type>& pos, double um_per_neuron, size_t& expected_neurons_per_dimension, bool* flags) {
    std::vector<Vec3<size_t>> pos_fixed;
    for (auto& p : pos) {
        pos_fixed.emplace_back((Vec3<size_t>)(p * (1 / um_per_neuron)));
    }

    for (auto& p : pos_fixed) {
        auto x = p.get_x();
        auto y = p.get_y();
        auto z = p.get_z();

        ASSERT_LE(x, expected_neurons_per_dimension);
        ASSERT_LE(y, expected_neurons_per_dimension);
        ASSERT_LE(z, expected_neurons_per_dimension);

        auto idx = x * expected_neurons_per_dimension * expected_neurons_per_dimension + y * expected_neurons_per_dimension + z;
        auto& val = flags[idx];
        ASSERT_FALSE(val);
        val = true;
    }
}

void generate_neuron_positions(std::vector<Vec3d>& positions,
    std::vector<std::string>& area_names, std::vector<SignalType>& types, std::mt19937& mt) {

    std::uniform_int_distribution<size_t> uid(1, 1000);
    std::uniform_real_distribution<double> urd(0.0, 1.0);

    auto num_neurons = uid(mt);
    auto frac_ex = urd(mt);
    auto um_per_neuron = urd(mt) * 100;

    auto part = std::make_shared<Partition>(1, 0);
    SubdomainFromNeuronDensity sfnd{ num_neurons, frac_ex, um_per_neuron, part };

    auto num_neurons_ = sfnd.desired_num_neurons();

    ASSERT_EQ(num_neurons, num_neurons_);

    auto box_length = sfnd.get_simulation_box_length().get_maximum();

    sfnd.fill_subdomain(0, 1, Vec3d{ 0 }, Vec3d{ box_length });

    positions = sfnd.neuron_positions(0, 1, Vec3d{ 0 }, Vec3d{ box_length });
    area_names = sfnd.neuron_area_names(0, 1, Vec3d{ 0 }, Vec3d{ box_length });
    types = sfnd.neuron_types(0, 1, Vec3d{ 0 }, Vec3d{ box_length });

    ASSERT_EQ(num_neurons, positions.size());
    ASSERT_EQ(num_neurons, area_names.size());
    ASSERT_EQ(num_neurons, types.size());

    sfnd.write_neurons_to_file("neurons.tmp");
}

bool operator<(const NeuronToSubdomainAssignment::Node& a, const NeuronToSubdomainAssignment::Node& b) {
    return NeuronToSubdomainAssignment::Node::less()(a, b);
}

TEST_F(NeuronAssignmentTest, test_constructor) {
    std::uniform_int_distribution<size_t> uid(1, 10000);
    std::uniform_real_distribution<double> urd(0.0, 1.0);

    mt.seed(rand());

    for (auto i = 0; i < iterations; i++) {
        auto num_neurons = uid(mt);
        auto frac_ex = urd(mt);
        auto um_per_neuron = urd(mt) * 100;

        auto part = std::make_shared<Partition>(1, 0);
        SubdomainFromNeuronDensity sfnd{ num_neurons, frac_ex, um_per_neuron, part };

        auto num_neurons_ = sfnd.desired_num_neurons();
        auto frac_ex_ = sfnd.desired_ratio_neurons_exc();

        ASSERT_EQ(num_neurons, num_neurons_);

        ASSERT_NEAR(frac_ex, frac_ex_, 1.0 / num_neurons);

        ASSERT_NEAR(sfnd.get_simulation_box_length().get_x(),
            ceil(pow(static_cast<double>(num_neurons), 1 / 3.)) * um_per_neuron,
            1.0 / num_neurons);

        ASSERT_EQ(0, sfnd.placed_num_neurons());
        ASSERT_EQ(0, sfnd.placed_ratio_neurons_exc());
    }
}

TEST_F(NeuronAssignmentTest, test_lazily_fill) {
    std::uniform_int_distribution<size_t> uid(1, 10000);
    std::uniform_real_distribution<double> urd(0.0, 1.0);

    mt.seed(rand());

    for (auto i = 0; i < iterations; i++) {
        auto num_neurons = uid(mt);
        auto frac_ex = urd(mt);
        auto um_per_neuron = urd(mt) * 100;

        auto lower_bound_ex = static_cast<size_t>(floor(num_neurons * frac_ex));
        auto upper_bound_ex = static_cast<size_t>(ceil(num_neurons * frac_ex));

        auto part = std::make_shared<Partition>(1, 0);
        SubdomainFromNeuronDensity sfnd{ num_neurons, frac_ex, um_per_neuron, part };

        auto box_length = sfnd.get_simulation_box_length().get_maximum();

        ASSERT_GE(box_length, 0);

        sfnd.fill_subdomain(0, 1, Vec3d{ 0 }, Vec3d{ box_length });

        auto num_neurons_ = sfnd.desired_num_neurons();
        auto frac_ex_ = sfnd.desired_ratio_neurons_exc();

        ASSERT_EQ(num_neurons, num_neurons_);

        ASSERT_NEAR(frac_ex, frac_ex_, 1.0 / num_neurons);

        ASSERT_EQ(sfnd.desired_num_neurons(), sfnd.placed_num_neurons());
        ASSERT_NEAR(sfnd.desired_ratio_neurons_exc(), sfnd.placed_ratio_neurons_exc(), 1.0 / num_neurons);

        ASSERT_LE(sfnd.desired_ratio_neurons_exc(), sfnd.placed_ratio_neurons_exc());
    }
}

TEST_F(NeuronAssignmentTest, test_lazily_fill_multiple) {
    std::uniform_int_distribution<size_t> uid(1, 10000);
    std::uniform_int_distribution<size_t> uid_fills(1, 10);
    std::uniform_real_distribution<double> urd(0.0, 1.0);

    mt.seed(rand());

    for (auto i = 0; i < iterations; i++) {
        auto num_neurons = uid(mt);
        auto frac_ex = urd(mt);
        auto um_per_neuron = urd(mt) * 100;

        auto lower_bound_ex = static_cast<size_t>(floor(num_neurons * frac_ex));
        auto upper_bound_ex = static_cast<size_t>(ceil(num_neurons * frac_ex));

        auto part = std::make_shared<Partition>(1, 0);
        SubdomainFromNeuronDensity sfnd{ num_neurons, frac_ex, um_per_neuron, part };

        auto box_length = sfnd.get_simulation_box_length().get_maximum();

        ASSERT_GE(box_length, 0);

        auto num_fills = uid_fills(mt);

        for (auto j = 0; j < num_fills; j++) {
            if (j >= 1) {
                ASSERT_THROW(sfnd.fill_subdomain(0, 1, Vec3d{ 0 }, Vec3d{ box_length }), RelearnException);
            } else {
                sfnd.fill_subdomain(0, 1, Vec3d{ 0 }, Vec3d{ box_length });
            }
        }

        auto num_neurons_ = sfnd.desired_num_neurons();
        auto frac_ex_ = sfnd.desired_ratio_neurons_exc();

        ASSERT_EQ(num_neurons, num_neurons_);

        ASSERT_NEAR(frac_ex, frac_ex_, 1.0 / num_neurons);

        ASSERT_EQ(sfnd.desired_num_neurons(), sfnd.placed_num_neurons());
        ASSERT_NEAR(sfnd.desired_ratio_neurons_exc(), sfnd.placed_ratio_neurons_exc(), 1.0 / num_neurons);

        ASSERT_LE(sfnd.desired_ratio_neurons_exc(), sfnd.placed_ratio_neurons_exc());
    }
}

TEST_F(NeuronAssignmentTest, test_lazily_fill_positions) {
    std::uniform_int_distribution<size_t> uid(1, 10000);
    std::uniform_real_distribution<double> urd(0.0, 1.0);

    mt.seed(rand());

    for (auto i = 0; i < iterations; i++) {
        auto num_neurons = uid(mt);
        auto frac_ex = urd(mt);
        auto um_per_neuron = urd(mt) * 100;

        auto expected_neurons_per_dimension = static_cast<size_t>(ceil(pow(num_neurons, 1.0 / 3.0)));
        auto size = expected_neurons_per_dimension * expected_neurons_per_dimension * expected_neurons_per_dimension;

        std::vector<bool> flags(size, false);

        auto part = std::make_shared<Partition>(1, 0);
        SubdomainFromNeuronDensity sfnd{ num_neurons, frac_ex, um_per_neuron, part };

        auto box_length = sfnd.get_simulation_box_length().get_maximum();

        sfnd.fill_subdomain(0, 1, Vec3d{ 0 }, Vec3d{ box_length });

        std::vector<Vec3d> pos = sfnd.neuron_positions(0, 1, Vec3d{ 0 }, Vec3d{ box_length });

        ASSERT_EQ(pos.size(), num_neurons);

        std::vector<Vec3<size_t>> pos_fixed;
        for (auto& p : pos) {
            pos_fixed.emplace_back((Vec3<size_t>)(p * (1 / um_per_neuron)));
        }

        for (auto& p : pos_fixed) {
            auto x = p.get_x();
            auto y = p.get_y();
            auto z = p.get_z();

            ASSERT_LE(x, expected_neurons_per_dimension);
            ASSERT_LE(y, expected_neurons_per_dimension);
            ASSERT_LE(z, expected_neurons_per_dimension);

            auto idx = x * expected_neurons_per_dimension * expected_neurons_per_dimension + y * expected_neurons_per_dimension + z;
            ASSERT_FALSE(flags[idx]);
            flags[idx] = true;
        }

        std::vector<SignalType> types = sfnd.neuron_types(0, 1, Vec3d{ 0 }, Vec3d{ box_length });

        size_t neurons_ex = 0;
        size_t neurons_in = 0;

        for (auto& type : types) {
            if (type == SignalType::EXCITATORY) {
                neurons_ex++;
            } else if (type == SignalType::INHIBITORY) {
                neurons_in++;
            }
        }

        auto frac_ex_ = (static_cast<double>(neurons_ex) / static_cast<double>(neurons_ex + neurons_in));
        ASSERT_NEAR(frac_ex, frac_ex_, 1.0 / static_cast<double>(num_neurons));
    }
}

TEST_F(NeuronAssignmentTest, test_lazily_fill_positions_multiple_subdomains) {
    std::uniform_int_distribution<size_t> uid(1, 10000);
    std::uniform_real_distribution<double> urd(0.0, 1.0);

    mt.seed(rand());

    std::uniform_int_distribution<size_t> uid_subdomains(1, 5);

    for (auto i = 0; i < iterations; i++) {
        auto num_neurons = uid(mt);
        auto frac_ex = urd(mt);
        auto um_per_neuron = urd(mt) * 100;

        auto subdomains_x = uid_subdomains(mt);
        auto subdomains_y = uid_subdomains(mt);
        auto subdomains_z = uid_subdomains(mt);

        auto total_subdomains = subdomains_x * subdomains_y * subdomains_z;

        auto expected_neurons_per_dimension = static_cast<size_t>(ceil(pow(num_neurons, 1.0 / 3.0)));
        auto size = expected_neurons_per_dimension * expected_neurons_per_dimension * expected_neurons_per_dimension;

        std::vector<bool> flags(size, false);

        auto part = std::make_shared<Partition>(1, 0);
        SubdomainFromNeuronDensity sfnd{ num_neurons, frac_ex, um_per_neuron, part };

        auto box_length = sfnd.get_simulation_box_length().get_maximum();

        auto subdomain_length_x = box_length / subdomains_x;
        auto subdomain_length_y = box_length / subdomains_y;
        auto subdomain_length_z = box_length / subdomains_z;

        sfnd.fill_subdomain(0, 1, Vec3d{ 0 }, Vec3d{ box_length });

        std::vector<Vec3d> pos;
        for (auto x_it = 0; x_it < subdomains_x; x_it++) {
            for (auto y_it = 0; y_it < subdomains_y; y_it++) {
                for (auto z_it = 0; z_it < subdomains_z; z_it++) {
                    Vec3d subdomain_min{ x_it * subdomain_length_x, y_it * subdomain_length_y, z_it * subdomain_length_z };
                    Vec3d subdomain_max{ (x_it + 1) * subdomain_length_x, (y_it + 1) * subdomain_length_y, (z_it + 1) * subdomain_length_z };

                    auto current_idx = z_it + y_it * subdomains_z + x_it * subdomains_z * subdomains_y;

                    if (x_it == 0 && y_it == 0 && z_it == 0) {
                        auto vec = sfnd.neuron_positions(current_idx, total_subdomains, subdomain_min, subdomain_max);
                        pos.insert(pos.end(), vec.begin(), vec.end());
                    } else {
                        ASSERT_THROW(auto vec = sfnd.neuron_positions(current_idx, total_subdomains, subdomain_min, subdomain_max), RelearnException);
                    }
                }
            }
        }

        ASSERT_EQ(pos.size(), num_neurons);

        std::vector<Vec3<size_t>> pos_fixed;
        for (auto& p : pos) {
            pos_fixed.emplace_back((Vec3<size_t>)(p * (1 / um_per_neuron)));
        }

        for (auto& p : pos_fixed) {
            auto x = p.get_x();
            auto y = p.get_y();
            auto z = p.get_z();

            ASSERT_LE(x, expected_neurons_per_dimension);
            ASSERT_LE(y, expected_neurons_per_dimension);
            ASSERT_LE(z, expected_neurons_per_dimension);

            auto idx = x * expected_neurons_per_dimension * expected_neurons_per_dimension + y * expected_neurons_per_dimension + z;
            ASSERT_FALSE(flags[idx]);
            flags[idx] = true;
        }

        std::vector<SignalType> types = sfnd.neuron_types(0, 1, Vec3d{ 0 }, Vec3d{ box_length });

        size_t neurons_ex = 0;
        size_t neurons_in = 0;

        for (auto& type : types) {
            if (type == SignalType::EXCITATORY) {
                neurons_ex++;
            } else if (type == SignalType::INHIBITORY) {
                neurons_in++;
            }
        }

        auto frac_ex_ = (static_cast<double>(neurons_ex) / static_cast<double>(neurons_ex + neurons_in));
        ASSERT_NEAR(frac_ex, frac_ex_, static_cast<double>(total_subdomains) / static_cast<double>(num_neurons));
    }
}

TEST_F(NeuronAssignmentTest, test_multiple_lazily_fill_positions_multiple_subdomains) {
    std::uniform_int_distribution<size_t> uid(1, 10000);
    std::uniform_real_distribution<double> urd(0.0, 1.0);

    mt.seed(rand());

    std::uniform_int_distribution<size_t> uid_subdomains(1, 5);

    for (auto i = 0; i < iterations; i++) {
        auto num_neurons = uid(mt);
        auto frac_ex = urd(mt);
        auto um_per_neuron = urd(mt) * 100;

        auto subdomains_x = uid_subdomains(mt);
        auto subdomains_y = uid_subdomains(mt);
        auto subdomains_z = uid_subdomains(mt);

        Vec3<size_t> subdomains{ subdomains_x, subdomains_y, subdomains_z };

        auto total_subdomains = subdomains_x * subdomains_y * subdomains_z;

        auto expected_neurons_per_dimension = static_cast<size_t>(ceil(pow(num_neurons, 1.0 / 3.0)));
        auto size = expected_neurons_per_dimension * expected_neurons_per_dimension * expected_neurons_per_dimension;

        auto flags = new bool[size];
        for (auto j = 0; j < size; j++) {
            flags[j] = false;
        }

        auto part = std::make_shared<Partition>(1, 0);
        SubdomainFromNeuronDensity sfnd{ num_neurons, frac_ex, um_per_neuron, part };

        auto box_length = sfnd.get_simulation_box_length().get_maximum();

        auto subdomain_length_x = box_length / subdomains_x;
        auto subdomain_length_y = box_length / subdomains_y;
        auto subdomain_length_z = box_length / subdomains_z;

        std::vector<Vec3d> pos;
        for (size_t x_it = 0; x_it < subdomains_x; x_it++) {
            for (size_t y_it = 0; y_it < subdomains_y; y_it++) {
                for (size_t z_it = 0; z_it < subdomains_z; z_it++) {
                    auto current_idx = z_it + y_it * subdomains_z + x_it * subdomains_z * subdomains_y;

                    Vec3d subdomain_min{};
                    Vec3d subdomain_max{};

                    Vec3<size_t> subdomain_pos{ x_it, y_it, z_it };
                    std::tie(subdomain_min, subdomain_max) = sfnd.get_subdomain_boundaries(subdomain_pos, subdomains);

                    sfnd.fill_subdomain(current_idx, total_subdomains, subdomain_min, subdomain_max);
                    auto vec = sfnd.neuron_positions(current_idx, total_subdomains, subdomain_min, subdomain_max);
                    pos.insert(pos.end(), vec.begin(), vec.end());
                }
            }
        }

        auto diff_neurons = std::max(pos.size(), num_neurons) - std::min(pos.size(), num_neurons);
        ASSERT_LE(diff_neurons, total_subdomains);

        check_positions(pos, um_per_neuron, expected_neurons_per_dimension, flags);

        std::vector<SignalType> types = sfnd.neuron_types(0, 1, Vec3d{ 0 }, Vec3d{ box_length });

        check_types_fraction(types, frac_ex, total_subdomains, num_neurons);

        delete[] flags;
    }
}

TEST_F(NeuronAssignmentTest, test_saving) {
    std::uniform_int_distribution<size_t> uid(1, 1000);
    std::uniform_real_distribution<double> urd(0.0, 1.0);

    mt.seed(rand());

    for (auto i = 0; i < iterations; i++) {
        std::vector<Vec3d> positions;
        std::vector<std::string> area_names;
        std::vector<SignalType> types;

        generate_neuron_positions(positions, area_names, types, mt);

        auto num_neurons = positions.size();

        std::ifstream file("neurons.tmp", std::ios::binary | std::ios::in);

        std::vector<std::string> lines;

        std::string str;
        while (std::getline(file, str)) {
            if (str[0] == '#') {
                continue;
            }

            lines.emplace_back(str);
        }

        file.close();

        ASSERT_EQ(num_neurons, lines.size());

        std::vector<bool> is_there(num_neurons);

        for (auto j = 0; j < num_neurons; j++) {
            const Vec3d& desired_position = positions[j];
            const std::string& desired_area_name = area_names[j];
            const SignalType& desired_signal_type = types[j];

            const std::string& current_line = lines[j];

            std::stringstream sstream(current_line);

            size_t id;
            double x;
            double y;
            double z;
            std::string area;
            std::string type_string;

            sstream
                >> id
                >> x
                >> y
                >> z
                >> area
                >> type_string;

            ASSERT_TRUE(0 < id);
            ASSERT_TRUE(id <= num_neurons);

            ASSERT_FALSE(is_there[id - 1]);
            is_there[id - 1] = true;

            ASSERT_NEAR(x, desired_position.get_x(), eps);
            ASSERT_NEAR(y, desired_position.get_y(), eps);
            ASSERT_NEAR(z, desired_position.get_z(), eps);

            SignalType type;
            if (type_string == "ex") {
                type = SignalType::EXCITATORY;
            } else if (type_string == "in") {
                type = SignalType::INHIBITORY;
            } else {
                ASSERT_TRUE(false);
            }

            ASSERT_TRUE(area == desired_area_name);
            ASSERT_TRUE(type == desired_signal_type);
        }
    }
}

TEST_F(NeuronAssignmentTest, test_reloading) {
    std::uniform_int_distribution<size_t> uid(1, 1000);
    std::uniform_real_distribution<double> urd(0.0, 1.0);

    mt.seed(rand());

    auto part = std::make_shared<Partition>(1, 0);

    for (auto i = 0; i < iterations; i++) {
        std::vector<Vec3d> positions;
        std::vector<std::string> area_names;
        std::vector<SignalType> types;

        generate_neuron_positions(positions, area_names, types, mt);

        auto num_neurons = positions.size();

        SubdomainFromFile sff{ "neurons.tmp", {}, part };
        part->set_total_num_neurons(sff.get_total_num_neurons_in_file());

        const auto box_length = sff.get_simulation_box_length().get_maximum();

        sff.fill_subdomain(0, 1, Vec3d{ 0 }, Vec3d{ box_length });

        std::vector<Vec3d> loaded_positions = sff.neuron_positions(0, 1, Vec3d{ 0 }, Vec3d{ box_length });
        std::vector<std::string> loaded_area_names = sff.neuron_area_names(0, 1, Vec3d{ 0 }, Vec3d{ box_length });
        std::vector<SignalType> loaded_types = sff.neuron_types(0, 1, Vec3d{ 0 }, Vec3d{ box_length });

        for (auto j = 0; j < num_neurons; j++) {
            const auto& curr_pos = positions[j];
            const auto& curr_loaded_pos = loaded_positions[j];

            ASSERT_NEAR(curr_pos.get_x(), curr_loaded_pos.get_x(), eps);
            ASSERT_NEAR(curr_pos.get_y(), curr_loaded_pos.get_y(), eps);
            ASSERT_NEAR(curr_pos.get_z(), curr_loaded_pos.get_z(), eps);

            const auto& curr_name = area_names[j];
            const auto& curr_loaded_name = loaded_area_names[j];

            ASSERT_EQ(curr_name, curr_loaded_name);

            const auto& curr_type = types[j];
            const auto& curr_loaded_type = loaded_types[j];

            ASSERT_EQ(curr_type, curr_loaded_type);
        }
    }
}

TEST_F(NeuronAssignmentTest, test_reloading_multiple) {
    std::uniform_int_distribution<size_t> uid(1, 1000);
    std::uniform_real_distribution<double> urd(0.0, 1.0);

    mt.seed(rand());

    auto part = std::make_shared<Partition>(1, 0);

    for (auto i = 0; i < iterations; i++) {
        std::vector<Vec3d> positions;
        std::vector<std::string> area_names;
        std::vector<SignalType> types;

        generate_neuron_positions(positions, area_names, types, mt);

        auto num_neurons = positions.size();

        auto part = std::make_shared<Partition>(1, 0);
        SubdomainFromFile sff{ "neurons.tmp", {}, part };

        part->set_total_num_neurons(sff.get_total_num_neurons_in_file());

        std::array<std::vector<Vec3d>, 8> loaded_positions;
        std::array<std::vector<std::string>, 8> loaded_area_names;
        std::array<std::vector<SignalType>, 8> loaded_types;

        std::vector<Vec3<size_t>> indices;
        indices.emplace_back(0, 0, 0);
        indices.emplace_back(0, 0, 1);
        indices.emplace_back(0, 1, 0);
        indices.emplace_back(1, 0, 0);
        indices.emplace_back(0, 1, 1);
        indices.emplace_back(1, 0, 1);
        indices.emplace_back(1, 1, 0);
        indices.emplace_back(1, 1, 1);

        const auto box_length = sff.get_simulation_box_length().get_maximum();

        std::vector<Vec3d> total_loaded_positions;
        std::vector<std::string> total_loaded_area_names;
        std::vector<SignalType> total_loaded_types;

        for (auto j = 0; j < 8; j++) {
            const auto& idx = indices[j];

            Vec3d min{ 0 };
            Vec3d max{ 0 };

            std::tie(min, max) = sff.get_subdomain_boundaries(idx, { 2, 2, 2 });

            sff.fill_subdomain(j, 8, min, max);

            loaded_positions[j] = sff.neuron_positions(j, 8, min, max);
            loaded_area_names[j] = sff.neuron_area_names(j, 8, min, max);
            loaded_types[j] = sff.neuron_types(j, 8, min, max);

            total_loaded_positions.insert(total_loaded_positions.end(), loaded_positions[j].begin(), loaded_positions[j].end());
            total_loaded_area_names.insert(total_loaded_area_names.end(), loaded_area_names[j].begin(), loaded_area_names[j].end());
            total_loaded_types.insert(total_loaded_types.end(), loaded_types[j].begin(), loaded_types[j].end());
        }

        ASSERT_EQ(num_neurons, total_loaded_positions.size());
        ASSERT_EQ(num_neurons, total_loaded_area_names.size());
        ASSERT_EQ(num_neurons, total_loaded_types.size());

        std::vector<int> random_sequence;
        sort_indices(positions, random_sequence);

        std::vector<int> total_loaded_sequence;
        sort_indices(total_loaded_positions, total_loaded_sequence);

        for (auto j = 0; j < num_neurons; j++) {
            auto random_idx = random_sequence[j];
            auto total_loaded_idx = total_loaded_sequence[j];

            const auto& curr_pos = positions[random_idx];
            const auto& curr_loaded_pos = total_loaded_positions[total_loaded_idx];

            ASSERT_NEAR(curr_pos.get_x(), curr_loaded_pos.get_x(), eps);
            ASSERT_NEAR(curr_pos.get_y(), curr_loaded_pos.get_y(), eps);
            ASSERT_NEAR(curr_pos.get_z(), curr_loaded_pos.get_z(), eps);

            const auto& curr_name = area_names[random_idx];
            const auto& curr_loaded_name = total_loaded_area_names[total_loaded_idx];

            ASSERT_EQ(curr_name, curr_loaded_name);

            const auto& curr_type = types[random_idx];
            const auto& curr_loaded_type = total_loaded_types[total_loaded_idx];

            ASSERT_EQ(curr_type, curr_loaded_type);
        }
    }
}

TEST_F(NeuronAssignmentTest, test_neuron_placement_store_and_load) {
    const std::string file{ "./test_output_positions_id.txt" };

    constexpr auto subdomain_id = 0;
    constexpr auto num_neurons = 10;
    constexpr auto frac_neurons_exc = 0.5;

    auto part = std::make_shared<Partition>(1, 0);
    // create from density
    SubdomainFromNeuronDensity sdnd{ num_neurons, frac_neurons_exc, 26, part };
    // fill_subdomain
    sdnd.fill_subdomain(subdomain_id, 1, Vec3d{ 0 }, Vec3d{ sdnd.get_simulation_box_length().get_maximum() });
    // save to file
    sdnd.write_neurons_to_file(file);

    // load from file
    SubdomainFromFile sdff{ file, {}, part };
    part->set_total_num_neurons(sdff.get_total_num_neurons_in_file());
    // fill_subdomain from file
    sdff.fill_subdomain(subdomain_id, 1, Vec3d{ 0 }, Vec3d{ sdff.get_simulation_box_length().get_maximum() });

    // check neuron placement numbers
    ASSERT_EQ(sdff.desired_num_neurons(), sdnd.desired_num_neurons());
    ASSERT_EQ(sdff.placed_num_neurons(), sdnd.placed_num_neurons());
    ASSERT_EQ(sdff.desired_ratio_neurons_exc(), sdnd.desired_ratio_neurons_exc());
    ASSERT_EQ(sdff.placed_ratio_neurons_exc(), sdnd.placed_ratio_neurons_exc());

    // check simulation_box_length
    // sdnd sets a box size in which it places neurons via estimation
    // sdff reads the file and uses a box size which fits the maximum position of any neuron, in which the neurons fit
    ASSERT_LE(sdff.get_simulation_box_length().get_x(), sdnd.get_simulation_box_length().get_x());
    ASSERT_LE(sdff.get_simulation_box_length().get_y(), sdnd.get_simulation_box_length().get_y());
    ASSERT_LE(sdff.get_simulation_box_length().get_z(), sdnd.get_simulation_box_length().get_z());

    // check for same number of subdomains
    ASSERT_EQ(sdff.get_nodes(subdomain_id).size(), sdnd.get_nodes(subdomain_id).size());

    // compare both neurons_in_subdomain maps for differences
    std::vector<NeuronToSubdomainAssignment::Node> diff{};
    std::set_symmetric_difference(std::begin(sdff.get_nodes(subdomain_id)), std::end(sdff.get_nodes(subdomain_id)),
        std::begin(sdnd.get_nodes(subdomain_id)), std::end(sdnd.get_nodes(subdomain_id)),
        std::back_inserter(diff));
    ASSERT_EQ(diff.size(), 0);

    // compare the written files of sdnd and sdff
    std::ifstream saved1{ file };
    sdff.write_neurons_to_file("./test_output_positions_id2.txt");
    std::ifstream saved2{ "./test_output_positions_id2.txt" };

    for (std::string line1{}, line2{}; std::getline(saved1, line1) && std::getline(saved2, line2);) {
        ASSERT_EQ(line1, line2);
    }
}
