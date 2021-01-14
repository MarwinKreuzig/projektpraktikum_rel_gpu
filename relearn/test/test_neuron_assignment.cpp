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

#include "commons.h"

#include "../source/NeuronToSubdomainAssignment.h"
#include "../source/Partition.h"
#include "../source/SubdomainFromNeuronDensity.h"
#include "../source/SubdomainFromFile.h"
#include "../source/RelearnException.h"



void sort_indices(const std::vector<Vec3d>& vectors, std::vector<int>& sequence) {
	sequence = std::vector<int>(vectors.size());
	std::iota(sequence.begin(), sequence.end(), 0);

	std::stable_sort(sequence.begin(), sequence.end(),
		[&vectors](int i1, int i2) {return vectors[i1] < vectors[i2]; });
}

void check_types_fraction(std::vector<SynapticElements::SignalType>& types, double& frac_ex, unsigned long long total_subdomains, const size_t& num_neurons) {
	size_t neurons_ex = 0;
	size_t neurons_in = 0;

	for (auto& type : types) {
		if (type == SynapticElements::SignalType::EXCITATORY) {
			neurons_ex++;
		}
		else if (type == SynapticElements::SignalType::INHIBITORY) {
			neurons_in++;
		}
	}

	auto frac_ex_ = (static_cast<double>(neurons_ex) / static_cast<double>(neurons_ex + neurons_in));
	EXPECT_NEAR(frac_ex, frac_ex_, static_cast<double>(total_subdomains) / static_cast<double>(num_neurons));
}

void check_positions(std::vector<NeuronToSubdomainAssignment::Position>& pos, double um_per_neuron, size_t& expected_neurons_per_dimension, bool* flags) {
	std::vector<Vec3<size_t>> pos_fixed;
	for (auto& p : pos) {
		pos_fixed.emplace_back((Vec3<size_t>)(p * (1 / um_per_neuron)));
	}

	for (auto& p : pos_fixed) {
		auto x = p.x;
		auto y = p.y;
		auto z = p.z;

		EXPECT_LE(x, expected_neurons_per_dimension);
		EXPECT_LE(y, expected_neurons_per_dimension);
		EXPECT_LE(z, expected_neurons_per_dimension);

		auto idx = x * expected_neurons_per_dimension * expected_neurons_per_dimension + y * expected_neurons_per_dimension + z;
		auto& val = flags[idx];
		EXPECT_FALSE(val);
		val = true;
	}
}

void generate_neuron_positions(std::vector<Vec3d>& positions,
	std::vector<std::string>& area_names, std::vector<SynapticElements::SignalType>& types) {

	std::uniform_int_distribution<size_t> uid(1, 1000);
	std::uniform_real_distribution<double> urd(0.0, 1.0);

	auto num_neurons = uid(mt);
	auto frac_ex = urd(mt);
	auto um_per_neuron = urd(mt) * 100;

	SubdomainFromNeuronDensity sfnd{ num_neurons, frac_ex, um_per_neuron };

	auto num_neurons_ = sfnd.desired_num_neurons();

	EXPECT_EQ(num_neurons, num_neurons_);

	auto box_length = sfnd.simulation_box_length.get_maximum();

	sfnd.fill_subdomain(0, 1, Vec3d{ 0 }, Vec3d{ box_length });

	sfnd.neuron_positions(0, 1, Vec3d{ 0 }, Vec3d{ box_length }, positions);
	sfnd.neuron_area_names(0, 1, Vec3d{ 0 }, Vec3d{ box_length }, area_names);
	sfnd.neuron_types(0, 1, Vec3d{ 0 }, Vec3d{ box_length }, types);

	EXPECT_EQ(num_neurons, positions.size());
	EXPECT_EQ(num_neurons, area_names.size());
	EXPECT_EQ(num_neurons, types.size());

	sfnd.write_neurons_to_file("neurons.tmp");
}

TEST(TestRandomNeuronPlacement, test_constructor) {
	std::uniform_int_distribution<size_t> uid(1, 10000);
	std::uniform_real_distribution<double> urd(0.0, 1.0);

	mt.seed(rand());

	for (auto i = 0; i < iterations; i++) {
		auto num_neurons = uid(mt);
		auto frac_ex = urd(mt);
		auto um_per_neuron = urd(mt) * 100;

		SubdomainFromNeuronDensity sfnd{ num_neurons, frac_ex, um_per_neuron };

		auto num_neurons_ = sfnd.desired_num_neurons();
		auto frac_ex_ = sfnd.desired_ratio_neurons_exc();

		EXPECT_EQ(num_neurons, num_neurons_);

		EXPECT_NEAR(frac_ex, frac_ex_, 1.0 / num_neurons);

		EXPECT_NEAR(sfnd.get_simulation_box_length().x,
			ceil(pow(static_cast<double>(num_neurons), 1 / 3.)) * um_per_neuron,
			1.0 / num_neurons);

		EXPECT_EQ(0, sfnd.placed_num_neurons());
		EXPECT_EQ(0, sfnd.placed_ratio_neurons_exc());
	}
}

TEST(TestRandomNeuronPlacement, test_lazily_fill) {
	std::uniform_int_distribution<size_t> uid(1, 10000);
	std::uniform_real_distribution<double> urd(0.0, 1.0);

	mt.seed(rand());

	for (auto i = 0; i < iterations; i++) {
		auto num_neurons = uid(mt);
		auto frac_ex = urd(mt);
		auto um_per_neuron = urd(mt) * 100;

		auto lower_bound_ex = static_cast<size_t>(floor(num_neurons * frac_ex));
		auto upper_bound_ex = static_cast<size_t>(ceil(num_neurons * frac_ex));

		SubdomainFromNeuronDensity sfnd{ num_neurons, frac_ex, um_per_neuron };

		auto box_length = sfnd.simulation_box_length.get_maximum();

		EXPECT_GE(box_length, 0);

		sfnd.fill_subdomain(0, 1, Vec3d{ 0 }, Vec3d{ box_length });

		auto num_neurons_ = sfnd.desired_num_neurons();
		auto frac_ex_ = sfnd.desired_ratio_neurons_exc();

		EXPECT_EQ(num_neurons, num_neurons_);

		EXPECT_NEAR(frac_ex, frac_ex_, 1.0 / num_neurons);

		EXPECT_EQ(sfnd.desired_num_neurons(), sfnd.placed_num_neurons());
		EXPECT_NEAR(sfnd.desired_ratio_neurons_exc(), sfnd.placed_ratio_neurons_exc(), 1.0 / num_neurons);

		EXPECT_LE(sfnd.desired_ratio_neurons_exc(), sfnd.placed_ratio_neurons_exc());
	}
}

TEST(TestRandomNeuronPlacement, test_lazily_fill_multiple) {
	setup();

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

		SubdomainFromNeuronDensity sfnd{ num_neurons, frac_ex, um_per_neuron };

		auto box_length = sfnd.simulation_box_length.get_maximum();

		EXPECT_GE(box_length, 0);

		auto num_fills = uid_fills(mt);

		for (auto j = 0; j < num_fills; j++) {
			if (j >= 1) {
				EXPECT_THROW(sfnd.fill_subdomain(0, 1, Vec3d{ 0 }, Vec3d{ box_length }), RelearnException);
			}
			else {
				sfnd.fill_subdomain(0, 1, Vec3d{ 0 }, Vec3d{ box_length });
			}
		}

		auto num_neurons_ = sfnd.desired_num_neurons();
		auto frac_ex_ = sfnd.desired_ratio_neurons_exc();

		EXPECT_EQ(num_neurons, num_neurons_);

		EXPECT_NEAR(frac_ex, frac_ex_, 1.0 / num_neurons);

		EXPECT_EQ(sfnd.desired_num_neurons(), sfnd.placed_num_neurons());
		EXPECT_NEAR(sfnd.desired_ratio_neurons_exc(), sfnd.placed_ratio_neurons_exc(), 1.0 / num_neurons);

		EXPECT_LE(sfnd.desired_ratio_neurons_exc(), sfnd.placed_ratio_neurons_exc());
	}
}

TEST(TestRandomNeuronPlacement, test_lazily_fill_positions) {
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

		SubdomainFromNeuronDensity sfnd{ num_neurons, frac_ex, um_per_neuron };

		auto box_length = sfnd.simulation_box_length.get_maximum();

		sfnd.fill_subdomain(0, 1, Vec3d{ 0 }, Vec3d{ box_length });

		std::vector<Vec3d> pos;
		sfnd.neuron_positions(0, 1, Vec3d{ 0 }, Vec3d{ box_length }, pos);

		EXPECT_EQ(pos.size(), num_neurons);

		std::vector<Vec3<size_t>> pos_fixed;
		for (auto& p : pos) {
			pos_fixed.emplace_back((Vec3<size_t>)(p * (1 / um_per_neuron)));
		}

		for (auto& p : pos_fixed) {
			auto x = p.x;
			auto y = p.y;
			auto z = p.z;

			EXPECT_LE(x, expected_neurons_per_dimension);
			EXPECT_LE(y, expected_neurons_per_dimension);
			EXPECT_LE(z, expected_neurons_per_dimension);

			auto idx = x * expected_neurons_per_dimension * expected_neurons_per_dimension + y * expected_neurons_per_dimension + z;
			EXPECT_FALSE(flags[idx]);
			flags[idx] = true;
		}

		std::vector<SynapticElements::SignalType> types;
		sfnd.neuron_types(0, 1, Vec3d{ 0 }, Vec3d{ box_length }, types);

		size_t neurons_ex = 0;
		size_t neurons_in = 0;

		for (auto& type : types) {
			if (type == SynapticElements::SignalType::EXCITATORY) {
				neurons_ex++;
			}
			else if (type == SynapticElements::SignalType::INHIBITORY) {
				neurons_in++;
			}
		}

		auto frac_ex_ = (static_cast<double>(neurons_ex) / static_cast<double>(neurons_ex + neurons_in));
		EXPECT_NEAR(frac_ex, frac_ex_, 1.0 / static_cast<double>(num_neurons));
	}
}

TEST(TestRandomNeuronPlacement, test_lazily_fill_positions_multiple_subdomains) {
	setup();

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

		SubdomainFromNeuronDensity sfnd{ num_neurons, frac_ex, um_per_neuron };

		auto box_length = sfnd.simulation_box_length.get_maximum();

		auto subdomain_length_x = box_length / subdomains_x;
		auto subdomain_length_y = box_length / subdomains_y;
		auto subdomain_length_z = box_length / subdomains_z;

		sfnd.fill_subdomain(0, 1, Vec3d{ 0 }, Vec3d{ box_length });

		std::vector<Vec3d> pos;
		for (auto x_it = 0; x_it < subdomains_x; x_it++) {
			for (auto y_it = 0; y_it < subdomains_y; y_it++) {
				for (auto z_it = 0; z_it < subdomains_z; z_it++) {
					Vec3d subdomain_min{ x_it * subdomain_length_x, y_it * subdomain_length_y , z_it * subdomain_length_z };
					Vec3d subdomain_max{ (x_it + 1) * subdomain_length_x, (y_it + 1) * subdomain_length_y , (z_it + 1) * subdomain_length_z };

					auto current_idx = z_it + y_it * subdomains_z + x_it * subdomains_z * subdomains_y;

					if (x_it == 0 && y_it == 0 && z_it == 0) {
						sfnd.neuron_positions(current_idx, total_subdomains, subdomain_min, subdomain_max, pos);
					}
					else {
						EXPECT_THROW(sfnd.neuron_positions(current_idx, total_subdomains, subdomain_min, subdomain_max, pos), RelearnException);
					}
				}
			}
		}

		EXPECT_EQ(pos.size(), num_neurons);

		std::vector<Vec3<size_t>> pos_fixed;
		for (auto& p : pos) {
			pos_fixed.emplace_back((Vec3<size_t>)(p * (1 / um_per_neuron)));
		}

		for (auto& p : pos_fixed) {
			auto x = p.x;
			auto y = p.y;
			auto z = p.z;

			EXPECT_LE(x, expected_neurons_per_dimension);
			EXPECT_LE(y, expected_neurons_per_dimension);
			EXPECT_LE(z, expected_neurons_per_dimension);

			auto idx = x * expected_neurons_per_dimension * expected_neurons_per_dimension + y * expected_neurons_per_dimension + z;
			EXPECT_FALSE(flags[idx]);
			flags[idx] = true;
		}

		std::vector<SynapticElements::SignalType> types;
		sfnd.neuron_types(0, 1, Vec3d{ 0 }, Vec3d{ box_length }, types);

		size_t neurons_ex = 0;
		size_t neurons_in = 0;

		for (auto& type : types) {
			if (type == SynapticElements::SignalType::EXCITATORY) {
				neurons_ex++;
			}
			else if (type == SynapticElements::SignalType::INHIBITORY) {
				neurons_in++;
			}
		}

		auto frac_ex_ = (static_cast<double>(neurons_ex) / static_cast<double>(neurons_ex + neurons_in));
		EXPECT_NEAR(frac_ex, frac_ex_, static_cast<double>(total_subdomains) / static_cast<double>(num_neurons));
	}
}

TEST(TestRandomNeuronPlacement, test_multiple_lazily_fill_positions_multiple_subdomains) {
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

		SubdomainFromNeuronDensity sfnd{ num_neurons, frac_ex, um_per_neuron };

		auto box_length = sfnd.simulation_box_length.get_maximum();

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
					sfnd.neuron_positions(current_idx, total_subdomains, subdomain_min, subdomain_max, pos);
				}
			}
		}

		auto diff_neurons = std::max(pos.size(), num_neurons) - std::min(pos.size(), num_neurons);
		EXPECT_LE(diff_neurons, total_subdomains);

		check_positions(pos, um_per_neuron, expected_neurons_per_dimension, flags);

		std::vector<SynapticElements::SignalType> types;
		sfnd.neuron_types(0, 1, Vec3d{ 0 }, Vec3d{ box_length }, types);

		check_types_fraction(types, frac_ex, total_subdomains, num_neurons);

		free(flags);
	}
}

TEST(TestRandomNeuronPlacement, test_saving) {
	std::uniform_int_distribution<size_t> uid(1, 1000);
	std::uniform_real_distribution<double> urd(0.0, 1.0);

	mt.seed(rand());

	for (auto i = 0; i < iterations; i++) {
		std::vector<Vec3d> positions;
		std::vector<std::string> area_names;
		std::vector<SynapticElements::SignalType> types;

		generate_neuron_positions(positions, area_names, types);

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

		EXPECT_EQ(num_neurons, lines.size());

		std::vector<bool> is_there(num_neurons);

		for (auto j = 0; j < num_neurons; j++) {
			const Vec3d& desired_position = positions[j];
			const std::string& desired_area_name = area_names[j];
			const SynapticElements::SignalType& desired_signal_type = types[j];

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

			EXPECT_TRUE(0 < id);
			EXPECT_TRUE(id <= num_neurons);

			EXPECT_FALSE(is_there[id]);
			is_there[id] = true;

			EXPECT_NEAR(x, desired_position.x, eps);
			EXPECT_NEAR(y, desired_position.y, eps);
			EXPECT_NEAR(z, desired_position.z, eps);

			SynapticElements::SignalType type;
			if (type_string == "ex") {
				type = SynapticElements::SignalType::EXCITATORY;
			}
			else if (type_string == "in") {
				type = SynapticElements::SignalType::INHIBITORY;
			}
			else {
				EXPECT_TRUE(false);
			}

			EXPECT_TRUE(area == desired_area_name);
			EXPECT_TRUE(type == desired_signal_type);
		}
	}
}

TEST(TestRandomNeuronPlacement, test_reloading) {
	std::uniform_int_distribution<size_t> uid(1, 1000);
	std::uniform_real_distribution<double> urd(0.0, 1.0);

	mt.seed(rand());

	auto part = std::make_shared<Partition>(1, 0);

	for (auto i = 0; i < iterations; i++) {
		std::vector<Vec3d> positions;
		std::vector<std::string> area_names;
		std::vector<SynapticElements::SignalType> types;

		generate_neuron_positions(positions, area_names, types);

		auto num_neurons = positions.size();

		SubdomainFromFile sff{ "neurons.tmp", part };

		std::vector<Vec3d> loaded_positions;
		std::vector<std::string> loaded_area_names;
		std::vector<SynapticElements::SignalType> loaded_types;

		const auto box_length = sff.get_simulation_box_length().get_maximum();

		sff.fill_subdomain(0, 1, Vec3d{ 0 }, Vec3d{ box_length });

		sff.neuron_positions(0, 1, Vec3d{ 0 }, Vec3d{ box_length }, loaded_positions);
		sff.neuron_area_names(0, 1, Vec3d{ 0 }, Vec3d{ box_length }, loaded_area_names);
		sff.neuron_types(0, 1, Vec3d{ 0 }, Vec3d{ box_length }, loaded_types);

		for (auto j = 0; j < num_neurons; j++) {
			const auto& curr_pos = positions[j];
			const auto& curr_loaded_pos = loaded_positions[j];

			EXPECT_NEAR(curr_pos.x, curr_loaded_pos.x, eps);
			EXPECT_NEAR(curr_pos.y, curr_loaded_pos.y, eps);
			EXPECT_NEAR(curr_pos.z, curr_loaded_pos.z, eps);

			const auto& curr_name = area_names[j];
			const auto& curr_loaded_name = loaded_area_names[j];

			EXPECT_EQ(curr_name, curr_loaded_name);

			const auto& curr_type = types[j];
			const auto& curr_loaded_type = loaded_types[j];

			EXPECT_EQ(curr_type, curr_loaded_type);
		}
	}
}

TEST(TestRandomNeuronPlacement, test_reloading_multiple) {
	std::uniform_int_distribution<size_t> uid(1, 1000);
	std::uniform_real_distribution<double> urd(0.0, 1.0);

	mt.seed(rand());

	auto part = std::make_shared<Partition>(1, 0);

	for (auto i = 0; i < iterations; i++) {
		std::vector<Vec3d> positions;
		std::vector<std::string> area_names;
		std::vector<SynapticElements::SignalType> types;

		generate_neuron_positions(positions, area_names, types);

		auto num_neurons = positions.size();

		SubdomainFromFile sff{ "neurons.tmp", part };

		std::array<std::vector<Vec3d>, 8> loaded_positions;
		std::array<std::vector<std::string>, 8> loaded_area_names;
		std::array<std::vector<SynapticElements::SignalType>, 8> loaded_types;

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
		std::vector<SynapticElements::SignalType> total_loaded_types;

		for (auto j = 0; j < 8; j++) {
			const auto& idx = indices[j];

			Vec3d min{ 0 };
			Vec3d max{ 0 };

			std::tie(min, max) = sff.get_subdomain_boundaries(idx, { 2, 2, 2 });

			sff.fill_subdomain(j, 8, min, max);

			sff.neuron_positions(j, 8, min, max, loaded_positions[j]);
			sff.neuron_area_names(j, 8, min, max, loaded_area_names[j]);
			sff.neuron_types(j, 8, min, max, loaded_types[j]);

			total_loaded_positions.insert(total_loaded_positions.end(), loaded_positions[j].begin(), loaded_positions[j].end());
			total_loaded_area_names.insert(total_loaded_area_names.end(), loaded_area_names[j].begin(), loaded_area_names[j].end());
			total_loaded_types.insert(total_loaded_types.end(), loaded_types[j].begin(), loaded_types[j].end());
		}

		EXPECT_EQ(num_neurons, total_loaded_positions.size());
		EXPECT_EQ(num_neurons, total_loaded_area_names.size());
		EXPECT_EQ(num_neurons, total_loaded_types.size());

		std::vector<int> random_sequence;
		sort_indices(positions, random_sequence);

		std::vector<int> total_loaded_sequence;
		sort_indices(total_loaded_positions, total_loaded_sequence);

		for (auto j = 0; j < num_neurons; j++) {
			auto random_idx = random_sequence[j];
			auto total_loaded_idx = total_loaded_sequence[j];

			const auto& curr_pos = positions[random_idx];
			const auto& curr_loaded_pos = total_loaded_positions[total_loaded_idx];

			EXPECT_NEAR(curr_pos.x, curr_loaded_pos.x, eps);
			EXPECT_NEAR(curr_pos.y, curr_loaded_pos.y, eps);
			EXPECT_NEAR(curr_pos.z, curr_loaded_pos.z, eps);

			const auto& curr_name = area_names[random_idx];
			const auto& curr_loaded_name = total_loaded_area_names[total_loaded_idx];

			EXPECT_EQ(curr_name, curr_loaded_name);

			const auto& curr_type = types[random_idx];
			const auto& curr_loaded_type = total_loaded_types[total_loaded_idx];

			EXPECT_EQ(curr_type, curr_loaded_type);
		}
	}
}




bool operator<(const NeuronToSubdomainAssignment::Node& a, const NeuronToSubdomainAssignment::Node& b) {
	return NeuronToSubdomainAssignment::Node::less()(a, b);
}

TEST(TestNeuronPlacementStoreLoad, test_neuron_placement_store_and_load) {
	const std::string file{ "./test_output_positions_id.txt" };

	constexpr auto subdomain_id = 0;
	constexpr auto num_neurons = 10;
	constexpr auto frac_neurons_exc = 0.5;

	// create from density
	SubdomainFromNeuronDensity sdnd{ num_neurons, frac_neurons_exc, 26 };
	// fill_subdomain
	sdnd.fill_subdomain(subdomain_id, 1, Vec3d{ 0 }, Vec3d{ sdnd.simulation_box_length.get_maximum() });
	// save to file
	sdnd.write_neurons_to_file(file);

	auto part = std::make_shared<Partition>(1, 0);
	// load from file
	SubdomainFromFile sdff{ file, part };
	// fill_subdomain from file
	sdff.fill_subdomain(subdomain_id, 1, Vec3d{ 0 }, Vec3d{ sdff.simulation_box_length.get_maximum() });

	// check neuron placement numbers
	EXPECT_EQ(sdff.desired_num_neurons(), sdnd.desired_num_neurons());
	EXPECT_EQ(sdff.placed_num_neurons(), sdnd.placed_num_neurons());
	EXPECT_EQ(sdff.desired_ratio_neurons_exc(), sdnd.desired_ratio_neurons_exc());
	EXPECT_EQ(sdff.placed_ratio_neurons_exc(), sdnd.placed_ratio_neurons_exc());

	// check simulation_box_length
	// sdnd sets a box size in which it places neurons via estimation
	// sdff reads the file and uses a box size which fits the maximum position of any neuron, in which the neurons fit
	EXPECT_LE(sdff.get_simulation_box_length().x, sdnd.get_simulation_box_length().x);
	EXPECT_LE(sdff.get_simulation_box_length().y, sdnd.get_simulation_box_length().y);
	EXPECT_LE(sdff.get_simulation_box_length().z, sdnd.get_simulation_box_length().z);

	// check for same number of subdomains
	EXPECT_EQ(sdff.neurons_in_subdomain.size(), sdnd.neurons_in_subdomain.size());

	// compare both neurons_in_subdomain maps for differences
	std::vector<decltype(sdff.neurons_in_subdomain)::value_type> diff{};
	std::set_symmetric_difference(std::begin(sdff.neurons_in_subdomain), std::end(sdff.neurons_in_subdomain),
		std::begin(sdnd.neurons_in_subdomain), std::end(sdnd.neurons_in_subdomain),
		std::back_inserter(diff));
	EXPECT_EQ(diff.size(), 0);

	// compare the written files of sdnd and sdff
	std::ifstream saved1{ file };
	sdff.write_neurons_to_file("./test_output_positions_id2.txt");
	std::ifstream saved2{ "./test_output_positions_id2.txt" };

	for (std::string line1{}, line2{}; std::getline(saved1, line1) && std::getline(saved2, line2);) {
		EXPECT_EQ(line1, line2);
	}
}
