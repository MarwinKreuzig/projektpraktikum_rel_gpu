#include "../googletest/include/gtest/gtest.h"

#include <cmath>
#include <random>

#include "commons.h"

#include "../source/Partition.h"
#include "../source/RelearnException.h"

constexpr const int upper_bound_my_rank = 32;
constexpr const int upper_bount_num_ranks = 32;

size_t round_to_next_exponent(size_t numToRound, size_t exponent) {
	auto log = std::log(static_cast<double>(numToRound)) / std::log(static_cast<double>(exponent));
	auto rounded_exp = std::ceil(log);
	auto new_val = std::pow(static_cast<double>(exponent), rounded_exp);
	return static_cast<size_t>(new_val);
}

TEST(TestPartition, test_partition_constructor_arguments) {
	std::uniform_int_distribution<size_t> uid_my_rank(0, upper_bound_my_rank);
	std::uniform_int_distribution<size_t> uid_num_ranks(0, upper_bount_num_ranks);

	for (auto i = 0; i < iterations; i++) {
		auto my_rank = uid_my_rank(mt);
		auto num_ranks = uid_num_ranks(mt);

		auto exponent = std::log2(static_cast<double>(num_ranks));
		auto floored_exponent = std::floor(exponent);
		auto exp_diff = exponent - floored_exponent;

		if (my_rank >= num_ranks || exp_diff > eps) {
			EXPECT_THROW(Partition part(num_ranks, my_rank), RelearnException);
		}
		else {
			EXPECT_NO_THROW(Partition part(num_ranks, my_rank));
		}
	}
}

TEST(TestPartition, test_partition_constructor) {
	std::uniform_int_distribution<size_t> uid_num_ranks(0, upper_bount_num_ranks);

	for (auto i = 0; i < iterations; i++) {
		auto my_ranks_rand = uid_num_ranks(mt);
		auto num_ranks = round_to_next_exponent(my_ranks_rand, 2);
		auto num_subdomains = round_to_next_exponent(my_ranks_rand, 8);

		auto my_subdomains = num_subdomains / num_ranks;

		auto oct_exponent = static_cast<size_t>(std::log(static_cast<double>(num_subdomains)) / std::log(8.0));
		auto num_subdomains_per_dim = static_cast<size_t>(std::ceil(std::pow(static_cast<double>(num_subdomains), 1.0 / 3.0)));

		Vec3d min, max;

		for (auto my_rank = 0; my_rank < num_ranks; my_rank++) {
			Partition partition(num_ranks, my_rank);

			EXPECT_EQ(partition.get_total_num_subdomains(), num_subdomains);
			EXPECT_EQ(partition.get_my_num_subdomains(), my_subdomains);

			EXPECT_EQ(partition.get_my_subdomain_id_start(), my_subdomains * my_rank);
			EXPECT_EQ(partition.get_my_subdomain_id_end(), my_subdomains * (my_rank + 1) - 1);

			EXPECT_EQ(partition.get_level_of_subdomain_trees(), oct_exponent);
			EXPECT_EQ(partition.get_num_subdomains_per_dimension(), num_subdomains_per_dim);

			EXPECT_THROW(partition.is_neuron_local(0), RelearnException);
			EXPECT_THROW(partition.get_my_num_neurons(), RelearnException);
			EXPECT_THROW(partition.get_simulation_box_size(), RelearnException);
			EXPECT_THROW(partition.get_subdomain_tree(0), RelearnException);
			EXPECT_THROW(partition.get_subdomain_id_from_pos(min), RelearnException);
			EXPECT_THROW(partition.get_global_id(0), RelearnException);
			EXPECT_THROW(partition.get_local_id(0), RelearnException);
		}
	}
}


