#include "../googletest/include/gtest/gtest.h"

#include <cmath>
#include <random>

#include "RelearnTest.hpp"

#include "../source/structure/Partition.h"
#include "../source/util/RelearnException.h"

constexpr const int upper_bound_my_rank = 32;
constexpr const int upper_bound_num_ranks = 32;

size_t round_to_next_exponent(size_t numToRound, size_t exponent) {
    auto log = std::log(static_cast<double>(numToRound)) / std::log(static_cast<double>(exponent));
    auto rounded_exp = std::ceil(log);
    auto new_val = std::pow(static_cast<double>(exponent), rounded_exp);
    return static_cast<size_t>(new_val);
}

TEST_F(PartitionTest, test_partition_constructor_arguments) {
    std::uniform_int_distribution<size_t> uid_my_rank(0, upper_bound_my_rank);
    std::uniform_int_distribution<size_t> uid_num_ranks(0, upper_bound_num_ranks);

    for (auto i = 0; i < iterations; i++) {
        auto my_rank = uid_my_rank(mt);
        auto num_ranks = uid_num_ranks(mt);

        auto exponent = std::log2(static_cast<double>(num_ranks));
        auto floored_exponent = std::floor(exponent);
        auto exp_diff = exponent - floored_exponent;

        if (my_rank >= num_ranks || exp_diff > eps) {
            ASSERT_THROW(Partition part(num_ranks, my_rank), RelearnException);
        } else {
            ASSERT_NO_THROW(Partition part(num_ranks, my_rank));
        }
    }
}
TEST_F(PartitionTest, test_partition_constructor) {
    std::uniform_int_distribution<size_t> uid_num_ranks(1, upper_bound_num_ranks);

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

            ASSERT_EQ(partition.get_total_num_subdomains(), num_subdomains);
            ASSERT_EQ(partition.get_my_num_subdomains(), my_subdomains);

            ASSERT_EQ(partition.get_my_subdomain_id_start(), my_subdomains * my_rank);
            ASSERT_EQ(partition.get_my_subdomain_id_end(), my_subdomains * (my_rank + 1) - 1);

            ASSERT_EQ(partition.get_level_of_subdomain_trees(), oct_exponent);
            ASSERT_EQ(partition.get_num_subdomains_per_dimension(), num_subdomains_per_dim);

            ASSERT_THROW(auto err = partition.is_neuron_local(0), RelearnException);
            ASSERT_THROW(auto err = partition.get_my_num_neurons(), RelearnException);
            ASSERT_THROW(auto err = partition.get_simulation_box_size(), RelearnException);
            ASSERT_THROW(auto err = partition.get_mpi_rank_from_pos(min), RelearnException);
            ASSERT_THROW(auto err = partition.get_global_id(0), RelearnException);
            ASSERT_THROW(auto err = partition.get_local_id(0), RelearnException);
        }
    }
}

