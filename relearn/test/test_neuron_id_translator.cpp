#include "../googletest/include/gtest/gtest.h"

#include "RelearnTest.hpp"

#include "../source/sim/random/RandomNeuronIdTranslator.h"
#include "../source/sim/NeuronIdTranslator.h"
#include "../source/structure/Partition.h"
#include "../source/util/RelearnException.h"

#include <numeric>
#include <vector>

TEST_F(NeuronIdTest, testRandomNeuronIdTranslatorLocalGlobal) {
    const auto num_ranks = get_adjusted_random_number_ranks();
    const auto num_subdomains = round_to_next_exponent(num_ranks, 8);
    const auto my_subdomains = num_subdomains / num_ranks;

    std::vector<std::vector<size_t>> number_local_neurons(num_ranks);
    size_t number_total_neurons = 0;

    for (auto my_rank = 0; my_rank < num_ranks; my_rank++) {
        number_local_neurons[my_rank] = std::vector<size_t>(my_subdomains);

        for (auto my_subdomain = 0; my_subdomain < my_subdomains; my_subdomain++) {
            const auto num_local_neurons = get_random_number_neurons();

            number_local_neurons[my_rank][my_subdomain] = num_local_neurons;
            number_total_neurons += num_local_neurons;
        }
    }

    std::vector<std::shared_ptr<Partition>> partitions{};
    std::vector<size_t> number_neurons(num_ranks);

    for (auto my_rank = 0; my_rank < num_ranks; my_rank++) {
        const auto& local_neurons = number_local_neurons[my_rank];

        std::vector<size_t> local_ids_start(my_subdomains, 0);
        std::vector<size_t> local_ids_ends(my_subdomains, 0);

        for (auto my_subdomain = 0; my_subdomain < my_subdomains; my_subdomain++) {
            if (my_subdomain > 0) {
                local_ids_start[my_subdomain] = local_ids_ends[my_subdomain - 1] + 1;
            }

            local_ids_ends[my_subdomain] = local_ids_start[my_subdomain] + local_neurons[my_subdomain] - 1;
        }

        auto partition = std::make_shared<Partition>(num_ranks, my_rank);
        partition->set_total_number_neurons(number_total_neurons);
        partition->set_subdomain_number_neurons(local_neurons);

        const auto num_local_neurons = std::reduce(local_neurons.begin(), local_neurons.end(), size_t(0));
        number_neurons[my_rank] = num_local_neurons;

        partitions.emplace_back(std::move(partition));
    }

    std::vector<size_t> start_ids(num_ranks, 0);
    std::vector<size_t> end_ids(num_ranks, 0);
    std::partial_sum(number_neurons.begin(), number_neurons.end(), end_ids.begin(), std::plus<double>());

    for (auto rank_id = 0; rank_id < num_ranks; rank_id++) {
        start_ids[rank_id] = end_ids[rank_id] - number_neurons[rank_id];
    }

    auto translator = [number_neurons](size_t num_local_neurons) -> std::vector<size_t> {
        return number_neurons;
    };

    for (auto rank_id = 0; rank_id < num_ranks; rank_id++) {
        auto partition = partitions[rank_id];

        RandomNeuronIdTranslator rnit(partition, translator);

        const auto number_local_neurons = number_neurons[rank_id];
        const auto start_local_ids = start_ids[rank_id];

        for (auto neuron_id = 0; neuron_id < number_local_neurons; neuron_id++) {
            const auto golden_global_id = neuron_id + start_local_ids;
            const auto global_id = rnit.get_global_id(neuron_id);

            ASSERT_EQ(global_id, golden_global_id);

            const auto local_id = rnit.get_local_id(golden_global_id);

            ASSERT_EQ(local_id, neuron_id);
        }
    }
}

TEST_F(NeuronIdTest, testRandomNeuronIdTranslatorLocalGlobalException) {
    const auto num_ranks = get_adjusted_random_number_ranks();
    const auto num_subdomains = round_to_next_exponent(num_ranks, 8);
    const auto my_subdomains = num_subdomains / num_ranks;

    std::vector<std::vector<size_t>> number_local_neurons(num_ranks);
    size_t number_total_neurons = 0;

    for (auto my_rank = 0; my_rank < num_ranks; my_rank++) {
        number_local_neurons[my_rank] = std::vector<size_t>(my_subdomains);

        for (auto my_subdomain = 0; my_subdomain < my_subdomains; my_subdomain++) {
            const auto num_local_neurons = get_random_number_neurons();

            number_local_neurons[my_rank][my_subdomain] = num_local_neurons;
            number_total_neurons += num_local_neurons;
        }
    }

    std::vector<std::shared_ptr<Partition>> partitions{};
    std::vector<size_t> number_neurons(num_ranks);

    for (auto my_rank = 0; my_rank < num_ranks; my_rank++) {
        const auto& local_neurons = number_local_neurons[my_rank];

        std::vector<size_t> local_ids_start(my_subdomains, 0);
        std::vector<size_t> local_ids_ends(my_subdomains, 0);

        for (auto my_subdomain = 0; my_subdomain < my_subdomains; my_subdomain++) {
            if (my_subdomain > 0) {
                local_ids_start[my_subdomain] = local_ids_ends[my_subdomain - 1] + 1;
            }

            local_ids_ends[my_subdomain] = local_ids_start[my_subdomain] + local_neurons[my_subdomain] - 1;
        }

        auto partition = std::make_shared<Partition>(num_ranks, my_rank);
        partition->set_total_number_neurons(number_total_neurons);
        partition->set_subdomain_number_neurons(local_neurons);

        const auto num_local_neurons = std::reduce(local_neurons.begin(), local_neurons.end(), size_t(0));
        number_neurons[my_rank] = num_local_neurons;

        partitions.emplace_back(std::move(partition));
    }

    std::vector<size_t> start_ids(num_ranks, 0);
    std::vector<size_t> end_ids(num_ranks, 0);
    std::partial_sum(number_neurons.begin(), number_neurons.end(), end_ids.begin(), std::plus<double>());

    for (auto rank_id = 0; rank_id < num_ranks; rank_id++) {
        start_ids[rank_id] = end_ids[rank_id] - number_neurons[rank_id];
    }

    auto translator = [number_neurons](size_t num_local_neurons) -> std::vector<size_t> {
        return number_neurons;
    };

    for (auto rank_id = 0; rank_id < num_ranks; rank_id++) {
        auto partition = partitions[rank_id];

        RandomNeuronIdTranslator rnit(partition, translator);

        const auto number_local_neurons = number_neurons[rank_id];
        const auto start_local_ids = start_ids[rank_id];
        const auto end_local_ids = end_ids[rank_id];

        for (auto counter = 0; counter < number_neurons_out_of_scope; counter++) {
            if (start_local_ids == 0) {
                break;
            }

            const auto too_small_id = get_random_integer<size_t>(0, start_local_ids - 1);
            ASSERT_THROW(rnit.get_local_id(too_small_id), RelearnException);
        }

        for (auto counter = 0; counter < number_neurons_out_of_scope; counter++) {
            const auto too_small_id = get_random_integer<size_t>(end_local_ids, end_local_ids + number_total_neurons);
            ASSERT_THROW(rnit.get_local_id(too_small_id), RelearnException);
        }

        for (auto counter = 0; counter < number_neurons_out_of_scope; counter++) {
            const auto too_large_global_id = get_random_integer<size_t>(number_local_neurons, number_local_neurons + number_total_neurons);
            ASSERT_THROW(rnit.get_global_id(too_large_global_id), RelearnException);
        }
    }
}

TEST_F(NeuronIdTest, testRandomNeuronIdTranslatorGlobalIdToRankNeuronId) {
    const auto num_ranks = get_adjusted_random_number_ranks();
    const auto num_subdomains = round_to_next_exponent(num_ranks, 8);
    const auto my_subdomains = num_subdomains / num_ranks;

    std::vector<std::vector<size_t>> number_local_neurons(num_ranks);
    size_t number_total_neurons = 0;

    for (auto my_rank = 0; my_rank < num_ranks; my_rank++) {
        number_local_neurons[my_rank] = std::vector<size_t>(my_subdomains);

        for (auto my_subdomain = 0; my_subdomain < my_subdomains; my_subdomain++) {
            const auto num_local_neurons = get_random_number_neurons();

            number_local_neurons[my_rank][my_subdomain] = num_local_neurons;
            number_total_neurons += num_local_neurons;
        }
    }

    std::vector<std::shared_ptr<Partition>> partitions{};
    std::vector<size_t> number_neurons(num_ranks);

    for (auto my_rank = 0; my_rank < num_ranks; my_rank++) {
        const auto& local_neurons = number_local_neurons[my_rank];

        std::vector<size_t> local_ids_start(my_subdomains, 0);
        std::vector<size_t> local_ids_ends(my_subdomains, 0);

        for (auto my_subdomain = 0; my_subdomain < my_subdomains; my_subdomain++) {
            if (my_subdomain > 0) {
                local_ids_start[my_subdomain] = local_ids_ends[my_subdomain - 1] + 1;
            }

            local_ids_ends[my_subdomain] = local_ids_start[my_subdomain] + local_neurons[my_subdomain] - 1;
        }

        auto partition = std::make_shared<Partition>(num_ranks, my_rank);
        partition->set_total_number_neurons(number_total_neurons);
        partition->set_subdomain_number_neurons(local_neurons);

        const auto num_local_neurons = std::reduce(local_neurons.begin(), local_neurons.end(), size_t(0));
        number_neurons[my_rank] = num_local_neurons;

        partitions.emplace_back(std::move(partition));
    }

    std::vector<size_t> start_ids(num_ranks, 0);
    std::vector<size_t> end_ids(num_ranks, 0);
    std::partial_sum(number_neurons.begin(), number_neurons.end(), end_ids.begin(), std::plus<double>());

    for (auto rank_id = 0; rank_id < num_ranks; rank_id++) {
        start_ids[rank_id] = end_ids[rank_id] - number_neurons[rank_id];
    }

    auto translator = [number_neurons](size_t num_local_neurons) -> std::vector<size_t> {
        return number_neurons;
    };

    std::vector<size_t> global_ids(number_total_neurons, 0);
    std::iota(global_ids.begin(), global_ids.end(), 0);

    for (auto rank_id = 0; rank_id < num_ranks; rank_id++) {
        auto partition = partitions[rank_id];

        RandomNeuronIdTranslator rnit(partition, translator);

        const auto number_local_neurons = number_neurons[rank_id];
        const auto start_local_ids = start_ids[rank_id];

        const auto& rnis = rnit.translate_global_ids(global_ids);
        ASSERT_EQ(rnis.size(), global_ids.size());

        for (const auto& neuron_id : global_ids) {
            ASSERT_TRUE(rnis.find(neuron_id) != rnis.end()) << neuron_id ;

            const auto& [rank, local_id] = rnis.at(neuron_id);

            ASSERT_LT(rank, num_ranks) << rank << ' ' << num_ranks;

            const auto remote_start = start_ids[rank];
            const auto expected_local_id = neuron_id - remote_start;

            ASSERT_EQ(local_id, expected_local_id) << rank << ' ' << expected_local_id << " vs. " << local_id << " for: " << neuron_id;
        }
    }
}
