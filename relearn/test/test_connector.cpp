#include "gtest/gtest.h"

#include "RelearnTest.hpp"

#include "algorithm/Connector.h"
#include "mpi/CommunicationMap.h"
#include "neurons/models/SynapticElements.h"

#include <map>
#include <vector>

TEST_F(ConnectorTest, testForwardConnectorIncoming) {
    const auto number_neurons = get_random_number_neurons();
    const auto number_ranks = get_random_number_ranks() + 1;
    const auto current_rank = get_random_rank(number_ranks);

    const auto& axons = create_axons(number_neurons);
    const auto& excitatory_dendrites = create_dendrites(number_neurons, SignalType::Excitatory);
    const auto& inhibitory_dendrites = create_dendrites(number_neurons, SignalType::Excitatory);

    // The following copies are intentional
    const auto previous_connected_excitatory_counts = excitatory_dendrites->get_connected_elements();
    const auto previous_grown_excitatory_counts = excitatory_dendrites->get_grown_elements();
    const auto previous_connected_inhibitory_counts = excitatory_dendrites->get_connected_elements();
    const auto previous_grown_inhibitory_counts = excitatory_dendrites->get_grown_elements();

    const auto& incoming_requests = create_incoming_requests(number_ranks, current_rank, number_neurons, 0, 9);

    auto [responses, synapses] = ForwardConnector::process_requests(incoming_requests, excitatory_dendrites, inhibitory_dendrites);
    auto [local_synapses, distant_in_synapses] = synapses;

    ASSERT_EQ(incoming_requests.size(), responses.size());

    const auto& request_sizes = incoming_requests.get_request_sizes();
    const auto& response_sizes = responses.get_request_sizes();

    for (auto i = 0; i < incoming_requests.size(); i++) {
        ASSERT_EQ(request_sizes[i], response_sizes[i]);
    }

    const auto& now_connected_excitatory_counts = excitatory_dendrites->get_connected_elements();
    const auto& now_grown_excitatory_counts = excitatory_dendrites->get_grown_elements();
    const auto& now_connected_inhibitory_counts = excitatory_dendrites->get_connected_elements();
    const auto& now_grown_inhibitory_counts = excitatory_dendrites->get_grown_elements();

    std::vector<unsigned int> newly_connected_excitatory_dendrites(number_neurons, 0);
    std::vector<unsigned int> newly_connected_inhibitory_dendrites(number_neurons, 0);

    for (auto i = 0; i < number_neurons; i++) {
        ASSERT_EQ(previous_grown_excitatory_counts[i], now_grown_excitatory_counts[i]);
        ASSERT_EQ(previous_grown_inhibitory_counts[i], now_grown_inhibitory_counts[i]);

        ASSERT_GE(now_connected_excitatory_counts[i], previous_connected_excitatory_counts[i]);
        ASSERT_GE(now_connected_inhibitory_counts[i], previous_connected_inhibitory_counts[i]);
                                                                                                              
        ASSERT_LE(now_connected_excitatory_counts[i], static_cast<unsigned int>(now_grown_excitatory_counts[i]));
        ASSERT_LE(now_connected_inhibitory_counts[i], static_cast<unsigned int>(now_grown_inhibitory_counts[i]));

        newly_connected_excitatory_dendrites[i] = now_connected_excitatory_counts[i] - previous_connected_excitatory_counts[i];
        newly_connected_inhibitory_dendrites[i] = now_connected_inhibitory_counts[i] - previous_connected_inhibitory_counts[i];
    }

    std::vector<unsigned int> accepted_excitatory_requests(number_neurons, 0);
    std::vector<unsigned int> accepted_inhibitory_requests(number_neurons, 0);

    LocalSynapses expected_local_synapses{};
    DistantInSynapses expected_distant_in_synapses{};

    const auto my_rank = MPIWrapper::get_my_rank();

    for (auto rank = 0; rank < number_ranks; rank++) {
        for (auto index = 0; index < request_sizes[rank]; index++) {
            const auto& [target_index, source_index, signal_type] = incoming_requests.get_request(rank, index);
            const auto& response = responses.get_request(rank, index);

            if (response == SynapseCreationResponse::Failed) {
                continue;
            }

            const auto& target_id = target_index.get_local_id();
            const auto& source_id = source_index.get_local_id();

            if (signal_type == SignalType::Excitatory) {
                accepted_excitatory_requests[target_id]++;
            } else {
                accepted_inhibitory_requests[target_id]++;
            }

            const auto weight = signal_type == SignalType::Excitatory ? 1 : -1;

            if (rank == my_rank) {
                expected_local_synapses.emplace_back(target_index, source_index, weight);
            } else {
                expected_distant_in_synapses.emplace_back(target_index, RankNeuronId{ rank, source_index }, weight);
            }
        }
    }

    ASSERT_EQ(local_synapses.size(), expected_local_synapses.size());
    ASSERT_EQ(distant_in_synapses.size(), expected_distant_in_synapses.size());

    std::sort(local_synapses.begin(), local_synapses.end());
    std::sort(distant_in_synapses.begin(), distant_in_synapses.end());
    std::sort(expected_local_synapses.begin(), expected_local_synapses.end());
    std::sort(expected_distant_in_synapses.begin(), expected_distant_in_synapses.end());

    for (auto i = 0; i < local_synapses.size(); i++) {
        const auto& [target_1, source_1, weight_1] = local_synapses[i];
        const auto& [target_2, source_2, weight_2] = expected_local_synapses[i];

        ASSERT_EQ(target_1, target_2);
        ASSERT_EQ(source_1, source_2);
        ASSERT_EQ(weight_1, weight_2);
    }

    for (auto i = 0; i < distant_in_synapses.size(); i++) {
        const auto& [target_1, source_1, weight_1] = distant_in_synapses[i];
        const auto& [target_2, source_2, weight_2] = expected_distant_in_synapses[i];

        ASSERT_EQ(target_1, target_2);
        ASSERT_EQ(source_1, source_2);
        ASSERT_EQ(weight_1, weight_2);
    }
}
