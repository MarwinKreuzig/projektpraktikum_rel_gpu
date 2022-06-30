#include "gtest/gtest.h"

#include "RelearnTest.hpp"

#include "algorithm/Connector.h"
#include "mpi/CommunicationMap.h"
#include "neurons/models/SynapticElements.h"

#include <map>
#include <vector>

TEST_F(ConnectorTest, testForwardConnectorExceptions) {
    const auto number_neurons_1 = get_random_number_neurons();
    const auto number_neurons_2 = get_random_number_neurons();

    const auto final_number_neurons = number_neurons_1 == number_neurons_2 ? number_neurons_2 + 1 : number_neurons_2;

    const auto number_ranks_1 = get_random_number_ranks() + 1;
    const auto number_ranks_2 = get_random_number_ranks() + 1;

    const auto final_number_ranks = number_ranks_1 == number_neurons_2 ? number_neurons_2 + 1 : number_ranks_2;

    const auto& excitatory_dendrites = create_dendrites(number_neurons_1, SignalType::Excitatory);
    const auto& inhibitory_dendrites = create_dendrites(final_number_neurons, SignalType::Inhibitory);

    std::shared_ptr<SynapticElements> empty = nullptr;

    CommunicationMap<SynapseCreationRequest> incoming_requests{ static_cast<int>(number_ranks_1) };

    ASSERT_THROW(auto val = ForwardConnector::process_requests(incoming_requests, empty, empty), RelearnException);
    ASSERT_THROW(auto val = ForwardConnector::process_requests(incoming_requests, excitatory_dendrites, empty), RelearnException);
    ASSERT_THROW(auto val = ForwardConnector::process_requests(incoming_requests, empty, inhibitory_dendrites), RelearnException);
    ASSERT_THROW(auto val = ForwardConnector::process_requests(incoming_requests, excitatory_dendrites, inhibitory_dendrites), RelearnException);

    CommunicationMap<SynapseCreationResponse> incoming_responses{ static_cast<int>(number_ranks_1) };
    ASSERT_THROW(auto val = ForwardConnector::process_responses(incoming_requests, incoming_responses, empty), RelearnException);

    CommunicationMap<SynapseCreationResponse> wrong_incoming_responses{ static_cast<int>(final_number_ranks) };
    ASSERT_THROW(auto val = ForwardConnector::process_responses(incoming_requests, wrong_incoming_responses, empty), RelearnException);
}

TEST_F(ConnectorTest, testForwardConnectorEmptyMap) {
    const auto number_neurons = get_random_number_neurons();
    const auto number_ranks = get_random_number_ranks() + 1;
    const auto current_rank = get_random_rank(number_ranks);

    const auto& excitatory_dendrites = create_dendrites(number_neurons, SignalType::Excitatory);
    const auto& inhibitory_dendrites = create_dendrites(number_neurons, SignalType::Inhibitory);

    // The following copies are intentional
    const auto previous_connected_excitatory = excitatory_dendrites->get_connected_elements();
    const auto previous_grown_excitatory = excitatory_dendrites->get_grown_elements();
    const auto previous_deltas_excitatory = excitatory_dendrites->get_deltas();
    const auto previous_types_excitatory = excitatory_dendrites->get_signal_types();

    const auto previous_connected_inhibitory = excitatory_dendrites->get_connected_elements();
    const auto previous_grown_inhibitory = excitatory_dendrites->get_grown_elements();
    const auto previous_deltas_inhibitory = excitatory_dendrites->get_deltas();
    const auto previous_types_inhibitory = excitatory_dendrites->get_signal_types();

    CommunicationMap<SynapseCreationRequest> incoming_requests{ static_cast<int>(number_ranks) };

    auto [responses, synapses] = ForwardConnector::process_requests(incoming_requests, excitatory_dendrites, inhibitory_dendrites);
    auto [local_synapses, distant_in_synapses] = synapses;

    ASSERT_EQ(responses.size(), incoming_requests.size());
    ASSERT_EQ(responses.get_number_ranks(), incoming_requests.get_number_ranks());
    ASSERT_EQ(responses.get_total_number_requests(), 0);

    ASSERT_TRUE(local_synapses.empty());
    ASSERT_TRUE(distant_in_synapses.empty());

    const auto& now_connected_excitatory = excitatory_dendrites->get_connected_elements();
    const auto& now_grown_excitatory = excitatory_dendrites->get_grown_elements();
    const auto& now_deltas_excitatory = excitatory_dendrites->get_deltas();
    const auto& now_types_excitatory = excitatory_dendrites->get_signal_types();

    const auto& now_connected_inhibitory = excitatory_dendrites->get_connected_elements();
    const auto& now_grown_inhibitory = excitatory_dendrites->get_grown_elements();
    const auto& now_deltas_inhibitory = excitatory_dendrites->get_deltas();
    const auto& now_types_inhibitory = excitatory_dendrites->get_signal_types();

    for (auto i = 0; i < number_neurons; i++) {
        ASSERT_EQ(previous_connected_excitatory[i], now_connected_excitatory[i]);
        ASSERT_EQ(previous_grown_excitatory[i], now_grown_excitatory[i]);
        ASSERT_EQ(previous_deltas_excitatory[i], now_deltas_excitatory[i]);
        ASSERT_EQ(previous_types_excitatory[i], now_types_excitatory[i]);

        ASSERT_EQ(previous_connected_inhibitory[i], now_connected_inhibitory[i]);
        ASSERT_EQ(previous_grown_inhibitory[i], now_grown_inhibitory[i]);
        ASSERT_EQ(previous_deltas_inhibitory[i], now_deltas_inhibitory[i]);
        ASSERT_EQ(previous_types_inhibitory[i], now_types_inhibitory[i]);
    }
}

TEST_F(ConnectorTest, testForwardConnectorMatchingRequests) {
    const auto number_neurons = get_random_number_neurons();
    const auto number_ranks = get_random_number_ranks() + 1;
    const auto current_rank = get_random_rank(number_ranks);

    const auto& excitatory_dendrites = create_dendrites(number_neurons, SignalType::Excitatory);
    const auto& inhibitory_dendrites = create_dendrites(number_neurons, SignalType::Inhibitory);

    CommunicationMap<SynapseCreationRequest> incoming_requests{ static_cast<int>(number_ranks) };

    auto number_excitatory_requests = 0;
    auto number_inhibitory_requests = 0;

    std::map<NeuronID, std::vector<SynapseCreationRequest>> excitatory_requests{};
    std::map<NeuronID, std::vector<SynapseCreationRequest>> inhibitory_requests{};

    for (const auto& id : NeuronID::range(number_neurons)) {
        const auto number_vacant_excitatory = excitatory_dendrites->get_free_elements(id);
        number_excitatory_requests += number_vacant_excitatory;

        for (auto i = 0U; i < number_vacant_excitatory; i++) {
            SynapseCreationRequest scr(id, NeuronID{ i }, SignalType::Excitatory);
            incoming_requests.append(1, scr);

            excitatory_requests[id].emplace_back(scr);
        }

        const auto number_vacant_inhibitory = inhibitory_dendrites->get_free_elements(id);
        number_inhibitory_requests += number_vacant_inhibitory;

        for (auto i = 0U; i < number_vacant_inhibitory; i++) {
            SynapseCreationRequest scr(id, NeuronID{ i }, SignalType::Inhibitory);
            incoming_requests.append(1, scr);

            inhibitory_requests[id].emplace_back(scr);
        }
    }

    auto [responses, synapses] = ForwardConnector::process_requests(incoming_requests, excitatory_dendrites, inhibitory_dendrites);
    auto [local_synapses, distant_in_synapses] = synapses;

    ASSERT_EQ(incoming_requests.size(), responses.size());

    const auto& request_sizes = incoming_requests.get_request_sizes();
    const auto& response_sizes = responses.get_request_sizes();

    for (auto i = 0; i < incoming_requests.size(); i++) {
        ASSERT_EQ(request_sizes[i], response_sizes[i]);
    }

    for (const auto& [rank, resps] : responses) {
        for (const auto resp : resps) {
            ASSERT_EQ(resp, SynapseCreationResponse::Succeeded);
        }
    }

    ASSERT_EQ(local_synapses.size(), 0);
    ASSERT_EQ(distant_in_synapses.size(), number_excitatory_requests + number_inhibitory_requests);

    for (const auto& id : NeuronID::range(number_neurons)) {
        const auto number_vacant_excitatory = excitatory_dendrites->get_free_elements(id);
        ASSERT_EQ(number_vacant_excitatory, 0);

        const auto number_vacant_inhibitory = inhibitory_dendrites->get_free_elements(id);
        ASSERT_EQ(number_vacant_inhibitory, 0);
    }

    for (const auto& [target_id, source_id, weight] : distant_in_synapses) {
        const auto& [source_rank, source_neuron_id] = source_id;

        ASSERT_EQ(source_rank, 1);
        ASSERT_EQ(std::abs(weight), 1);
    }
}

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
    const auto previous_connected_inhibitory_counts = inhibitory_dendrites->get_connected_elements();
    const auto previous_grown_inhibitory_counts = inhibitory_dendrites->get_grown_elements();

    const auto& [incoming_requests, number_excitatory_requests, number_inhibitory_requests]
        = create_incoming_requests(number_ranks, current_rank, number_neurons, 0, 9);

    auto [responses, synapses] = ForwardConnector::process_requests(incoming_requests, excitatory_dendrites, inhibitory_dendrites);
    auto [local_synapses, distant_in_synapses] = synapses;

    // There are as many requests as responses
    ASSERT_EQ(incoming_requests.size(), responses.size());

    const auto& request_sizes = incoming_requests.get_request_sizes();
    const auto& response_sizes = responses.get_request_sizes();

    // For each rank: The number of responses matches the number of requests
    for (auto i = 0; i < incoming_requests.size(); i++) {
        ASSERT_EQ(request_sizes[i], response_sizes[i]);
    }

    const auto& now_connected_excitatory_counts = excitatory_dendrites->get_connected_elements();
    const auto& now_grown_excitatory_counts = excitatory_dendrites->get_grown_elements();
    const auto& now_connected_inhibitory_counts = inhibitory_dendrites->get_connected_elements();
    const auto& now_grown_inhibitory_counts = inhibitory_dendrites->get_grown_elements();

    std::vector<unsigned int> newly_connected_excitatory_dendrites(number_neurons, 0);
    std::vector<unsigned int> newly_connected_inhibitory_dendrites(number_neurons, 0);

    // The grown elements did not change. There are now not less connected then before, and not more than grown
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

    // If there are still vacant elements, then all requests are connected
    for (auto i = 0; i < number_neurons; i++) {
        const auto vacant_excitatory_elements = excitatory_dendrites->get_free_elements(NeuronID{ i });
        if (vacant_excitatory_elements > 0) {
            ASSERT_EQ(newly_connected_excitatory_dendrites[i], number_excitatory_requests[i]);
        }

        const auto vacant_inhibitory_elements = inhibitory_dendrites->get_free_elements(NeuronID{ i });
        if (vacant_inhibitory_elements > 0) {
            ASSERT_EQ(newly_connected_inhibitory_dendrites[i], number_inhibitory_requests[i]);
        }
    }

    std::vector<unsigned int> accepted_excitatory_requests(number_neurons, 0);
    std::vector<unsigned int> accepted_inhibitory_requests(number_neurons, 0);

    LocalSynapses expected_local_synapses{};
    DistantInSynapses expected_distant_in_synapses{};

    const auto my_rank = MPIWrapper::get_my_rank();

    // Extract things from the return value
    for (auto rank = 0; rank < number_ranks; rank++) {
        for (auto index = 0; index < request_sizes[rank]; index++) {
            const auto& [target_index, source_index, signal_type] = incoming_requests.get_request(rank, index);
            const auto& response = responses.get_request(rank, index);

            if (response == SynapseCreationResponse::Failed) {
                continue;
            }

            const auto& target_id = target_index.get_neuron_id();
            const auto& source_id = source_index.get_neuron_id();

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

    // The sizes of the return values match the number of accepted responses
    ASSERT_EQ(local_synapses.size(), expected_local_synapses.size());
    ASSERT_EQ(distant_in_synapses.size(), expected_distant_in_synapses.size());

    std::sort(local_synapses.begin(), local_synapses.end());
    std::sort(distant_in_synapses.begin(), distant_in_synapses.end());
    std::sort(expected_local_synapses.begin(), expected_local_synapses.end());
    std::sort(expected_distant_in_synapses.begin(), expected_distant_in_synapses.end());

    // All and only the accepted local synapses are returned
    for (auto i = 0; i < local_synapses.size(); i++) {
        const auto& [target_1, source_1, weight_1] = local_synapses[i];
        const auto& [target_2, source_2, weight_2] = expected_local_synapses[i];

        ASSERT_EQ(target_1, target_2);
        ASSERT_EQ(source_1, source_2);
        ASSERT_EQ(weight_1, weight_2);
    }

    // All and only the accepted distant in synapses are returned
    for (auto i = 0; i < distant_in_synapses.size(); i++) {
        const auto& [target_1, source_1, weight_1] = distant_in_synapses[i];
        const auto& [target_2, source_2, weight_2] = expected_distant_in_synapses[i];

        ASSERT_EQ(target_1, target_2);
        ASSERT_EQ(source_1, source_2);
        ASSERT_EQ(weight_1, weight_2);
    }

    for (auto i = 0; i < number_neurons; i++) {
        ASSERT_EQ(accepted_excitatory_requests[i], newly_connected_excitatory_dendrites[i]);
        ASSERT_EQ(accepted_inhibitory_requests[i], newly_connected_inhibitory_dendrites[i]);
    }
}
