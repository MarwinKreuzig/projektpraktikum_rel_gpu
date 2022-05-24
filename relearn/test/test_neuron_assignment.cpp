#include "../googletest/include/gtest/gtest.h"

#include <algorithm>
#include <fstream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>

#include "RelearnTest.hpp"

#include "../source/sim/NeuronToSubdomainAssignment.h"
#include "../source/structure/Partition.h"
#include "../source/sim/random/SubdomainFromNeuronDensity.h"
#include "../source/sim/random/SubdomainFromNeuronPerRank.h"
#include "../source/sim/file/SubdomainFromFile.h"

void NeuronAssignmentTest::generate_neuron_positions(std::vector<Vec3d>& positions,
    std::vector<std::string>& area_names, std::vector<SignalType>& types) {

    const auto number_neurons = get_random_number_neurons();
    const auto fraction_excitatory_neurons = get_random_percentage();
    const auto um_per_neuron = get_random_percentage() * 100;

    const auto part = std::make_shared<Partition>(1, 0);
    SubdomainFromNeuronDensity sfnd{ number_neurons, fraction_excitatory_neurons, um_per_neuron, part };

    sfnd.initialize();

    positions = sfnd.get_neuron_positions_in_subdomain(0, 1);
    area_names = sfnd.get_neuron_area_names_in_subdomain(0, 1);
    types = sfnd.get_neuron_types_in_subdomain(0, 1);

    sfnd.write_neurons_to_file("neurons.tmp");
}

void NeuronAssignmentTest::generate_synapses(std::vector<std::tuple<NeuronID, NeuronID, int>>& synapses, size_t number_neurons) {
    const auto number_synapses = get_random_number_synapses();

    std::map<std::pair<NeuronID, NeuronID>, int> synapse_map{};

    for (auto i = 0; i < number_synapses; i++) {
        const auto source = get_random_neuron_id(number_neurons);
        const auto target = get_random_neuron_id(number_neurons);
        const auto weight = get_random_synapse_weight();

        synapse_map[{ source, target }] += weight;
    }

    synapses.resize(0);

    for (const auto& [pair, weight] : synapse_map) {
        const auto& [source, target] = pair;
        if (weight != 0) {
            synapses.emplace_back(source, target, weight);
        }
    }
}

double calculate_excitatory_fraction(const std::vector<SignalType>& types) {
    auto number_excitatory = 0;
    auto number_inhibitory = 0;

    for (const auto& type : types) {
        if (type == SignalType::Excitatory) {
            number_excitatory++;
        } else {
            number_inhibitory++;
        }
    }

    const auto ratio = static_cast<double>(number_excitatory) / static_cast<double>(number_excitatory + number_inhibitory);
    return ratio;
}

void write_synapses_to_file(const std::vector<std::tuple<NeuronID, NeuronID, int>>& synapses, std::filesystem::path path) {
    std::ofstream of(path);

    for (const auto& [target, source, weight] : synapses) {
        of << (target.get_local_id() + 1) << ' ' << (source.get_local_id() + 1) << ' ' << weight << '\n';
    }
}

TEST_F(NeuronAssignmentTest, testDensityTooFewNeurons) {
    const auto golden_number_ranks = get_adjusted_random_number_ranks();
    const auto golden_fraction_excitatory_neurons = get_random_percentage();
    const auto golden_um_per_neuron = get_random_percentage() * 100;

    for (auto rank = 0; rank < golden_number_ranks; rank++) {
        const auto part = std::make_shared<Partition>(golden_number_ranks, rank);
        ASSERT_THROW(SubdomainFromNeuronDensity sfnd(rank, golden_fraction_excitatory_neurons, golden_um_per_neuron, part), RelearnException);
    }
}

TEST_F(NeuronAssignmentTest, testDensityConstructorSingleSubdomain) {
    const auto golden_number_neurons = get_random_number_neurons() + 100;
    const auto golden_fraction_excitatory_neurons = get_random_percentage();
    const auto golden_um_per_neuron = get_random_percentage() * 100;

    const auto part = std::make_shared<Partition>(1, 0);
    SubdomainFromNeuronDensity sfnd{ golden_number_neurons, golden_fraction_excitatory_neurons, golden_um_per_neuron, part };

    const auto number_neurons = sfnd.get_requested_number_neurons();
    const auto fraction_excitatory_neurons = sfnd.get_requested_ratio_excitatory_neurons();

    ASSERT_EQ(golden_number_neurons, number_neurons);

    ASSERT_NEAR(golden_fraction_excitatory_neurons, fraction_excitatory_neurons, 1.0 / golden_number_neurons);

    const auto& [sim_box_min, sim_box_max] = part->get_simulation_box_size();
    const auto box_length = (sim_box_max - sim_box_min).get_maximum();

    const auto golden_box_length = calculate_box_length(golden_number_neurons, golden_um_per_neuron);
    ASSERT_NEAR(box_length, golden_box_length, 1.0 / golden_number_neurons);

    ASSERT_EQ(0, sfnd.get_number_placed_neurons());
    ASSERT_EQ(0, sfnd.get_ratio_placed_excitatory_neurons());
}

TEST_F(NeuronAssignmentTest, testDensityInitializeSingleSubdomain) {
    const auto golden_number_neurons = get_random_number_neurons() + 100;
    const auto golden_fraction_excitatory_neurons = get_random_percentage();
    const auto golden_um_per_neuron = get_random_percentage() * 100;

    const auto lower_bound_ex = static_cast<size_t>(floor(golden_number_neurons * golden_fraction_excitatory_neurons));
    const auto upper_bound_ex = static_cast<size_t>(ceil(golden_number_neurons * golden_fraction_excitatory_neurons));

    const auto part = std::make_shared<Partition>(1, 0);
    SubdomainFromNeuronDensity sfnd{ golden_number_neurons, golden_fraction_excitatory_neurons, golden_um_per_neuron, part };

    const auto& [sim_box_min, sim_box_max] = part->get_simulation_box_size();
    const auto box_length = (sim_box_max - sim_box_min).get_maximum();

    const auto golden_box_length = calculate_box_length(golden_number_neurons, golden_um_per_neuron);
    ASSERT_NEAR(box_length, golden_box_length, 1.0 / golden_number_neurons);

    sfnd.initialize();

    const auto requested_number_neurons = sfnd.get_requested_number_neurons();
    const auto placed_number_neurons = sfnd.get_number_placed_neurons();

    const auto requested_fraction_excitatory_neurons = sfnd.get_requested_ratio_excitatory_neurons();
    const auto placed_fraction_excitatory_neurons = sfnd.get_ratio_placed_excitatory_neurons();

    ASSERT_EQ(golden_number_neurons, requested_number_neurons);

    ASSERT_NEAR(golden_fraction_excitatory_neurons, requested_fraction_excitatory_neurons, 1.0 / golden_number_neurons);

    ASSERT_EQ(requested_number_neurons, placed_number_neurons);
    ASSERT_NEAR(requested_fraction_excitatory_neurons, placed_fraction_excitatory_neurons, 1.0 / golden_number_neurons);

    ASSERT_LE(requested_fraction_excitatory_neurons, placed_fraction_excitatory_neurons);
}

TEST_F(NeuronAssignmentTest, testDensityNeuronAttributesSizesSingleSubdomain) {
    const auto golden_number_neurons = get_random_number_neurons() + 100;
    const auto golden_fraction_excitatory_neurons = get_random_percentage();
    const auto golden_um_per_neuron = get_random_percentage() * 100;

    const auto part = std::make_shared<Partition>(1, 0);
    SubdomainFromNeuronDensity sfnd{ golden_number_neurons, golden_fraction_excitatory_neurons, golden_um_per_neuron, part };

    sfnd.initialize();

    const auto placed_number_neurons = sfnd.get_number_placed_neurons();
    const auto placed_fraction_excitatory_neurons = sfnd.get_ratio_placed_excitatory_neurons();

    const auto& positions = sfnd.get_neuron_positions_in_subdomain(0, 1);
    const auto& types = sfnd.get_neuron_types_in_subdomain(0, 1);
    const auto& area_names = sfnd.get_neuron_area_names_in_subdomain(0, 1);
    const auto placed_number_neurons_in_subdomain = sfnd.get_number_neurons_in_subdomain(0, 1);

    ASSERT_EQ(placed_number_neurons, placed_number_neurons_in_subdomain);
    ASSERT_EQ(placed_number_neurons, positions.size());
    ASSERT_EQ(placed_number_neurons, types.size());
    ASSERT_EQ(placed_number_neurons, area_names.size());

    const auto& all_positions = sfnd.get_neuron_positions_in_subdomains(0, 0, 1);
    const auto& all_types = sfnd.get_neuron_types_in_subdomains(0, 0, 1);
    const auto& all_area_names = sfnd.get_neuron_area_names_in_subdomains(0, 0, 1);
    const auto all_placed_neurons_in_subdomains = sfnd.get_number_neurons_in_subdomains(0, 0, 1);

    ASSERT_EQ(placed_number_neurons, all_placed_neurons_in_subdomains);
    ASSERT_EQ(placed_number_neurons, all_positions.size());
    ASSERT_EQ(placed_number_neurons, all_types.size());
    ASSERT_EQ(placed_number_neurons, all_area_names.size());
}

TEST_F(NeuronAssignmentTest, testDensityNeuronAttributesSemanticSingleSubdomain) {
    const auto golden_number_neurons = get_random_number_neurons() + 100;
    const auto golden_fraction_excitatory_neurons = get_random_percentage();
    const auto golden_um_per_neuron = get_random_percentage() * 100;

    const auto part = std::make_shared<Partition>(1, 0);
    SubdomainFromNeuronDensity sfnd{ golden_number_neurons, golden_fraction_excitatory_neurons, golden_um_per_neuron, part };

    sfnd.initialize();

    const auto placed_number_neurons_in_subdomain = sfnd.get_number_placed_neurons();
    const auto placed_ratio_excitatory_neurons = sfnd.get_ratio_placed_excitatory_neurons();

    const auto& positions = sfnd.get_neuron_positions_in_subdomain(0, 1);
    const auto& types = sfnd.get_neuron_types_in_subdomain(0, 1);
    const auto& area_names = sfnd.get_neuron_area_names_in_subdomain(0, 1);

    const auto calculated_ratio_excitatory_neurons = calculate_excitatory_fraction(types);
    ASSERT_NEAR(placed_ratio_excitatory_neurons, calculated_ratio_excitatory_neurons, 1.0 / golden_number_neurons);

    const auto& [sim_box_min, sim_box_max] = part->get_simulation_box_size();
    const auto neurons_per_dimension = pow(golden_number_neurons, 1. / 3);
    const auto number_boxes = static_cast<size_t>(ceil(neurons_per_dimension));

    std::vector<bool> box_full(number_boxes * number_boxes * number_boxes, false);

    for (const auto& position : positions) {
        ASSERT_TRUE(position.check_in_box(sim_box_min, sim_box_max));
        Vec3s cast_position = (Vec3s)(position / golden_um_per_neuron);

        const auto x = cast_position.get_x();
        const auto y = cast_position.get_y();
        const auto z = cast_position.get_z();

        ASSERT_LE(x, number_boxes);
        ASSERT_LE(y, number_boxes);
        ASSERT_LE(z, number_boxes);

        const auto flag = box_full[z * number_boxes * number_boxes + y * number_boxes + x];

        ASSERT_FALSE(flag);
        box_full[z * number_boxes * number_boxes + y * number_boxes + x] = true;
    }
}

TEST_F(NeuronAssignmentTest, testDensityWriteToFileSingleSubdomain) {
    std::vector<Vec3d> positions{};
    std::vector<std::string> area_names{};
    std::vector<SignalType> types{};

    generate_neuron_positions(positions, area_names, types);

    const auto number_neurons = positions.size();

    std::ifstream file("neurons.tmp", std::ios::binary | std::ios::in);

    std::vector<std::string> lines{};

    std::string str{};
    while (std::getline(file, str)) {
        if (str[0] == '#') {
            continue;
        }

        lines.emplace_back(str);
    }

    file.close();

    ASSERT_EQ(number_neurons, lines.size());

    std::vector<bool> is_there(number_neurons, false);

    for (auto j = 0; j < number_neurons; j++) {
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
        ASSERT_TRUE(id <= number_neurons);

        ASSERT_FALSE(is_there[id - 1]);
        is_there[id - 1] = true;

        ASSERT_NEAR(x, desired_position.get_x(), eps);
        ASSERT_NEAR(y, desired_position.get_y(), eps);
        ASSERT_NEAR(z, desired_position.get_z(), eps);

        SignalType type;
        if (type_string == "ex") {
            type = SignalType::Excitatory;
        } else if (type_string == "in") {
            type = SignalType::Inhibitory;
        } else {
            ASSERT_TRUE(false);
        }

        ASSERT_TRUE(area == desired_area_name);
        ASSERT_TRUE(type == desired_signal_type);
    }
}

TEST_F(NeuronAssignmentTest, testPerRankTooFewNeurons) {
    const auto golden_number_ranks = get_adjusted_random_number_ranks();
    const auto golden_fraction_excitatory_neurons = get_random_percentage();
    const auto golden_um_per_neuron = get_random_percentage() * 100;

    for (auto rank = 0; rank < golden_number_ranks; rank++) {
        const auto part = std::make_shared<Partition>(golden_number_ranks, rank);
        ASSERT_THROW(SubdomainFromNeuronPerRank sfnpr(0, golden_fraction_excitatory_neurons, golden_um_per_neuron, part), RelearnException);
    }
}

TEST_F(NeuronAssignmentTest, testPerRankSingleSubdomain) {
    const auto golden_number_neurons = get_random_number_neurons() + 100;
    const auto golden_fraction_excitatory_neurons = get_random_percentage();
    const auto golden_um_per_neuron = get_random_percentage() * 100;

    const auto part = std::make_shared<Partition>(1, 0);
    SubdomainFromNeuronPerRank sfnpr{ golden_number_neurons, golden_fraction_excitatory_neurons, golden_um_per_neuron, part };

    const auto number_neurons = sfnpr.get_requested_number_neurons();
    const auto fraction_excitatory_neurons = sfnpr.get_requested_ratio_excitatory_neurons();

    ASSERT_EQ(golden_number_neurons, number_neurons);

    ASSERT_NEAR(golden_fraction_excitatory_neurons, fraction_excitatory_neurons, 1.0 / golden_number_neurons);

    const auto& [sim_box_min, sim_box_max] = part->get_simulation_box_size();
    const auto box_length = (sim_box_max - sim_box_min).get_maximum();

    const auto golden_box_length = calculate_box_length(golden_number_neurons, golden_um_per_neuron);
    ASSERT_NEAR(box_length, golden_box_length, 1.0 / golden_number_neurons);

    ASSERT_EQ(0, sfnpr.get_number_placed_neurons());
    ASSERT_EQ(0, sfnpr.get_ratio_placed_excitatory_neurons());
}

TEST_F(NeuronAssignmentTest, testPerRankConstructorMultipleSubdomains) {
    const auto golden_number_ranks = get_adjusted_random_number_ranks();
    const auto number_subdomains = round_to_next_exponent(golden_number_ranks, 8);
    const auto golden_number_neurons = get_random_number_neurons() + number_subdomains * 50;
    const auto golden_fraction_excitatory_neurons = get_random_percentage();
    const auto golden_um_per_neuron = get_random_percentage() * 100;

    for (auto rank = 0; rank < golden_number_ranks; rank++) {
        const auto part = std::make_shared<Partition>(golden_number_ranks, rank);
        SubdomainFromNeuronPerRank sfnpr{ golden_number_neurons, golden_fraction_excitatory_neurons, golden_um_per_neuron, part };

        const auto number_neurons = sfnpr.get_requested_number_neurons();
        const auto fraction_excitatory_neurons = sfnpr.get_requested_ratio_excitatory_neurons();

        ASSERT_EQ(golden_number_neurons * golden_number_ranks, number_neurons);

        ASSERT_NEAR(golden_fraction_excitatory_neurons, fraction_excitatory_neurons, 1.0 / golden_number_neurons);

        const auto& [sim_box_min, sim_box_max] = part->get_simulation_box_size();
        const auto box_length = (sim_box_max - sim_box_min).get_maximum();

        const auto number_neurons_per_box_max = static_cast<size_t>(ceil(static_cast<double>(golden_number_neurons) / part->get_number_local_subdomains()));

        const auto golden_box_length = calculate_box_length(number_neurons_per_box_max, golden_um_per_neuron) * part->get_number_subdomains_per_dimension();
        ASSERT_NEAR(box_length, golden_box_length, 1.0 / golden_number_neurons);

        ASSERT_EQ(0, sfnpr.get_number_placed_neurons());
        ASSERT_EQ(0, sfnpr.get_ratio_placed_excitatory_neurons());
    }
}

TEST_F(NeuronAssignmentTest, testPerRankInitializeSingleSubdomain) {
    const auto golden_number_neurons = get_random_number_neurons() + 100;
    const auto golden_fraction_excitatory_neurons = get_random_percentage();
    const auto golden_um_per_neuron = get_random_percentage() * 100;

    const auto lower_bound_ex = static_cast<size_t>(floor(golden_number_neurons * golden_fraction_excitatory_neurons));
    const auto upper_bound_ex = static_cast<size_t>(ceil(golden_number_neurons * golden_fraction_excitatory_neurons));

    const auto part = std::make_shared<Partition>(1, 0);
    SubdomainFromNeuronPerRank sfnpr{ golden_number_neurons, golden_fraction_excitatory_neurons, golden_um_per_neuron, part };

    const auto& [sim_box_min, sim_box_max] = part->get_simulation_box_size();
    const auto box_length = (sim_box_max - sim_box_min).get_maximum();

    const auto golden_box_length = calculate_box_length(golden_number_neurons, golden_um_per_neuron);
    ASSERT_NEAR(box_length, golden_box_length, 1.0 / golden_number_neurons);

    sfnpr.initialize();

    const auto requested_number_neurons = sfnpr.get_requested_number_neurons();
    const auto placed_number_neurons = sfnpr.get_number_placed_neurons();

    const auto requested_fraction_excitatory_neurons = sfnpr.get_requested_ratio_excitatory_neurons();
    const auto placed_fraction_excitatory_neurons = sfnpr.get_ratio_placed_excitatory_neurons();

    ASSERT_EQ(golden_number_neurons, requested_number_neurons);

    ASSERT_NEAR(golden_fraction_excitatory_neurons, requested_fraction_excitatory_neurons, 1.0 / golden_number_neurons);

    ASSERT_EQ(requested_number_neurons, placed_number_neurons);
    ASSERT_NEAR(requested_fraction_excitatory_neurons, placed_fraction_excitatory_neurons, 1.0 / golden_number_neurons);

    ASSERT_LE(requested_fraction_excitatory_neurons, placed_fraction_excitatory_neurons);
}

TEST_F(NeuronAssignmentTest, testPerRankInitializeMultipleSubdomains) {
    const auto golden_number_ranks = get_adjusted_random_number_ranks();
    const auto number_subdomains = round_to_next_exponent(golden_number_ranks, 8);
    const auto golden_number_neurons = get_random_number_neurons() + number_subdomains * 50;
    const auto golden_fraction_excitatory_neurons = get_random_percentage();
    const auto golden_um_per_neuron = get_random_percentage() * 100;

    auto accumulated_placed_neurons = 0;
    auto accumulated_ratio_excitatory_neurons = 0.0;

    for (auto rank = 0; rank < golden_number_ranks; rank++) {
        const auto part = std::make_shared<Partition>(golden_number_ranks, rank);
        SubdomainFromNeuronPerRank sfnpr{ golden_number_neurons, golden_fraction_excitatory_neurons, golden_um_per_neuron, part };

        sfnpr.initialize();

        const auto placed_number_neurons = sfnpr.get_number_placed_neurons();
        const auto placed_ratio_excitatory_neurons = sfnpr.get_ratio_placed_excitatory_neurons();

        ASSERT_EQ(placed_number_neurons, golden_number_neurons);
        ASSERT_NEAR(golden_fraction_excitatory_neurons, placed_ratio_excitatory_neurons, static_cast<double>(number_subdomains) / golden_number_neurons);
    }
}

TEST_F(NeuronAssignmentTest, testPerRankNeuronAttributesSizesSingleSubdomain) {
    const auto golden_number_neurons = get_random_number_neurons() + 100;
    const auto golden_fraction_excitatory_neurons = get_random_percentage();
    const auto golden_um_per_neuron = get_random_percentage() * 100;

    const auto part = std::make_shared<Partition>(1, 0);
    SubdomainFromNeuronPerRank sfnpr{ golden_number_neurons, golden_fraction_excitatory_neurons, golden_um_per_neuron, part };

    sfnpr.initialize();

    const auto placed_number_neurons = sfnpr.get_number_placed_neurons();
    const auto placed_fraction_excitatory_neurons = sfnpr.get_ratio_placed_excitatory_neurons();

    const auto& positions = sfnpr.get_neuron_positions_in_subdomain(0, 1);
    const auto& types = sfnpr.get_neuron_types_in_subdomain(0, 1);
    const auto& area_names = sfnpr.get_neuron_area_names_in_subdomain(0, 1);
    const auto placed_number_neurons_in_subdomain = sfnpr.get_number_neurons_in_subdomain(0, 1);

    ASSERT_EQ(placed_number_neurons, placed_number_neurons_in_subdomain);
    ASSERT_EQ(placed_number_neurons, positions.size());
    ASSERT_EQ(placed_number_neurons, types.size());
    ASSERT_EQ(placed_number_neurons, area_names.size());

    const auto& all_positions = sfnpr.get_neuron_positions_in_subdomains(0, 0, 1);
    const auto& all_types = sfnpr.get_neuron_types_in_subdomains(0, 0, 1);
    const auto& all_area_names = sfnpr.get_neuron_area_names_in_subdomains(0, 0, 1);
    const auto all_placed_neurons_in_subdomains = sfnpr.get_number_neurons_in_subdomains(0, 0, 1);

    ASSERT_EQ(placed_number_neurons, all_placed_neurons_in_subdomains);
    ASSERT_EQ(placed_number_neurons, all_positions.size());
    ASSERT_EQ(placed_number_neurons, all_types.size());
    ASSERT_EQ(placed_number_neurons, all_area_names.size());
}

TEST_F(NeuronAssignmentTest, testPerRankNeuronAttributesSizeMultipleSubdomains) {
    const auto golden_number_ranks = get_adjusted_random_number_ranks();
    const auto number_subdomains = round_to_next_exponent(golden_number_ranks, 8);
    const auto number_subdomains_per_rank = number_subdomains / golden_number_ranks;
    const auto golden_number_neurons = get_random_number_neurons() + number_subdomains * 50;
    const auto golden_fraction_excitatory_neurons = get_random_percentage();
    const auto golden_um_per_neuron = get_random_percentage() * 100;

    size_t accumulated_placed_neurons = 0;

    for (auto rank = 0; rank < golden_number_ranks; rank++) {
        const auto part = std::make_shared<Partition>(golden_number_ranks, rank);
        SubdomainFromNeuronPerRank sfnpr{ golden_number_neurons, golden_fraction_excitatory_neurons, golden_um_per_neuron, part };

        sfnpr.initialize();

        const auto placed_number_neurons = sfnpr.get_number_placed_neurons();
        accumulated_placed_neurons += placed_number_neurons;

        const auto& all_positions = sfnpr.get_neuron_positions_in_subdomains(number_subdomains_per_rank * rank, number_subdomains_per_rank * rank + number_subdomains_per_rank - 1, number_subdomains);
        const auto& all_types = sfnpr.get_neuron_types_in_subdomains(number_subdomains_per_rank * rank, number_subdomains_per_rank * rank + number_subdomains_per_rank - 1, number_subdomains);
        const auto& all_area_names = sfnpr.get_neuron_area_names_in_subdomains(number_subdomains_per_rank * rank, number_subdomains_per_rank * rank + number_subdomains_per_rank - 1, number_subdomains);
        const auto all_placed_neurons_in_subdomains = sfnpr.get_number_neurons_in_subdomains(number_subdomains_per_rank * rank, number_subdomains_per_rank * rank + number_subdomains_per_rank - 1, number_subdomains);

        ASSERT_EQ(placed_number_neurons, all_placed_neurons_in_subdomains);
        ASSERT_EQ(placed_number_neurons, all_positions.size());
        ASSERT_EQ(placed_number_neurons, all_types.size());
        ASSERT_EQ(placed_number_neurons, all_area_names.size());

        size_t counter = 0;

        for (auto subdomain_id = 0; subdomain_id < number_subdomains_per_rank; subdomain_id++) {
            const auto& positions = sfnpr.get_neuron_positions_in_subdomain(number_subdomains_per_rank * rank + subdomain_id, number_subdomains);
            const auto& types = sfnpr.get_neuron_types_in_subdomain(number_subdomains_per_rank * rank + subdomain_id, number_subdomains);
            const auto& area_names = sfnpr.get_neuron_area_names_in_subdomain(number_subdomains_per_rank * rank + subdomain_id, number_subdomains);
            const auto placed_neurons_in_subdomain = sfnpr.get_number_neurons_in_subdomain(number_subdomains_per_rank * rank + subdomain_id, number_subdomains);

            ASSERT_EQ(placed_neurons_in_subdomain, positions.size());
            ASSERT_EQ(placed_neurons_in_subdomain, types.size());
            ASSERT_EQ(placed_neurons_in_subdomain, area_names.size());

            counter += placed_neurons_in_subdomain;
        }

        ASSERT_EQ(counter, placed_number_neurons);
    }

    ASSERT_EQ(accumulated_placed_neurons, golden_number_ranks * golden_number_neurons);
}

TEST_F(NeuronAssignmentTest, testPerRankNeuronAttributesSemanticSingleSubdomain) {
    const auto golden_number_neurons = get_random_number_neurons() + 100;
    const auto golden_fraction_excitatory_neurons = get_random_percentage();
    const auto golden_um_per_neuron = get_random_percentage() * 100;

    const auto part = std::make_shared<Partition>(1, 0);
    SubdomainFromNeuronPerRank sfnpr{ golden_number_neurons, golden_fraction_excitatory_neurons, golden_um_per_neuron, part };

    sfnpr.initialize();

    const auto placed_number_neurons_in_subdomain = sfnpr.get_number_placed_neurons();
    const auto placed_ratio_excitatory_neurons = sfnpr.get_ratio_placed_excitatory_neurons();

    const auto& positions = sfnpr.get_neuron_positions_in_subdomain(0, 1);
    const auto& types = sfnpr.get_neuron_types_in_subdomain(0, 1);
    const auto& area_names = sfnpr.get_neuron_area_names_in_subdomain(0, 1);

    const auto calculated_ratio_excitatory_neurons = calculate_excitatory_fraction(types);
    ASSERT_NEAR(placed_ratio_excitatory_neurons, calculated_ratio_excitatory_neurons, 1.0 / golden_number_neurons);

    const auto& [sim_box_min, sim_box_max] = part->get_simulation_box_size();
    const auto neurons_per_dimension = pow(golden_number_neurons, 1. / 3);
    const auto number_boxes = static_cast<size_t>(ceil(neurons_per_dimension));

    std::vector<bool> box_full(number_boxes * number_boxes * number_boxes, false);

    for (const auto& position : positions) {
        ASSERT_TRUE(position.check_in_box(sim_box_min, sim_box_max));
        Vec3s cast_position = (Vec3s)(position / golden_um_per_neuron);

        const auto x = cast_position.get_x();
        const auto y = cast_position.get_y();
        const auto z = cast_position.get_z();

        ASSERT_LE(x, number_boxes);
        ASSERT_LE(y, number_boxes);
        ASSERT_LE(z, number_boxes);

        const auto flag = box_full[z * number_boxes * number_boxes + y * number_boxes + x];

        ASSERT_FALSE(flag);
        box_full[z * number_boxes * number_boxes + y * number_boxes + x] = true;
    }
}

TEST_F(NeuronAssignmentTest, testPerRankNeuronAttributesSemanticMultipleSubdomains) {
    const auto golden_number_ranks = get_adjusted_random_number_ranks();
    const auto number_subdomains = round_to_next_exponent(golden_number_ranks, 8);
    const auto number_subdomains_per_rank = number_subdomains / golden_number_ranks;
    const auto golden_number_neurons = get_random_number_neurons() + number_subdomains * 50;
    const auto golden_fraction_excitatory_neurons = get_random_percentage();
    const auto golden_um_per_neuron = get_random_percentage() * 100;

    for (auto rank = 0; rank < golden_number_ranks; rank++) {
        const auto part = std::make_shared<Partition>(golden_number_ranks, rank);
        SubdomainFromNeuronPerRank sfnpr{ golden_number_neurons, golden_fraction_excitatory_neurons, golden_um_per_neuron, part };

        sfnpr.initialize();

        const auto placed_number_neurons = sfnpr.get_number_placed_neurons();
        const auto placed_ratio_excitatory_neurons = sfnpr.get_ratio_placed_excitatory_neurons();

        const auto& all_positions = sfnpr.get_neuron_positions_in_subdomains(number_subdomains_per_rank * rank, number_subdomains_per_rank * rank + number_subdomains_per_rank - 1, number_subdomains);
        const auto& all_types = sfnpr.get_neuron_types_in_subdomains(number_subdomains_per_rank * rank, number_subdomains_per_rank * rank + number_subdomains_per_rank - 1, number_subdomains);
        const auto all_placed_neurons_in_subdomains = sfnpr.get_number_neurons_in_subdomains(number_subdomains_per_rank * rank, number_subdomains_per_rank * rank + number_subdomains_per_rank - 1, number_subdomains);

        const auto calculated_ratio_excitatory_neurons = calculate_excitatory_fraction(all_types);
        ASSERT_NEAR(placed_ratio_excitatory_neurons, calculated_ratio_excitatory_neurons, static_cast<double>(number_subdomains) / golden_number_neurons) << golden_number_neurons;

        for (auto subdomain_id = 0; subdomain_id < number_subdomains_per_rank; subdomain_id++) {
            const auto& positions = sfnpr.get_neuron_positions_in_subdomain(number_subdomains_per_rank * rank + subdomain_id, number_subdomains);
            const auto& types = sfnpr.get_neuron_types_in_subdomain(number_subdomains_per_rank * rank + subdomain_id, number_subdomains);
            const auto placed_neurons_in_subdomain = sfnpr.get_number_neurons_in_subdomain(number_subdomains_per_rank * rank + subdomain_id, number_subdomains);

            const auto calculated_ratio_excitatory_neurons_subdomain = calculate_excitatory_fraction(types);
            ASSERT_NEAR(placed_ratio_excitatory_neurons, calculated_ratio_excitatory_neurons_subdomain, static_cast<double>(number_subdomains) / placed_neurons_in_subdomain) << golden_number_neurons;

            const auto& [subdomain_min, subdomain_max] = part->get_subdomain_boundaries(subdomain_id);

            const auto& subdomain_length = subdomain_max - subdomain_min;
            const auto& boxes_in_subdomain = subdomain_length / golden_um_per_neuron;

            const auto number_boxes = static_cast<size_t>(ceil(boxes_in_subdomain.calculate_p_norm(1.0)));

            std::vector<bool> box_full(number_boxes * number_boxes * number_boxes, false);

            for (const auto& position : positions) {
                ASSERT_TRUE(position.check_in_box(subdomain_min, subdomain_max));
                Vec3s cast_position = (Vec3s)((position - subdomain_min) / golden_um_per_neuron);

                const auto x = cast_position.get_x();
                const auto y = cast_position.get_y();
                const auto z = cast_position.get_z();

                ASSERT_LE(x, number_boxes);
                ASSERT_LE(y, number_boxes);
                ASSERT_LE(z, number_boxes);

                const auto flag = box_full[z * number_boxes * number_boxes + y * number_boxes + x];

                ASSERT_FALSE(flag);
                box_full[z * number_boxes * number_boxes + y * number_boxes + x] = true;
            }
        }
    }
}

TEST_F(NeuronAssignmentTest, testFileLoadSingleSubdomain) {
    std::vector<Vec3d> positions{};
    std::vector<std::string> area_names{};
    std::vector<SignalType> types{};

    generate_neuron_positions(positions, area_names, types);

    const auto number_neurons = positions.size();

    const auto part = std::make_shared<Partition>(1, 0);
    SubdomainFromFile sff{ "neurons.tmp", {}, part };

    sff.initialize();

    const auto& loaded_positions = sff.get_neuron_positions_in_subdomain(0, 1);
    const auto& loaded_area_names = sff.get_neuron_area_names_in_subdomain(0, 1);
    const auto& loaded_types = sff.get_neuron_types_in_subdomain(0, 1);

    for (auto j = 0; j < number_neurons; j++) {
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

TEST_F(NeuronAssignmentTest, testFileLoadNetworkSingleSubdomain) {
    std::vector<Vec3d> positions{};
    std::vector<std::string> area_names{};
    std::vector<SignalType> types{};

    generate_neuron_positions(positions, area_names, types);

    const auto number_neurons = positions.size();

    std::vector<std::tuple<NeuronID, NeuronID, int>> synapses{};

    generate_synapses(synapses, number_neurons);
    write_synapses_to_file(synapses, "synapses.tmp");

    const auto part = std::make_shared<Partition>(1, 0);
    SubdomainFromFile sff{ "neurons.tmp", "synapses.tmp", part };

    sff.initialize();

    const auto loader = sff.get_synapse_loader();

    const auto& [local_synapses, in_synapses, out_synapses] = loader->load_synapses();

    ASSERT_TRUE(in_synapses.empty());
    ASSERT_TRUE(out_synapses.empty());

    std::map<std::pair<NeuronID, NeuronID>, int> synapse_map{};

    for (const auto& [target, source, weight] : local_synapses) {
        synapse_map[{ target, source }] += weight;
    }

    for (const auto& [target, source, weight] : synapses) {
        synapse_map[{ target, source }] -= weight;
    }

    for (const auto& [_, weight] : synapse_map) {
        ASSERT_EQ(weight, 0);
    }
}

TEST_F(NeuronAssignmentTest, testFileRoi14SingleSubdomainONCE) {
#ifdef _WIN32
    std::filesystem::path path_to_neurons{ "../../input/roi_split/1-4/new_positions.txt" };
    std::optional<std::filesystem::path> path_to_synapses{ "../../input/roi_split/1-4/new_synapses.txt" };
#else
    std::filesystem::path path_to_neurons{ "../input/roi_split/1-4/new_positions.txt" };
    std::optional<std::filesystem::path> path_to_synapses{ "../input/roi_split/1-4/new_synapses.txt" };
#endif

    const auto part = std::make_shared<Partition>(1, 0);
    SubdomainFromFile sff{ path_to_neurons, path_to_synapses, part };

    sff.initialize();

    const auto& global_ids = sff.get_neuron_global_ids_in_subdomain(0, 1);
    const auto& positions = sff.get_neuron_positions_in_subdomain(0, 1);

    ASSERT_EQ(global_ids.size(), 426124);
    ASSERT_EQ(positions.size(), 426124);

    const auto sl = sff.get_synapse_loader();

    const auto& [local_synapses, in_synapses, out_synapses] = sl->load_synapses();

    ASSERT_TRUE(in_synapses.empty());
    ASSERT_TRUE(out_synapses.empty());

    std::vector<int> found_in_synapses(426124, 0);
    std::vector<int> found_out_synapses(426124, 0);

    for (const auto& [source_id, target_id, weight] : local_synapses) {
        found_in_synapses[target_id.get_local_id()] += weight;
        found_out_synapses[source_id.get_local_id()] += weight;
    }

    ASSERT_EQ(found_in_synapses.size(), 426124);
    ASSERT_EQ(found_out_synapses.size(), 426124);

    for (auto neuron_id = 0; neuron_id < 426124; neuron_id++) {
        ASSERT_EQ(found_in_synapses[neuron_id], 5) << neuron_id;
        ASSERT_EQ(found_out_synapses[neuron_id], 5) << neuron_id;
    }
}

TEST_F(NeuronAssignmentTest, testFileRoi15SingleSubdomainONCE) {
#ifdef _WIN32
    std::filesystem::path path_to_neurons{ "../../input/roi_split/1-5/new_positions.txt" };
    std::optional<std::filesystem::path> path_to_synapses{ "../../input/roi_split/1-5/new_synapses.txt" };
#else
    std::filesystem::path path_to_neurons{ "../input/roi_split/1-5/new_positions.txt" };
    std::optional<std::filesystem::path> path_to_synapses{ "../input/roi_split/1-5/new_synapses.txt" };
#endif

    const auto part = std::make_shared<Partition>(1, 0);
    SubdomainFromFile sff{ path_to_neurons, path_to_synapses, part };

    sff.initialize();

    const auto& global_ids = sff.get_neuron_global_ids_in_subdomain(0, 1);
    const auto& positions = sff.get_neuron_positions_in_subdomain(0, 1);

    ASSERT_EQ(global_ids.size(), 426124);
    ASSERT_EQ(positions.size(), 426124);

    const auto sl = sff.get_synapse_loader();

    const auto& [local_synapses, in_synapses, out_synapses] = sl->load_synapses();

    ASSERT_TRUE(in_synapses.empty());
    ASSERT_TRUE(out_synapses.empty());

    std::vector<int> found_in_synapses(426124, 0);
    std::vector<int> found_out_synapses(426124, 0);

    for (const auto& [source_id, target_id, weight] : local_synapses) {
        found_in_synapses[target_id.get_local_id()] += weight;
        found_out_synapses[source_id.get_local_id()] += weight;
    }

    ASSERT_EQ(found_in_synapses.size(), 426124);
    ASSERT_EQ(found_out_synapses.size(), 426124);

    for (auto neuron_id = 0; neuron_id < 426124; neuron_id++) {
        ASSERT_EQ(found_in_synapses[neuron_id], 6) << neuron_id;
        ASSERT_EQ(found_out_synapses[neuron_id], 6) << neuron_id;
    }
}

TEST_F(NeuronAssignmentTest, testFileRoi16SingleSubdomainONCE) {
#ifdef _WIN32
    std::filesystem::path path_to_neurons{ "../../input/roi_split/1-6/new_positions.txt" };
    std::optional<std::filesystem::path> path_to_synapses{ "../../input/roi_split/1-6/new_synapses.txt" };
#else
    std::filesystem::path path_to_neurons{ "../input/roi_split/1-6/new_positions.txt" };
    std::optional<std::filesystem::path> path_to_synapses{ "../input/roi_split/1-6/new_synapses.txt" };
#endif

    const auto part = std::make_shared<Partition>(1, 0);
    SubdomainFromFile sff{ path_to_neurons, path_to_synapses, part };

    sff.initialize();

    const auto& global_ids = sff.get_neuron_global_ids_in_subdomain(0, 1);
    const auto& positions = sff.get_neuron_positions_in_subdomain(0, 1);

    ASSERT_EQ(global_ids.size(), 426124);
    ASSERT_EQ(positions.size(), 426124);

    const auto sl = sff.get_synapse_loader();

    const auto& [local_synapses, in_synapses, out_synapses] = sl->load_synapses();

    ASSERT_TRUE(in_synapses.empty());
    ASSERT_TRUE(out_synapses.empty());

    std::vector<int> found_in_synapses(426124, 0);
    std::vector<int> found_out_synapses(426124, 0);

    for (const auto& [source_id, target_id, weight] : local_synapses) {
        found_in_synapses[target_id.get_local_id()] += weight;
        found_out_synapses[source_id.get_local_id()] += weight;
    }

    ASSERT_EQ(found_in_synapses.size(), 426124);
    ASSERT_EQ(found_out_synapses.size(), 426124);

    for (auto neuron_id = 0; neuron_id < 426124; neuron_id++) {
        ASSERT_EQ(found_in_synapses[neuron_id], 7) << neuron_id;
        ASSERT_EQ(found_out_synapses[neuron_id], 7) << neuron_id;
    }
}

TEST_F(NeuronAssignmentTest, testFileRoi17SingleSubdomainONCE) {
#ifdef _WIN32
    std::filesystem::path path_to_neurons{ "../../input/roi_split/1-7/new_positions.txt" };
    std::optional<std::filesystem::path> path_to_synapses{ "../../input/roi_split/1-7/new_synapses.txt" };
#else
    std::filesystem::path path_to_neurons{ "../input/roi_split/1-7/new_positions.txt" };
    std::optional<std::filesystem::path> path_to_synapses{ "../input/roi_split/1-7/new_synapses.txt" };
#endif

    const auto part = std::make_shared<Partition>(1, 0);
    SubdomainFromFile sff{ path_to_neurons, path_to_synapses, part };

    sff.initialize();

    const auto& global_ids = sff.get_neuron_global_ids_in_subdomain(0, 1);
    const auto& positions = sff.get_neuron_positions_in_subdomain(0, 1);

    ASSERT_EQ(global_ids.size(), 426124);
    ASSERT_EQ(positions.size(), 426124);

    const auto sl = sff.get_synapse_loader();

    const auto& [local_synapses, in_synapses, out_synapses] = sl->load_synapses();

    ASSERT_TRUE(in_synapses.empty());
    ASSERT_TRUE(out_synapses.empty());

    std::vector<int> found_in_synapses(426124, 0);
    std::vector<int> found_out_synapses(426124, 0);

    for (const auto& [source_id, target_id, weight] : local_synapses) {
        found_in_synapses[target_id.get_local_id()] += weight;
        found_out_synapses[source_id.get_local_id()] += weight;
    }

    ASSERT_EQ(found_in_synapses.size(), 426124);
    ASSERT_EQ(found_out_synapses.size(), 426124);

    for (auto neuron_id = 0; neuron_id < 426124; neuron_id++) {
        ASSERT_EQ(found_in_synapses[neuron_id], 8) << neuron_id;
        ASSERT_EQ(found_out_synapses[neuron_id], 8) << neuron_id;
    }
}
