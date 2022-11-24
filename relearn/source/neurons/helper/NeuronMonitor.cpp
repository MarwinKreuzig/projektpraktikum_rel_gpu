#include "NeuronMonitor.h"

#include "Config.h"
#include "io/LogFiles.h"
#include "mpi/MPIWrapper.h"
#include "neurons/LocalAreaTranslator.h"

#include <fstream>
#include <string>

void NeuronMonitor::init_print_file() {
    const auto& path = LogFiles::get_output_path() / "neuron_monitors";
    if (!std::filesystem::exists(path)) {
        std::filesystem::create_directories(path);
    }

    const auto& file_path = path / (MPIWrapper::get_my_rank_str() + '_' + std::to_string(target_neuron_id.get_neuron_id()) + ".csv");
    std::ofstream outfile(file_path, std::ios_base::out | std::ios_base::trunc);

    constexpr auto description = "# Step;Fired;Fired Fraction;x;Secondary Variable;Calcium;Target Calcium;Synaptic Input;Background Activity;Grown Axons;Connected Axons;Grown Excitatory Dendrites;Connected Excitatory Dendrites;Grown Inhibitory Dendrites;Connected Inhibitory Dendrites\n";

    outfile << std::setprecision(Constants::print_precision);
    outfile.imbue(std::locale());

    outfile << "# Rank: " << MPIWrapper::get_my_rank_str() << "\n";
    outfile << "# Neuron ID: " << target_neuron_id.get_neuron_id() << "\n";
    outfile << "# Area name: " << neurons_to_monitor->get_local_area_translator()->get_area_name_for_neuron_id(target_neuron_id.get_neuron_id()) << "\n";
    outfile << "# Area id: " << neurons_to_monitor->get_local_area_translator()->get_area_id_for_neuron_id(target_neuron_id.get_neuron_id()) << "\n";
    outfile << description;
}

void NeuronMonitor::flush_current_contents() {
    std::filesystem::path path = LogFiles::get_output_path() / "neuron_monitors";

    const auto& file_path = path / (MPIWrapper::get_my_rank_str() + '_' + std::to_string(target_neuron_id.get_neuron_id()) + ".csv");
    std::ofstream outfile(file_path, std::ios_base::ate | std::ios_base::app);

    constexpr auto filler = ";";

    auto current_step = static_cast<decltype(Config::monitor_step)>(0);
    for (const auto& info : informations) {
        outfile << current_step << filler;
        outfile << info.get_fired() << filler;
        outfile << info.get_fraction_fired() << filler;
        outfile << info.get_x() << filler;
        outfile << info.get_secondary() << filler;
        outfile << info.get_calcium() << filler;
        outfile << info.get_target_calcium() << filler;
        outfile << info.get_synaptic_input() << filler;
        outfile << info.get_background_activity() << filler;
        outfile << info.get_axons() << filler;
        outfile << info.get_axons_connected() << filler;
        outfile << info.get_excitatory_dendrites_grown() << filler;
        outfile << info.get_excitatory_dendrites_connected() << filler;
        outfile << info.get_inhibitory_dendrites_grown() << filler;
        outfile << info.get_inhibitory_dendrites_connected() << '\n';

        current_step += Config::monitor_step;
    }

    informations.clear();
}
