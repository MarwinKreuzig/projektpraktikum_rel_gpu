#pragma once

/*
 * This file is part of the RELeARN software developed at Technical University Darmstadt
 *
 * Copyright (c) 2020, Technical University of Darmstadt, Germany
 *
 * This software may be modified and distributed under the terms of a BSD-style license.
 * See the LICENSE file in the base directory for details.
 *
 */

#include "RelearnTest.hpp"
#include "neurons/Neurons.h"
#include "neurons/input/SynapticInputCalculators.h"
#include "neurons/input/BackgroundActivityCalculators.h"

class NeuronsTest : public RelearnTest {
protected:
    static Neurons create_neurons_object(std::shared_ptr<Partition>& partition) {
        auto model = std::make_unique<models::PoissonModel>(models::PoissonModel::default_h,
                                                            std::make_unique<LinearSynapticInputCalculator>(SynapticInputCalculator::default_conductance),
                                                            std::make_unique<NullBackgroundActivityCalculator>(),
                                                            std::make_unique<Stimulus>(),
                                                            models::PoissonModel::default_x_0,
                                                            models::PoissonModel::default_tau_x,
                                                            models::PoissonModel::default_refractory_period);
        auto calcium = std::make_unique<CalciumCalculator>();
        calcium->set_initial_calcium_calculator([](MPIRank /*mpi_rank*/, NeuronID::value_type /*neuron_id*/) { return 0.0; });
        calcium->set_target_calcium_calculator([](MPIRank /*mpi_rank*/, NeuronID::value_type /*neuron_id*/) { return 0.0; });
        auto dends_ex = std::make_unique<SynapticElements>(ElementType::Dendrite, 0.2);
        auto dends_in = std::make_unique<SynapticElements>(ElementType::Dendrite, 0.2);
        auto axs = std::make_unique<SynapticElements>(ElementType::Axon, 0.2);

        Neurons neurons{ partition, std::move(model), std::move(calcium), std::move(axs), std::move(dends_ex), std::move(dends_in) };
        return neurons;
    }
};
